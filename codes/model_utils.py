import warnings, math
import torch.nn as nn
import torch
from torch.autograd import Function
import torch.nn.functional as F
from torch.utils.data import Sampler
import numpy as np
from sklearn.utils import resample
from scipy.special import expit
from sklearn.metrics import roc_auc_score
import copy
import xgboost as xgb
from sklearn.pipeline import Pipeline
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics.pairwise import laplacian_kernel
from sklearn.svm import SVC
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import RandomizedSearchCV


warnings.filterwarnings("ignore")

device = (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

def bucketize_tensor(X):
    X_bucket = torch.zeros_like(X, dtype=torch.long)

    X_bucket[X == 0] = 0
    X_bucket[X == 1] = 1
    X_bucket[X >= 2] = 2

    return X_bucket

class SelfAttention(nn.Module):
    def __init__(self, hidden_dim, feature_dim, head = 1, dropout = 0.3):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=head, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim) # if key and query smaller dimension, manually implement attention
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, return_split=False): # (self, x, mutation_position, lambda_pos = 0.1)
        x = x.unsqueeze(1)

        key = self.key(x)
        query = self.query(x)
        value = self.value(x)
        attn_features, attn_weights = self.attention(query, key, value, need_weights = True, average_attn_weights=False)
        attn_features = self.norm1(self.dropout(attn_features) + x)
        
        return attn_features


class mut_model(nn.Module):
    def __init__(self, mut_dim, hidden_dim, feature_dim, interaction = False, monosex = None, monocancer = False, monotrt = False):
        super().__init__()
        self.interaction = interaction
        self.monosex = monosex
        self.cancer = monocancer
        self.monotrt = monotrt
        self.sex_emb = nn.Embedding(3, 4)
        if monosex is None:
            self.linear1 = nn.Linear(7, hidden_dim) # 7
        else:
            self.linear1 = nn.Linear(7, hidden_dim)#, feature_dim)

        self.norm = nn.LayerNorm(feature_dim)

        self.linear2 = nn.Linear(hidden_dim, feature_dim)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, mutation):
        mut_features = mutation[:, :-7]  
        X_bin = (mut_features > 0).float()
        count_feat = X_bin.sum(dim=1, keepdim=True)
        max_feat = mut_features.max(dim=1, keepdim=True).values
        mean_feat = mut_features.mean(dim=1, keepdim=True)
        count_feat = torch.log1p(count_feat)
        max_feat = torch.log1p(max_feat)
        mut_features_emb = torch.cat(
            [count_feat, max_feat, mean_feat],
            dim=1
        )
        mut_features_selected = mutation[:, -7:-3]

        cohort_type = mutation[:, -3].long()
        if self.monosex is None:
            # interactions
            mutation_interact = torch.cat([mut_features_selected, mut_features_emb], dim=1)
            feature = self.linear1(mutation_interact)
        else:
            mutation_interact = torch.cat([mut_features_selected, mut_features_emb], dim=1)
            feature = self.linear1(mutation_interact)           

        h = self.gelu(feature)
        feature = self.dropout(h + feature)
        feature = self.linear2(feature)
        feature = self.norm(feature)
        x = self.gelu(feature)
        return x, mutation[:, -1].long(), cohort_type


class gene_model(nn.Module):
    def __init__(self, mut_dim, hidden_dim, feature_dim, interaction = False, monosex = None, monocancer = False, monotrt = False):
        super().__init__()
        self.interaction = interaction
        self.monosex = monosex
        self.cancer = monocancer
        self.monotrt = monotrt
        self.cohort_emb = nn.Embedding(8, 4)
        self.sex_emb = nn.Embedding(3,4)
        self.gate_layer = nn.Linear(4, mut_dim-3)
        if interaction and (monosex is None):
            self.linear1 = nn.Linear(2*(mut_dim-3), hidden_dim) #nn.Linear(mut_dim-2, hidden_dim) 
        elif (monosex is not None) and interaction:
            self.linear1 = nn.Linear(2*(mut_dim-3), hidden_dim)#, hidden_dim)
        elif (monosex is not None) and not interaction:
            self.linear1 = nn.Linear(mut_dim-3, hidden_dim)#, hidden_dim)
        else:
            self.linear1 = nn.Linear(mut_dim-3, hidden_dim)#, hidden_dim)
        #self.mlp2 = nn.Linear(64, hidden_dim)

        self.attn_proj = nn.Linear(mut_dim-3, hidden_dim)
        self.pos_proj = nn.Linear(mut_dim, hidden_dim)
        self.attention = SelfAttention(hidden_dim, hidden_dim)#(mut_dim-2, mut_dim-2)
        self.norm = nn.LayerNorm(feature_dim)
        self.linear2 = nn.Linear(hidden_dim, feature_dim)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, mutation):

        mut_features = mutation[:, :-3]  
        
        sex = mutation[:, -1].long()
        cohort_vec = self.cohort_emb(mutation[:, -3].long())
        if torch.all(sex == 1):
            gate = torch.sigmoid(self.gate_layer(cohort_vec))
        else:
            gate = 1 + 0.5 * torch.tanh(self.gate_layer(cohort_vec))
        feat_x_cohort = mut_features * gate
        if self.interaction and (self.monosex is None):
            mutation_interact = torch.cat([mut_features, feat_x_cohort], dim = 1)
            feature = self.linear1(mutation_interact)
        elif self.interaction and (self.monosex is not None):
            mutation_interact = torch.cat([mut_features, feat_x_cohort], dim = 1)
            feature = self.linear1(mutation_interact)
        elif (self.monosex is not None) and not self.interaction:
            mut_features = mut_features
            feature = self.linear1(mut_features)
        else:
            mut_features = torch.cat([mut_features], dim = 1)
            feature = self.linear1(mut_features)

        h = self.gelu(feature)
        feature = self.dropout(h + feature)
        feature = self.linear2(feature)
        feature = self.norm(feature)
        x = self.gelu(feature)

        return x


# gradient reversal layer
class GradientReversalLayer(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None

class DomainDiscriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_domain):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),                      # IMPORTANT
            nn.Linear(hidden_dim, n_domain) # NO activation here
        )

    def forward(self, z, alpha=0.01):
        reversed_z = GradientReversalLayer.apply(z, alpha)
        return self.network(reversed_z)


class LoRALinear(nn.Module):
    def __init__(self, base_layer: nn.Linear, r: int = 8, alpha: float = 16.0,
                 dropout: float = 0.3, freeze_base: bool = True):
        super().__init__()
        assert isinstance(base_layer, nn.Linear)
        self.in_features  = base_layer.in_features
        self.out_features = base_layer.out_features
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        # Wrap the existing linear layer
        self.base = base_layer
        if freeze_base:
            for p in self.base.parameters():
                p.requires_grad = False

        # LoRA params
        self.lora_A = nn.Parameter(torch.zeros(r, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, r))

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)  # start with no effect

    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    def forward(self, x):
        base_out = self.base(x)
        x_d = self.dropout(x)
        lora_update = (x_d @ self.lora_A.t().to(device)) @ self.lora_B.t().to(device)
        return base_out + self.scaling * lora_update

def apply_lora_to_linears(module: nn.Module,
                          r: int = 8,
                          alpha: float = 16.0,
                          dropout: float = 0.3,
                          freeze_base: bool = True):
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            setattr(module, name, LoRALinear(child, r=r, alpha=alpha,
                                             dropout=dropout, freeze_base=freeze_base))
        else:
            apply_lora_to_linears(child, r=r, alpha=alpha,
                                  dropout=dropout, freeze_base=freeze_base)



# classification head, model for final feature prediction
class Classifier_fusion(nn.Module):
    def __init__(self, hidden_dim, feature_dim, dropout = 0.3, modality = 'gene+mut', n_cancer = 1, sex = None):
        super().__init__()
        
        self.proj = nn.Linear(feature_dim, hidden_dim)
        self.modality = modality
        self.sex = sex
        self.sex_emb = nn.Embedding(3,2)

        if modality == 'gene+mut':
            if self.sex is None:
                self.attention = SelfAttention(2*feature_dim+2, 2*feature_dim+4)
                self.shared_head = nn.Linear(2*feature_dim+2, feature_dim)
                self.adapter_male = nn.Linear(2*feature_dim+2, feature_dim)
                self.adapter_female = nn.Linear(2*feature_dim+2, feature_dim)
                self.linear1 = nn.Linear(2*feature_dim+2, feature_dim) # for concat fusion
                self.norm1 = nn.LayerNorm(2*feature_dim+2)
            else:
                self.attention = SelfAttention(2*feature_dim, 2*feature_dim)
                self.shared_head = nn.Linear(2*feature_dim, feature_dim)
                self.adapter_male = nn.Linear(2*feature_dim, feature_dim)
                self.adapter_female = nn.Linear(2*feature_dim, feature_dim)
                self.linear1 = nn.Linear(2*feature_dim, feature_dim) # for concat fusion
                self.norm1 = nn.LayerNorm(2*feature_dim)
        else:
            if self.sex is None:
                self.attention = SelfAttention(feature_dim+2, feature_dim)
                self.shared_head = nn.Linear(feature_dim+2, feature_dim)
                self.linear1 = nn.Linear(feature_dim+2, feature_dim) # for concat fusion
                self.norm1 = nn.LayerNorm(feature_dim+2)        
                self.adapter_male = nn.Linear(feature_dim+2, feature_dim)
                self.adapter_female = nn.Linear(feature_dim+2, feature_dim)                
            else:
                self.attention = SelfAttention(feature_dim, feature_dim)
                self.shared_head = nn.Linear(feature_dim, feature_dim)
                self.linear1 = nn.Linear(feature_dim, feature_dim) # for concat fusion
                self.norm1 = nn.LayerNorm(feature_dim)        
                self.adapter_male = nn.Linear(feature_dim, feature_dim)
                self.adapter_female = nn.Linear(feature_dim, feature_dim)

        self.norm = nn.LayerNorm(feature_dim)
        self.tanh = nn.Tanh()
        self.gelu = nn.GELU()
        self.heads = nn.ModuleList([
            nn.Linear(feature_dim, 1) for _ in range(n_cancer)
        ])
        self.linear2 = nn.Linear(feature_dim, 1)

        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, source_features_gene, source_features_mut, sex):

        source_features_gene = F.normalize(source_features_gene, dim=1)
        source_features_mut = F.normalize(source_features_mut, dim=1)
        sex_emb = self.sex_emb(sex)

        ### fusion
        if self.sex is None:
            if self.modality == 'gene+mut':
                z = torch.cat([source_features_gene, source_features_mut, sex_emb], dim = 1) ## concat, linear1(2*hidden_dim+batch_size, hidden_dim)
            elif self.modality == 'gene':
                #z = torch.cat([source_features_gene, tme_features.unsqueeze(1)], dim = 2)
                z = torch.cat([source_features_gene, sex_emb], dim = 1)
            else:
                z = torch.cat([source_features_mut, sex_emb], dim = 1)
        else:
            if self.modality == 'gene+mut':
                z = torch.cat([source_features_gene, source_features_mut], dim = 1) ## concat, linear1(2*hidden_dim+batch_size, hidden_dim)
            elif self.modality == 'gene':
                z = source_features_gene
            else:
                z = source_features_mut

        z = F.normalize(z, dim=1)
        ### classifier
        if z is None:
            return None

        if self.sex is None:
            shared_z = self.shared_head(z)
            # initialize
            delta_z = torch.zeros_like(shared_z)
            mask_male = sex == 0
            mask_female = sex == 1
            # male residual
            if mask_male.any():
                delta_z[mask_male] = (
                    self.adapter_male(z[mask_male])
                )
            # female residual
            if mask_female.any():
                delta_z[mask_female] = (
                    self.adapter_female(z[mask_female])
                )
            # final representation
            z = shared_z + 0.5*delta_z
        else:
            z = self.linear1(z)

        h = self.gelu(z)
        z = self.dropout(h+z)
        outputs = self.linear2(z)
        out = outputs.squeeze()
        return out

def compute_group_attributions_fusion(feature_extractor_gene, feature_extractor_mut, classifier, x_batch):

    emb_layers = []
    if feature_extractor_gene is not None:
        feature_extractor_gene = feature_extractor_gene.to(device)
        feature_extractor_gene.eval()
        emb_layers += [
        m for m in feature_extractor_gene.modules()
        if isinstance(m, nn.Embedding)
    ]
    if feature_extractor_mut is not None:
        feature_extractor_mut = feature_extractor_mut.to(device)
        feature_extractor_mut.eval()
        emb_layers += [
        m for m in feature_extractor_mut.modules()
        if isinstance(m, nn.Embedding)
    ]
    classifier = classifier.to(device)
    classifier.eval()

    # ---- find embeddings to hook (categorical features) ----
    saved = {}; hooks = []
    def make_hook(name):
        def _hook(module, inp, out):
            saved[name] = out.clone()
            #out.retain_grad()
        return _hook
    for i, m in enumerate(emb_layers):
        hooks.append(m.register_forward_hook(make_hook(f"emb{i}")))

    # ---- grad-enabled forward (no torch.no_grad()) ----
    mut = x_batch[0].to(device)
    gene = x_batch[3].to(device)

    is_cont_mut = mut.dtype.is_floating_point
    is_cont_gene = gene.dtype.is_floating_point
    if is_cont_mut:
        mut = mut.clone().detach().requires_grad_(True)
    if is_cont_gene:
        gene = gene.clone().detach().requires_grad_(True)

    features_mut, features_gene = None, None
    if feature_extractor_mut is not None:
        features_mut, features_sex, _ = feature_extractor_mut(mut)
    if feature_extractor_gene is not None:
        features_gene = feature_extractor_gene(gene)

    if features_mut is not None:
        features_mut = features_mut.clone()
    if features_gene is not None:
        features_gene = features_gene.clone()

    if features_sex is not None:
        features_sex = features_sex.clone()

    logits = classifier(features_mut, features_gene, features_sex)
    if logits is None:
        for h in hooks:
            h.remove()
        return None
    logit = logits.sum()
    inputs_for_grad = [mut, gene]

    grads = torch.autograd.grad(
        logit,
        inputs_for_grad,
        retain_graph=True,
        allow_unused=True
    )

    # ---- collect attributions ----
    results = {}
    mut_grad = grads[0]
    gene_grad = grads[1]

    if mut_grad is not None:
        mut_attr = (mut_grad * mut).abs().detach() 
        mut_attr = mut_attr.reshape(mut_attr.size(0), -1)
        results["mut_cont"] = mut_attr.mean(dim=0)

    if gene_grad is not None:
        gene_attr = (gene_grad * gene).abs().detach()   # [B, D_gene]
        gene_attr = gene_attr.reshape(gene_attr.size(0), -1)
        results["gene_cont"] = gene_attr.mean(dim=0)

    for name, out in saved.items():
        if out.grad is None:
            continue
        A = out.grad.abs().sum(dim=-1).detach()
        A = A.reshape(A.size(0), -1)
        results[name] = A.mean(dim=0)

    for h in hooks: h.remove()
    return results



class EarlyStopping:
    def __init__(self, patience=5, mode="max", min_delta=1e-4):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best_score = None
        self.counter = 0
        self.best_states = None

    def step(self, score, models):
        if self.best_score is None:
            self.best_score = score
            #self.best_states = [m.state_dict() for m in models]
            self.best_states = [copy.deepcopy(m.state_dict()) for m in models]
            return False

        improvement = (
            score > self.best_score + self.min_delta
            if self.mode == "max"
            else score < self.best_score - self.min_delta
        )

        if improvement:
            self.best_score = score
            self.counter = 0
            self.best_states = [m.state_dict() for m in models]
        else:
            self.counter = self.counter + 1

        return self.counter >= self.patience

    def restore(self, models):
        for m, state in zip(models, self.best_states):
            m.load_state_dict(state)



class BalancedBatchSampler(Sampler):
    def __init__(self, labels, batch_size=32):
        self.labels = labels.detach().cpu().numpy()
        self.batch_size = batch_size

        self.half = batch_size // 2

        self.pos_idx = np.where(self.labels == 1)[0]
        self.neg_idx = np.where(self.labels == 0)[0]

    def __iter__(self):
        for _ in range(len(self)):
            pos = np.random.choice(self.pos_idx, self.half, replace=True)
            neg = np.random.choice(self.neg_idx, self.half, replace=True)

            batch = np.concatenate([pos, neg])
            np.random.shuffle(batch)

            yield batch.tolist()

    def __len__(self):
        return len(self.labels) // self.batch_size



def normalize_weights(w, target_sum):
    w = np.maximum(w, 0)
    s = w.sum()
    if s == 0:
        w = np.ones_like(w) / len(w)
        s = 1.0
    return w / s * target_sum


def fitness(weights, logits_matrix, y_true, target_sum):
    weights = normalize_weights(weights, target_sum)
    ensemble_logits = np.dot(logits_matrix, weights)
    auc = roc_auc_score(y_true, ensemble_logits)
    return auc

def genetic_optimize_weights(
    logits_matrix, y_true, target_sum = 1,
    pop_size=40,
    generations=50,
    mutation_rate=0.2,
    mutation_scale=0.1,
    random_state=123
):

    if target_sum == 0:
        return [0] * logits_matrix.shape[1], 0
    rng = np.random.default_rng(random_state)
    logits_matrix = np.asarray(logits_matrix)
    y_true = np.asarray(y_true).reshape(-1)

    assert logits_matrix.shape[0] == len(y_true)

    # initialize random weights
    n_models = logits_matrix.shape[1]
    # initialize random weights
    population = rng.random((pop_size, n_models))
    population = np.array([
        normalize_weights(w, target_sum)
        for w in population
    ])

    for gen in range(generations):
        scores = np.array([
            fitness(w, logits_matrix, y_true, target_sum)
            for w in population
        ])


        # keep top half
        top_idx = np.argsort(scores)[-pop_size // 2:]
        parents = population[top_idx]

        new_population = []

        # elitism: keep best
        best_idx = np.argmax(scores)
        new_population.append(population[best_idx])

        while len(new_population) < pop_size:
            p1, p2 = parents[rng.integers(len(parents), size=2)]

            # crossover
            alpha = rng.random()
            child = alpha * p1 + (1 - alpha) * p2

            # mutation
            if rng.random() < mutation_rate:
                child += rng.normal(0, mutation_scale, size=n_models)

            child = normalize_weights(child, target_sum)
            new_population.append(child)

        population = np.array(new_population)

    final_scores = np.array([
        fitness(w, logits_matrix, y_true, target_sum)
        for w in population
    ])


    best_idx = np.argmax(final_scores)
    best_w = normalize_weights(population[best_idx], target_sum)
    best_auc = final_scores[best_idx]

    return best_w, best_auc


class MultiModelClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self):

        self.xgb_model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            eval_metric='auc',
            random_state=123
        )

        self.brf_model = BalancedRandomForestClassifier(
            n_estimators=500,
            max_depth=None,
            max_features='sqrt',
            min_samples_leaf=5,
            random_state=123
        )
        self.svm_model = SVC(
            C=1.0,
            kernel='rbf',
            probability=True,
            random_state=123,
            class_weight="balanced",
            gamma = 0.1
        )

        self.svm_sigmoid = SVC(
                kernel='sigmoid',
                C=1,
                gamma='scale',
                probability=True,
                random_state=123
            )

        self.lr_model = LogisticRegression(
                penalty='elasticnet',
                solver='saga',
                l1_ratio=0.2,
                C=0.1,
                max_iter=5000
            )

        self.svm_linear = SVC(
                kernel='linear',
                C=1,
                probability=True,
                random_state=123
            )

        self.cat_model = CatBoostClassifier(
            iterations=200,
            depth=4,
            learning_rate=0.05,
            loss_function='Logloss',
            verbose=False
        )

        self.ert_model = ExtraTreesClassifier(
            n_estimators=500,
            max_depth=None,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            class_weight='balanced',
            bootstrap=True,
            random_state=123,
            n_jobs=-1
        )

        self.lgbm_model = LGBMClassifier(
            objective='binary',
            n_estimators=300,
            learning_rate=0.03,
            num_leaves=15,
            max_depth=4,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=1.0,
            reg_lambda=1.0,
            class_weight='balanced',
            random_state=42,
            verbosity=-1
        )

    def fit(self, X, y):

        self.xgb_model.fit(X, y)
        self.brf_model.fit(X, y)
        self.svm_model.fit(X,y)
        self.svm_sigmoid.fit(X,y)
        self.svm_linear.fit(X, y)
        self.lr_model.fit(X, y)
        return self

    def get_models(self):

        return {
            'xgb': self.xgb_model,
            'brf': self.brf_model,
            'svm': self.svm_model,
            'svm_linear': self.svm_linear,
            'svm_sigmoid': self.svm_sigmoid,
            'lr': self.lr_model,
        }

    def predict_proba(self, X):

        xgb_prob = self.xgb_model.predict_proba(X)[:, 1].reshape(-1)
        brf_prob = self.brf_model.predict_proba(X)[:, 1].reshape(-1)
        svm_prob = self.svm_model.predict_proba(X)[:, 1].reshape(-1)
        svm_sigmoid_prob = self.svm_sigmoid.predict_proba(X)[:, 1].reshape(-1)
        svm_linear_prob = self.svm_linear.predict_proba(X)[:, 1].reshape(-1)
        lr_prob = self.lr_model.predict_proba(X)[:, 1].reshape(-1)

        return {
            "xgb_prob": xgb_prob,
            "brf_prob": brf_prob,
            "svm_prob": svm_prob,
            "svm_sigmoid_prob": svm_sigmoid_prob,
            "svm_linear_prob": svm_linear_prob,
            "lr_prob": lr_prob,
        }
