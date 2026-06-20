import os
os.environ["CUDA_LAUNCH_BLOCKING"]="1"

import torch
torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch_geometric.data import Data
from positional_encodings.torch_encodings import PositionalEncoding1D
from torch.utils.data import WeightedRandomSampler
import torch.optim as optim
import pandas as pd
import random
import torch.utils.data as du
os.environ["SCIPY_ARRAY_API"] = "1"

from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, f1_score, confusion_matrix, precision_recall_curve, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold, cross_validate, RepeatedStratifiedKFold
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
import xgboost as xgb
from scipy.special import expit
from sklearn.linear_model import SGDClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.base import clone
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pickle

import numpy as np
import requests
import warnings, copy
import concurrent.futures
import json
from inmoose.pycombat import pycombat_norm
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from itertools import cycle

from layers import convert_to_sn, remove_all_normalization_layers

import argparse
from processing import get_all_score, set_seed, zscore_norm, get_2dpos, prepare_mut_data, encode_domain, remove_nonmutate, prepare_test_data, _to_tensor_df, best_threshold_macro_f1, quantile_transformation, cohort_align_zscore
from selection import feature_selection

from model_utils import SelfAttention, gene_model, mut_model, tme_extractor, Classifier_fusion, DomainDiscriminator, compute_group_attributions_fusion, mmd_loss, BalancedBatchSampler, EarlyStopping, ResidualModel, LoRALinear, apply_lora_to_linears, genetic_optimize_weights, MultiModelClassifier
warnings.filterwarnings("ignore")

device = (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
print("device: ", device)

def safe_transform(le, values):
    mapping = {c: i for i, c in enumerate(le.classes_)}
    unknown = len(le.classes_)
    return [mapping.get(v, unknown) for v in values]

def pretrain_tcga(gene, pos, clin, num_epochs, batch_size, learning_rate, epsilon, hidden_dim, feature_dim, top_tme_scores,
                        interaction = True, modality = "gene+mut", cancer = "allcancer", monosex = None, trt = None, lodo = False):

    fold_metrics = []
    scores_all = {"logit":[], "modality": []}
    score_ind_all = []
    all_thresholds = []
    fold_cohort = []
    label = 'resp'

    monocancer = False
    if (cancer == 'allcancer') or ("+" in cancer):
        monocancer = False
    
    monotrt = False
    if trt == 'PD1':
        monotrt = True

### no cv, train full model 
    for pack in prepare_mut_data(pos, clin, label_id=label, k_fold = 1, test_size=0, gene=gene, tme_feature = top_tme_scores):

        X_train_df, y_train, gene_train_df, dom_train1 = pack['X_train'], pack['y_train'], pack['gene_train'], pack['dom_train1']
        X_tr, y_tr, _, gene_train = _to_tensor_df(X_train_df, y_train, dom_train1, device, gene_train_df)

        target, target_response, gene_target, target_dom = pack['X_val'], pack['y_val'], pack['gene_val'], pack['dom_val1']
        target, target_response, target_dom, gene_target = _to_tensor_df(target, target_response, target_dom, device, gene_target)
 
        feature_extractor_gene = gene_model(gene_train.shape[1], hidden_dim, feature_dim, interaction = interaction, monosex = monosex, 
                                monocancer = monocancer, monotrt = monotrt)
        feature_extractor_mut = mut_model(X_tr.shape[1], hidden_dim, feature_dim, interaction = interaction, monosex = monosex, 
                                monocancer = monocancer, monotrt = monotrt)
        feature_extractor_gene = feature_extractor_gene.to(device)
        feature_extractor_mut = feature_extractor_mut.to(device)
        n_domain = 1
        if cancer != 'melanoma':
            n_domain = 3 
        classifier = Classifier_fusion(hidden_dim, feature_dim, modality = modality, dropout = 0.3, n_cancer = n_domain).to(device)

        sampler = BalancedBatchSampler(y_tr, batch_size=batch_size)
        source_loader1 = du.DataLoader(du.TensorDataset(X_tr, y_tr, gene_train), batch_sampler=sampler)

        optimizer = optim.Adam(list(feature_extractor_gene.parameters()) + list(feature_extractor_mut.parameters()) +
                                list(classifier.parameters()), lr=learning_rate, eps = epsilon, weight_decay=1e-03)

        # Loss functions
        pos_class = (y_tr == 1).sum()
        neg_class = (y_tr == 0).sum()
        pos_weight = neg_class / (pos_class + 1e-8)   # avoid division by zero
        pos_weight = torch.tensor(pos_weight, dtype=torch.float32).to(device)
        classification_criterion = nn.BCEWithLogitsLoss(pos_weight = pos_weight)#nn.BCELoss()

    # -----------------------------
    # Training Loop
    # -----------------------------
        best_val_auc = -float("inf")
        best_scores = None
        best_f1 = -float("inf")
        for epoch in range(num_epochs):
            feature_extractor_gene.train()
            feature_extractor_mut.train()
            classifier.train()

            for _, source_batch1 in enumerate(source_loader1):
            
                source_data1, source_labels1, source_gene = source_batch1
                source_data1, source_labels1, source_gene = source_data1.to(device), source_labels1.to(device), source_gene.to(device)

                source_features1, sex, _ = feature_extractor_mut(source_data1)
                source_gene1 = feature_extractor_gene(source_gene)

                class_preds1 = classifier(source_gene1, source_features1, sex)
                task_loss = classification_criterion(class_preds1, source_labels1.to(torch.float32))
                total_loss = task_loss
                
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

    threshold = np.median(all_thresholds)
    score_df = None
        
    return  feature_extractor_gene, feature_extractor_mut, classifier, threshold, score_df



def finetune(feature_extractor_gene, feature_extractor_mut, gene, pos, clin, num_epochs, batch_size, learning_rate, epsilon, hidden_dim, feature_dim,
                top_tme_scores, interaction = True, modality = "gene+mut", cancer = "allcancer",
                monosex = None, trt = None, lodo = False, param_grids = None):


    label = 'resp'
    monocancer = False
    if (cancer == 'allcancer') or ("+" in cancer):
        monocancer = False
    else:
        cancer_type = None
    
    monotrt = False
    if trt == 'PD1':
        monotrt = True

    for pack in prepare_mut_data(pos, clin, label_id=label, k_fold = 1, test_size=0.2, gene=gene, tme_feature = top_tme_scores):

        X_train_df, y_train, gene_train_df, dom_train1 = pack['X_train'], pack['y_train'], pack['gene_train'], pack['dom_train1']
        X_tr, y_tr, _, gene_train = _to_tensor_df(X_train_df, y_train, dom_train1, device, gene_train_df)

        target, target_response, gene_tgt, target_dom = pack['X_val'], pack['y_val'], pack['gene_val'], pack['dom_val1']
        target, target_response, target_dom, gene_target = _to_tensor_df(target, target_response, target_dom, device, gene_tgt)
 
        if feature_extractor_gene is not None:
            mut_model1 = copy.deepcopy(feature_extractor_mut)
            gene_model1 = copy.deepcopy(feature_extractor_gene)

            apply_lora_to_linears(mut_model1,  r=16, alpha=16.0, dropout=0.3, freeze_base=True)
            apply_lora_to_linears(gene_model1, r=16, alpha=16.0, dropout=0.3, freeze_base=True)

        else:
            gene_model1 = gene_model(gene_train.shape[1], hidden_dim, feature_dim, interaction = interaction, monosex = monosex, 
                                    monocancer = monocancer, monotrt = monotrt).to(device)
            mut_model1 = mut_model(X_tr.shape[1], hidden_dim, feature_dim, interaction = interaction, monosex = monosex, 
                                    monocancer = monocancer, monotrt = monotrt).to(device)

        n_domain = 1
        if cancer == 'allcancer':
            n_domain = len(pos['cancer_type'].unique())

        classifier = Classifier_fusion(hidden_dim, feature_dim, modality = modality, dropout = 0.3, n_cancer = n_domain).to(device)

        sampler = BalancedBatchSampler(y_tr, batch_size=batch_size)
        source_loader1 = du.DataLoader(du.TensorDataset(X_tr, y_tr, gene_train), batch_sampler=sampler)

        dc = DomainDiscriminator(2*feature_dim, feature_dim, n_domain).to(device)
        optimizer = optim.Adam(list(mut_model1.parameters()) + list(gene_model1.parameters())+
                list(classifier.parameters()), lr=learning_rate, eps = epsilon, weight_decay=1e-03)

        pos_class = (y_tr == 1).sum()
        neg_class = (y_tr == 0).sum()
        pos_weight = neg_class / (pos_class + 1e-8)   # avoid division by zero
        pos_weight = torch.tensor(pos_weight, dtype=torch.float32).to(device)
        classification_criterion = nn.BCEWithLogitsLoss(pos_weight = pos_weight)
        criterion_domain = nn.CrossEntropyLoss()

        early_stopper = EarlyStopping(patience=5, mode="max", min_delta=1e-4)
        best_f1 = -float("inf")
        best_res_auc = -float("inf")

        if monosex == 'Female':
            num_epochs = 100
        for epoch in range(num_epochs):#(tr_num_epoch):

            gene_model1.train()
            mut_model1.train()
            classifier.train()
            dc.train()
            alpha = min(0.1, epoch / num_epochs)
            for _, source_batch1 in enumerate(source_loader1):
            
                source_data1, source_labels1, source_gene = source_batch1
                source_data1, source_labels1, source_gene = source_data1.to(device), source_labels1.to(device), source_gene.to(device)

                source_gene1 = gene_model1(source_gene)
                source_features1, sex, cancer_type = mut_model1(source_data1)
                features = torch.cat([source_gene1, source_features1], axis = 1)

                class_preds1 = classifier(source_gene1, source_features1, sex)
                task_loss = classification_criterion(class_preds1, source_labels1.to(torch.float32))
                total_loss = task_loss
                prob = torch.sigmoid(class_preds1)

                if cancer == 'allcancer':
                    domain_logits = dc(features, alpha = 0.1)
                    domain_loss = criterion_domain(domain_logits, cancer_type)
                    total_loss += alpha * domain_loss

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
            
            mut_model1.eval()
            gene_model1.eval()
            classifier.eval()

            with torch.no_grad():
                outputs_mut, outputs_sex, outputs_cancer = mut_model1(target)
                outputs_gene = gene_model1(gene_target)
                features = torch.cat([outputs_gene, outputs_mut], axis = 1)
                features = torch.nan_to_num(features, nan=1e-06)

                outputs = classifier(outputs_gene, outputs_mut, outputs_sex).view(-1, 1).reshape(-1)

                prob = torch.sigmoid(outputs)
                features = features#.detach().cpu().numpy()
                labels = target_response#.detach().cpu().numpy()
                probs = prob#.detach().cpu().numpy()
                base_logits = outputs#.detach().cpu().numpy()
                final_logits = probs
                probs = final_logits.squeeze()
                
                probs1 = probs.detach().cpu().numpy()
                fpr, tpr, thresholds = roc_curve(target_response.cpu().numpy(), probs1)
                optimal_threshold, _ = best_threshold_macro_f1(target_response.cpu().numpy(), probs1)

                if optimal_threshold == np.inf:
                    preds = (probs >= 0.5).long()
                else:
                    preds = (probs >= optimal_threshold).long()

                y_true = target_response.cpu().numpy()
                y_pred = preds.cpu().numpy()
                y_prob = probs.cpu().numpy()

            accuracy = accuracy_score(y_true, y_pred)
            auc = roc_auc_score(y_true, y_prob)

            if auc > best_res_auc:
                best_res_auc = auc

            f1 = f1_score(y_true, y_pred, average='macro')
            if f1 > best_f1:
                best_f1 = f1
                best_threshold_f1 = optimal_threshold
            print(f"------ Finetune NN Validation: Iteration Loss: {task_loss:.4f}, ACC: {accuracy:.4f}, AUC: {auc:.4f}, F1 score: {f1:.4f}, threshold: {optimal_threshold:.4f}")

        gene_model1.eval()
        mut_model1.eval()
        classifier.eval()

        res_features = []
        res_labels = []
        res_probs = []
        tr_logits = []

        with torch.no_grad():
            for _, source_batch1 in enumerate(source_loader1):
                source_data1, source_labels1, source_gene = source_batch1
                source_data1, source_labels1, source_gene = source_data1.to(device), source_labels1.to(device), source_gene.to(device)

                source_gene1 = gene_model1(source_gene)
                source_sex = source_data1[:,-1].long()
                source_features1, sex, _ = mut_model1(source_data1)
                features = torch.cat([source_gene1, source_features1], axis = 1)

                class_preds1 = classifier(source_gene1, source_features1, sex)
                task_loss = classification_criterion(class_preds1, source_labels1.to(torch.float32))
                total_loss = task_loss# + 0.01 * class_preds1.std()
                prob = torch.sigmoid(class_preds1)
                res_features.append(features.detach().cpu().numpy())
                res_labels.append(source_labels1.detach().cpu().numpy())
                res_probs.append(prob.detach().cpu().numpy())
                tr_logits.append(class_preds1.view(-1, 1).detach().cpu().numpy())

            X = np.vstack(res_features)
            p = np.vstack(res_probs).flatten()
            y = np.hstack(res_labels)
            base_logit = np.concatenate([x.reshape(-1) for x in tr_logits])
            ## OOF prediction
            ensemble_model = MultiModelClassifier()
            target_sum = 0.5
            models = ensemble_model.get_models()
            model_names = list(models.keys())
            n_models = len(model_names)

            if not os.path.exists(f"data/best_params_{monosex}.json"):
                best_params = {}
                for name, model in models.items():
                    if name not in param_grids:
                        continue
                    search = RandomizedSearchCV(
                        estimator=model,
                        param_distributions=param_grids[name],
                        n_iter=20,
                        scoring='roc_auc',
                        cv=5,
                        random_state=123,
                        n_jobs=-1
                    )
                    search.fit(X, y)
                    best_params[name] = search.best_params_
                with open(f"data/best_params_{monosex}.json", "w") as f:
                    json.dump(best_params, f, indent=4)
            else:
                print(f"reading parameters from data/best_params_{monosex}.json")
                with open(f"data/best_params_{monosex}.json") as f:
                    best_params = json.load(f)

            for name, params in best_params.items():
                if name in models:
                    models[name].set_params(**params)

            oof_preds = np.zeros((len(y), n_models))
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)

            for tr_idx, val_idx in skf.split(X, y):
                X_tr, X_val = X[tr_idx], X[val_idx]
                y_tr = y[tr_idx]
                for m_idx, name in enumerate(model_names):
                    model = clone(models[name])
                    model.fit(X_tr, y_tr)
                    oof_preds[val_idx, m_idx] = (
                        model.predict_proba(X_val)[:, 1]
                    )

            ga_w, ga_auc = genetic_optimize_weights(oof_preds,y, target_sum)

            ensemble_model = MultiModelClassifier()
            model_map = {
                'xgb': ensemble_model.xgb_model,
                'brf': ensemble_model.brf_model,
                'svm': ensemble_model.svm_model,
                'svm_linear': ensemble_model.svm_linear,
                'svm_sigmoid': ensemble_model.svm_sigmoid,
                'lr': ensemble_model.lr_model,
                # 'cat': ensemble_model.cat_model,
                # 'ert': ensemble_model.ert_model,
                # 'lgbm': ensemble_model.lgbm_model
            }

            for name, params in best_params.items():
                if name in model_map:
                    model_map[name].set_params(**params)
            ensemble_model.fit(X,y)

            outputs_mut, outputs_sex, _ = mut_model1(target)
            outputs_gene = gene_model1(gene_target)
            features = torch.cat([outputs_gene, outputs_mut], axis = 1)
            features = torch.nan_to_num(features, nan=1e-06)

            outputs = classifier(outputs_gene, outputs_mut, outputs_sex).view(-1, 1).reshape(-1)

            prob = torch.sigmoid(outputs)
            features = features.detach().cpu().numpy()
            labels = target_response.detach().cpu().numpy()
            probs = prob.detach().cpu().numpy()
            base_logits = outputs.detach().cpu().numpy()

            model_probs = ensemble_model.predict_proba(features)

            aux_probs = np.column_stack([
                v for v in model_probs.values()
            ])

            final_logits = (1-target_sum) * probs + np.dot(aux_probs, ga_w)
            probs = final_logits.squeeze()
            
            probs1 = probs#.cpu().numpy()
            fpr, tpr, thresholds = roc_curve(target_response.cpu().numpy(), probs1)
            optimal_threshold, _ = best_threshold_macro_f1(target_response.cpu().numpy(), probs1)

            if optimal_threshold == np.inf:
                preds = (probs >= 0.5)#.long()
            else:
                preds = (probs >= optimal_threshold)#.long()

            y_true = target_response.cpu().numpy()
            y_pred = preds#.cpu().numpy()
            y_prob = probs#.cpu().numpy()

        y = target_response.detach().cpu().numpy()
        p_pos = probs[y==1].mean()
        p_neg = probs[y==0].mean()
        print("validation prob for pos and neg", p_pos, p_neg)

        accuracy = accuracy_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_prob)

        if auc > best_res_auc:
            best_res_auc = auc

        f1 = f1_score(y_true, y_pred, average='macro')
        if f1 > best_f1:
            best_f1 = f1
            best_threshold_f1 = optimal_threshold
        print(f"------ Finetune Ems Validation: Iteration Loss: {task_loss:.4f}, ACC: {accuracy:.4f}, AUC: {auc:.4f}, F1 score: {f1:.4f}, threshold: {optimal_threshold:.4f}")

    score_df = None
    threshold = optimal_threshold
            
    return  gene_model1, mut_model1, classifier, threshold, score_df, ensemble_model, ga_w

def predict_fusionmodel(predict_data, feature_extractor_gene, feature_extractor_mut, classifier, ensemble_model,
                        threshold, modality, ga_w = None, predict_cohort = None):

    print("---- prediction ----")
    target_mutation, target_response, target_gene1, cohort = predict_data[0], predict_data[1], predict_data[3], predict_data[4]

    target_mutation, target_gene1 = target_mutation.to(device), target_gene1.to(device)
    gene_location, mut_location = gene_location.to(device), mut_location.to(device)

    feature_extractor_gene.eval()
    feature_extractor_mut.eval()
    classifier.eval()

    with torch.no_grad():

        outputs_mut, outputs_sex, _ = feature_extractor_mut(target_mutation)
        outputs_gene = feature_extractor_gene(target_gene1)
        features = torch.cat([outputs_gene, outputs_mut], axis = 1)

        outputs = classifier(outputs_gene, outputs_mut, outputs_sex).view(-1, 1).reshape(-1)
        probs = expit(outputs.detach().cpu().numpy())
        features = features.detach().cpu().numpy()
        model_probs = ensemble_model.predict_proba(features)

        target_sum = 0.5
        aux_probs = np.column_stack([
                v for v in model_probs.values()
            ])

        final_logits = (1-target_sum) * expit(outputs.detach().cpu().numpy()) + np.dot(aux_probs, ga_w)
        probs = final_logits.squeeze()

        fpr, tpr, thresholds = roc_curve(target_response, probs)#.cpu().numpy())

        if threshold != np.inf:
            preds = (probs >= threshold)#.long()
        else:
            preds = (probs >= 0.5)#.long()

        y_true = target_response
        y_pred = preds#.detach().cpu().numpy()
        y_prob = probs#.cpu().numpy()
        print(y_prob[:10])

    y = target_response.detach().cpu().numpy()
    p_pos = probs[y==1].mean()
    p_neg = probs[y==0].mean()
    print("test prob for pos and neg", p_pos, p_neg)

    accuracy = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    print(f"Confusion Matrix → TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")
    f1 = f1_score(y_true, y_pred, average='macro')
    print("AUC: {:.4f}, accuracy: {:.4f}, f1 score: {:.4f}, threshold: {:.4f}".format(auc, accuracy, f1, threshold))
    print("GA prob: ", ga_w)
    output = []
    output.append(auc)
    output.append(accuracy)
    output.append(f1)
    if predict_cohort is not None:
        per_cohort_auc = {}
        for cohort in predict_cohort.unique():
            mask = (predict_cohort == cohort).values   # convert to numpy boolean mask
            auc = roc_auc_score(y_true[mask], y_prob[mask])
            per_cohort_auc[cohort] = auc
        print("per cohort auc: ", per_cohort_auc)
    output.append(per_cohort_auc)
    output.append(ga_w)
    return output, y_prob, y_pred #score.detach().cpu().tolist()


def main(args):
    set_seed(args.repeat)
    outdir = "output/pretrain_" + args.pretrain + "_out_" + args.TRT + "_int_" + str(args.interaction) + "_" + args.cancer + "_" + str(int(args.repeat)) + ".csv"
    score_outdir = "output/pretrain_" + args.pretrain + "_score_" + args.TRT + "_" + str(int(args.repeat)) + ".csv"

    with open(f"data/param_grid.json") as f:
        param_grids = json.load(f)

    gene_norm_union_minmax = pd.read_csv("data/sample_expression.csv", index_col = 0) ## use a customed input here
    gene_norm_clin_pd1 = pd.read_csv("data/combined_clin_tme.csv", index_col = 0)
    gene_norm_clin_pd1 = gene_norm_clin_pd1[~gene_norm_clin_pd1['Gender'].isna()]
    gene_pos_grouped_pd1 = pd.read_csv("data/sample_mutation.csv", index_col = 0)

    if args.cancer != "allcancer":
        if "+" in args.cancer:
            selected_cancer = [i.strip() for i in args.cancer.split("+")]
            gene_norm_clin_pd1 = gene_norm_clin_pd1[gene_norm_clin_pd1['cancer'].isin(selected_cancer)]
        else:
            gene_norm_clin_pd1 = gene_norm_clin_pd1[gene_norm_clin_pd1['cancer'] == args.cancer]
    gene_norm_union_minmax = gene_norm_union_minmax.loc[:,gene_norm_union_minmax.columns.isin(gene_norm_clin_pd1.index)]
    gene_pos_grouped_pd1 = gene_pos_grouped_pd1[gene_pos_grouped_pd1.index.isin(gene_norm_clin_pd1.index)]

    tcga_gene = pd.read_csv("data/sample_tcga_expression.csv", index_col = 0)
    tcga_clin = pd.read_csv("data/tcga_clin_tme.csv", index_col = 0)
    tcga_clin = tcga_clin[~tcga_clin['Gender'].isna()]
    tcga_mutation = pd.read_csv("data/sample_tcga_mutation.csv", index_col = 0)

    tcga_clin = tcga_clin[tcga_clin['source'] != 'luad']
    tcga_gene = tcga_gene.loc[:,tcga_gene.columns.isin(tcga_clin.index)]
    tcga_mutation = tcga_mutation[tcga_mutation.index.isin(tcga_mutation.index)]

    tcga_mapping = {'melanoma': 'skcm', 'bladder': 'blca', 'ccrcc': 'kirc'}
    tcga_cancer_selected = [tcga_mapping[i] for i in gene_norm_clin_pd1['cancer'].unique()]
    if args.cancer != 'allcancer': 
        tcga_clin = tcga_clin[tcga_clin['cancer'].isin(tcga_cancer_selected)]
        tcga_gene = tcga_gene.loc[:,tcga_gene.columns.isin(tcga_clin.index)]
        tcga_mutation = tcga_mutation[tcga_mutation.index.isin(tcga_clin.index)]
    gene_pos_grouped_pd1 = remove_nonmutate(gene_pos_grouped_pd1)
    pd1_samples = list(set(gene_pos_grouped_pd1.index) & set(gene_norm_clin_pd1.index))
    pd1_genesamples = list(set(gene_norm_union_minmax.columns) & set(gene_norm_clin_pd1.index))
    gene_pos_grouped_pd1 = gene_pos_grouped_pd1.loc[pd1_samples]
    gene_norm_union_minmax = gene_norm_union_minmax[pd1_genesamples]

    tcga_mutation = remove_nonmutate(tcga_mutation)
    tcga_samples = list(set(tcga_mutation.index) & set(tcga_clin.index) & set(tcga_gene.columns))
    tcga_mutation = tcga_mutation.loc[tcga_samples]
    tcga_gene = tcga_gene[tcga_samples]
    tcga_clin = tcga_clin.loc[tcga_samples]

    ## common mutated genes for training and test set
    mutated_genes = list(set(gene_pos_grouped_pd1.columns) & set(tcga_mutation.columns))
    gene_pos_grouped_pd1 = gene_pos_grouped_pd1[mutated_genes]
    tcga_mutation = tcga_mutation[mutated_genes]

    expr_genes = list(set(gene_norm_union_minmax.index) & set(tcga_gene.index))
    gene_norm_union_minmax = gene_norm_union_minmax.loc[expr_genes]
    tcga_gene = tcga_gene.loc[expr_genes]

    gene_norm_clin = gene_norm_clin_pd1.loc[pd1_samples]
    gene_pos_grouped = gene_pos_grouped_pd1
    gene_pos_grouped = gene_pos_grouped.fillna(0)
    tcga_mutation = tcga_mutation.fillna(0)

    gene_norm_clin_gene = gene_norm_clin_pd1.loc[pd1_genesamples]
    gene_norm_union = gene_norm_union_minmax
    gene_norm_union = gene_norm_union.fillna(0)
    gene_norm_union = gene_norm_union.T
    tcga_gene = tcga_gene.T.fillna(0)

    X_bin = (gene_pos_grouped > 0).astype(float)
    gene_pos_grouped['mut_count'] = np.log1p(
        X_bin.sum(axis=1)
    )
    gene_pos_grouped['mut_max'] = np.log1p(
        gene_pos_grouped.max(axis=1)
    )
    gene_pos_grouped['mut_mean'] = (
        gene_pos_grouped.mean(axis=1)
    )

    gene_pos_grouped['sex'] = gene_norm_clin['Gender'].map({'M': 0, 'F':1, np.nan: -1})
    gene_pos_grouped['total_count'] = gene_norm_clin['tm_norm']
    gene_pos_grouped['msi'] = gene_norm_clin['msi_status']
    gene_norm_clin = gene_norm_clin.rename(columns={"MET": "MET_TUMOR"})

    gene_pos_grouped['cancer_type'] = gene_norm_clin['cancer'].map({
        "melanoma": 0,
        "ccrcc": 1,
        "bladder": 2
    })

    gene_norm_union['source1'] = gene_norm_clin_gene['source'].map({
        "Braun": 0,
        "IMmotion150": 1,
        "Liu": 2,
        "Riaz": 3,
        "Hugo": 4,
        "Damrauer": 5,
        "IMvigor210": 6
    })

    gene_pos_grouped['source1'] = gene_norm_clin['source'].map({
        "Braun": 0,
        "IMmotion150": 1,
        "Liu": 2,
        "Riaz": 3,
        "Hugo": 4,
        "Damrauer": 5,
        "IMvigor210": 6
    })

    X_bin = (tcga_mutation > 0).astype(float)
    tcga_mutation['mut_count'] = np.log1p(
        X_bin.sum(axis=1)
    )
    tcga_mutation['mut_max'] = np.log1p(
        tcga_mutation.max(axis=1)
    )
    tcga_mutation['mut_mean'] = (
        tcga_mutation.mean(axis=1)
    )

    tcga_mutation['sex'] = tcga_clin['Gender'].map({'M': 0, 'F':1, np.nan: -1})
    tcga_mutation['total_count'] = tcga_clin['tm_norm']
    tcga_mutation['msi'] = tcga_clin['msi_status']
    tcga_mutation['source1'] = 7
    tcga_gene['sex'] = tcga_clin['Gender'].map({'M': 0, 'F':1, np.nan: -1})
    tcga_gene['msi'] = tcga_clin['msi_status']
    tcga_gene['source1'] = 7
    tcga_mutation['trt1'] = 0
    tcga_gene['trt1'] = 0
    tcga_clin = tcga_clin.rename(columns={"MET": "MET_TUMOR"})
    tcga_clin['domain_labels1'] = 0

    tcga_mutation['cancer_type'] = tcga_clin['cancer'].map({
        "skcm": 0,
        "kirc": 1,
        "blca": 2
    })

    tcga_gene['cancer_type'] = tcga_clin['cancer'].map({
        "skcm": 0,
        "kirc": 1,
        "blca": 2
    })

    gene_norm_union['cancer_type'] = gene_norm_clin_gene['cancer'].map({
        "melanoma": 0,
        "ccrcc": 1,
        "bladder": 2
    })

    gene_norm_clin = encode_domain(gene_norm_clin, 'trt')
    gene_pos_grouped['trt1'] = gene_norm_clin['domain_labels1']
    gene_norm_union['sex'] = gene_norm_clin_gene['Gender'].map({'M': 0, 'F':1, np.nan: -1})
    gene_norm_union['msi'] = gene_norm_clin_gene['msi_status']
    gene_norm_clin_gene = encode_domain(gene_norm_clin_gene, 'trt')
    gene_norm_union['trt1'] = gene_norm_clin_gene['domain_labels1']
    gene_norm_clin_gene = gene_norm_clin_gene.rename(columns={"MET": "MET_TUMOR"})
    gene_norm_union = gene_norm_union.fillna(0)
    gene_pos_grouped = gene_pos_grouped.fillna(0)

    sample_genepos = gene_norm_union.index.intersection(gene_pos_grouped.index)
    clin_common = gene_norm_clin.loc[sample_genepos]
    clin_common = clin_common[clin_common['source'] != 'va']

    tcga_sample_genepos = tcga_gene.index.intersection(tcga_mutation.index)
    tcga_clin_common = tcga_clin.loc[tcga_sample_genepos]

    with open("data/combined_heldout_samples.json") as f:
        heldout = json.load(f)

    male_heldout_fusion = heldout['male_fusion_heldout']
    male_heldout_fusion = list(set(male_heldout_fusion) & set(clin_common.index))
    female_heldout_fusion = heldout['female_fusion_heldout']
    female_heldout_fusion = list(set(female_heldout_fusion) & set(clin_common.index))
    heldouts = male_heldout_fusion + female_heldout_fusion
    heldouts = random.sample(heldouts, len(heldouts))
    heldouts = [i for i in heldouts if i in clin_common.index]

    cols_to_move1 = ['cancer_type', 'trt1', 'sex']
    cols_to_move = ['total_count', 'mut_count', 'mut_max', 'mut_mean', 'cancer_type', 'trt1', 'sex']

    pd1_heldout_fusion = clin_common.loc[heldouts].query("trt == 'PD1'").index
    train_samples_fusion = list(set(clin_common[clin_common['trt'] == 'PD1'].index) - set(pd1_heldout_fusion))

    train_samples_fusion = random.sample(train_samples_fusion, len(train_samples_fusion))
    clin_common_train = clin_common.loc[train_samples_fusion]
    clin_common_test = clin_common.loc[heldouts]
    clin_common_train_male, clin_common_train_female = clin_common_train[clin_common_train['Gender'] == 'M'], clin_common_train[clin_common_train['Gender'] == 'F']
    clin_common_test_male, clin_common_test_female = clin_common_test[clin_common_test['Gender'] == 'M'], clin_common_test[clin_common_test['Gender'] == 'F']

    gene_common_test = gene_norm_union.loc[heldouts]
    gene_common_test_male, gene_common_test_female = gene_common_test.loc[clin_common_test_male.index], gene_common_test.loc[clin_common_test_female.index]
    gene_common_train = gene_norm_union.loc[train_samples_fusion]
    gene_common_train_female1 = gene_common_train.loc[clin_common_train_female.index]
    gene_common_train_male, gene_common_train_female = gene_common_train.loc[clin_common_train_male.index], gene_common_train.loc[clin_common_train_female.index]


    ### normalization
    gene_common_train_norm, gene_common_test_norm = zscore_norm(gene_common_train.drop(columns = cols_to_move1), gene_common_test.drop(columns = cols_to_move1), method = 'minmax')
    gene_common_train_norm_male, gene_common_test_norm_male = zscore_norm(gene_common_train_male.drop(columns = cols_to_move1), gene_common_test_male.drop(columns = cols_to_move1), method = 'minmax')
    gene_common_train_norm_female, gene_common_test_norm_female = zscore_norm(gene_common_train_female.drop(columns = cols_to_move1), gene_common_test_female.drop(columns = cols_to_move1), method = 'minmax')

    gene_common_train = pd.concat([gene_common_train_norm, gene_common_train[cols_to_move1]], axis = 1)
    gene_common_train_male = pd.concat([gene_common_train_norm_male, gene_common_train_male[cols_to_move1]], axis = 1)
    gene_common_train_female = pd.concat([gene_common_train_norm_female, gene_common_train_female[cols_to_move1]], axis = 1)

    gene_common_test = pd.concat([gene_common_test_norm, gene_common_test[cols_to_move1]], axis = 1)
    gene_common_test_male = pd.concat([gene_common_test_norm_male, gene_common_test_male[cols_to_move1]], axis = 1)
    gene_common_test_female = pd.concat([gene_common_test_norm_female, gene_common_test_female[cols_to_move1]], axis = 1)


    mutation_common_train = gene_pos_grouped.loc[train_samples_fusion]
    mutation_common_test = gene_pos_grouped.loc[heldouts]
    mutation_common_train_male, mutation_common_train_female = mutation_common_train.loc[clin_common_train_male.index], mutation_common_train.loc[clin_common_train_female.index]
    mutation_common_test_male, mutation_common_test_female = mutation_common_test.loc[clin_common_test_male.index], mutation_common_test.loc[clin_common_test_female.index]

    tcga_clin_common_train = tcga_clin_common
    tcga_clin_common_train_male, tcga_clin_common_train_female = tcga_clin_common_train[tcga_clin_common_train['Gender'] == 'M'], tcga_clin_common_train[tcga_clin_common_train['Gender'] == 'F']
    tcga_gene_common_train = tcga_gene.loc[tcga_clin_common.index]
    tcga_gene_common_train_male, tcga_gene_common_train_female = tcga_gene_common_train.loc[tcga_clin_common_train_male.index], tcga_gene_common_train.loc[tcga_clin_common_train_female.index]
    ## normalization
    tcga_gene_common_train_norm = zscore_norm(tcga_gene_common_train.drop(columns = cols_to_move1), method = 'minmax')
    tcga_gene_common_train_norm_male = zscore_norm(tcga_gene_common_train_male.drop(columns = cols_to_move1), method = 'minmax')
    tcga_gene_common_train_norm_female = zscore_norm(tcga_gene_common_train_female.drop(columns = cols_to_move1), method = 'minmax')    
 

    tcga_gene_common_train = pd.concat([tcga_gene_common_train_norm, tcga_gene_common_train[cols_to_move1]], axis = 1)
    tcga_gene_common_train_male = pd.concat([tcga_gene_common_train_norm_male, tcga_gene_common_train_male[cols_to_move1]], axis = 1)
    tcga_gene_common_train_female = pd.concat([tcga_gene_common_train_norm_female, tcga_gene_common_train_female[cols_to_move1]], axis = 1)

    tcga_mutation_common_train = tcga_mutation.loc[tcga_clin_common.index]
    tcga_mutation_common_train_male, tcga_mutation_common_train_female = tcga_mutation_common_train.loc[tcga_clin_common_train_male.index], tcga_mutation_common_train.loc[tcga_clin_common_train_female.index]


    # selected_features_deg = feature_selection(gene_common_train, mutation_common_train, clin_common_train)
    # selected_features_deg_male = feature_selection(gene_common_train_male, mutation_common_train_male, clin_common_train_male)
    # selected_features_deg_female = feature_selection(gene_common_train_female1, mutation_common_train_female1, clin_common_train_female1, male_gene = selected_features_deg_male['expr'],
    #                                 merged_gene = selected_features_deg['expr'])

    with open("data/combined_deg_features_filtered.json") as f:
        selected_features_deg = json.load(f)       
    with open("data/combined_deg_features_male_filtered.json") as f:
        selected_features_deg_male = json.load(f)   
    with open("data/combined_deg_features_female_filtered.json") as f:
        selected_features_deg_female = json.load(f) 

    literature_genes = ["BMP2", "SELE", "CD274", "SH3TC1", "CHST15", "LAG3", "CKLF", "TLR7", "ESCO2", "RXRA", "IFNG", "CXCL9", "CXCL10", "GZMB", "PRF1", "CD8A", "PDCD1", "TIGIT"]
    female_tme = ['PDCD1','XIAP','GZMA','CD8B','TNFRSF14','SLC20A1','CXCL2','CMKLR1','HLA-DRB1','LAMP3','CD276','CXCR4','CXCL9','CD8A','BMPR2','HLA-E','CD2','GZMK','NKG7']
    male_tme = ['PRF1','B2M','HLA-E','GZMK','HLA-DRA','CD86','STAT1','XIAP','CXCL9','CCL21']
    merged_tme = ['IL2RG','CD200','CCL20','NKG7','CXCL10','CD276','CCL21','CXCR4','HLA-DRA','CXCL2','CIITA','TAGAP','CMKLR1','GZMA','LAMP3','IDO1','CD4','CD28','HLA-DRB1','CCL19','STAT1']

    merged_feature = selected_features_deg['expr'] + literature_genes + merged_tme
    male_feature = selected_features_deg_male['expr'] + literature_genes + male_tme
    female_feature = selected_features_deg_female['expr'] + literature_genes + female_tme

    selected_gene_features = list(dict.fromkeys(x for x in merged_feature if x in gene_common_train.columns and x in tcga_gene_common_train.columns))
    selected_mutation_features = [x for x in merged_feature if x in mutation_common_train.columns and x in tcga_mutation_common_train.columns]
    selected_gene_features_male = list(dict.fromkeys(x for x in male_feature if x in gene_common_train.columns and x in tcga_gene_common_train.columns))
    selected_mutation_features_male = [x for x in male_feature if x in mutation_common_train.columns and x in tcga_mutation_common_train.columns]
    selected_gene_features_female = list(dict.fromkeys(x for x in female_feature if x in gene_common_train.columns and x in tcga_gene_common_train.columns))
    selected_mutation_features_female = [x for x in female_feature if x in mutation_common_train.columns and x in tcga_mutation_common_train.columns]

    feature_names = ['CYT1', 'CYT2',
       'IFNr', 'TLS', 'TIS', 'TIP Hot', 'TIP Cold', 'CS Polarity', 'IMPRES', 'SIA']

    selected_features_gene = selected_gene_features + cols_to_move1
    selected_features_mut = selected_mutation_features + cols_to_move
    selected_features_gene_male = selected_gene_features_male + cols_to_move1
    selected_features_mut_male = selected_mutation_features_male + cols_to_move
    selected_features_gene_female = selected_gene_features_female + cols_to_move1
    selected_features_mut_female = selected_mutation_features_female + cols_to_move

    gene_common_train, gene_common_train_male, gene_common_train_female = gene_common_train[selected_features_gene], gene_common_train_male[selected_features_gene_male], gene_common_train_female[selected_features_gene_female]
    mutation_common_train, mutation_common_train_male, mutation_common_train_female = mutation_common_train[selected_features_mut], mutation_common_train_male[selected_features_mut_male], mutation_common_train_female[selected_features_mut_female]
    gene_common_test, gene_common_test_male, gene_common_test_female = gene_common_test[selected_features_gene], gene_common_test_male[selected_features_gene_male], gene_common_test_female[selected_features_gene_female]
    mutation_common_test, mutation_common_test_male, mutation_common_test_female = mutation_common_test[selected_features_mut], mutation_common_test_male[selected_features_mut_male], mutation_common_test_female[selected_features_mut_female]
    tcga_gene_common_train, tcga_gene_common_train_male, tcga_gene_common_train_female = tcga_gene_common_train[selected_features_gene], tcga_gene_common_train_male[selected_features_gene_male], tcga_gene_common_train_female[selected_features_gene_female]
    tcga_mutation_common_train, tcga_mutation_common_train_male, tcga_mutation_common_train_female = tcga_mutation_common_train[selected_features_mut], tcga_mutation_common_train_male[selected_features_mut_male], tcga_mutation_common_train_female[selected_features_mut_female]


    modalities = ['gene+mut', 'gene', 'mut'] # 'gene+mut+tme'
    all_outputs = []
    all_scores = []
    all_models = {}
    for modality in modalities:
        print("modality: ", modality)
        print("---- training merged model ----")

        if args.pretrain == 'TCGA':
            merged_model = pretrain_tcga(tcga_gene_common_train, tcga_mutation_common_train, tcga_clin_common_train, args.num_epochs, args.batch_size, args.lr, args.epsilon, 
                            args.hidden_dim, args.feature_dim,
                            top_tme_scores = feature_names, interaction = args.interaction, modality = modality, cancer = args.cancer,
                            monosex = None, trt = args.TRT, lodo = False)
        else:
            merged_model = [None, None, None]

        merged_model_ft = finetune(merged_model[0], merged_model[1], merged_model[2], gene_common_train, mutation_common_train, clin_common_train, 
                        args.num_epochs, args.batch_size, args.lr, args.epsilon, 
                        args.hidden_dim, args.feature_dim,
                        top_tme_scores = feature_names, interaction = args.interaction, modality = modality, cancer = args.cancer,
                        monosex = None, trt = args.TRT, lodo = True, param_grids = param_grids)


        print("---- training male model ----")

        if args.pretrain == 'TCGA':
            male_model = pretrain_tcga(tcga_gene_common_train_male, tcga_mutation_common_train_male, tcga_clin_common_train_male, args.num_epochs, args.batch_size, args.lr, args.epsilon, 
                            args.hidden_dim, args.feature_dim,
                            top_tme_scores = feature_names, interaction = args.interaction, modality = modality, cancer = args.cancer,
                            monosex = 'Male', trt = args.TRT, lodo = False)
        else:
            male_model = [None, None, None]
        male_model_ft = finetune(male_model[0], male_model[1], male_model[2], gene_common_train_male, mutation_common_train_male, clin_common_train_male, 
                args.num_epochs, args.batch_size, args.lr, args.epsilon, 
                args.hidden_dim, args.feature_dim,
                top_tme_scores = feature_names, interaction = args.interaction, modality = modality, cancer = args.cancer,
                monosex = 'Male', trt = args.TRT, lodo = True, param_grids = param_grids)

        print("---- training female model ----")
        if args.pretrain == 'TCGA':
            female_model = pretrain_tcga(tcga_gene_common_train_female, tcga_mutation_common_train_female, tcga_clin_common_train_female, args.num_epochs, args.batch_size, args.lr, args.epsilon, 
                            args.hidden_dim, args.feature_dim,
                            top_tme_scores = feature_names, interaction = args.interaction, modality = modality, cancer = args.cancer,
                            monosex = 'Female', trt = args.TRT, lodo = False)
        else:
            female_model = [None, None, None]

        female_model_ft = finetune(female_model[0], female_model[1], female_model[2], gene_common_train_female, mutation_common_train_female, clin_common_train_female, 
                args.num_epochs, args.batch_size, args.lr, args.epsilon, 
                args.hidden_dim, args.feature_dim,
                top_tme_scores = feature_names, interaction = args.interaction, modality = modality, cancer = args.cancer,
                monosex = 'Female', trt = args.TRT, lodo = True, param_grids = param_grids)

        all_models[modality] = {
            'merged': merged_model_ft,
            'male': male_model_ft,
            'female': female_model_ft
        }


        test_fusion = prepare_test_data(mutation_common_test, clin_common_test, gene = gene_common_test)
        test_fusion_male = prepare_test_data(mutation_common_test_male, clin_common_test_male, gene = gene_common_test_male)
        test_fusion_female = prepare_test_data(mutation_common_test_female, clin_common_test_female, gene = gene_common_test_female)


        merged_output, merged_score, merged_pred = predict_fusionmodel(test_fusion, merged_model_ft[0], merged_model_ft[1], merged_model_ft[2], merged_model_ft[4], merged_model_ft[5], 
                merged_model_ft[3], modality, merged_model_ft[6], predict_cohort = clin_common_test['source'])
        male_output, male_score, male_pred = predict_fusionmodel(test_fusion_male, male_model_ft[0], male_model_ft[1], male_model_ft[2], male_model_ft[4], male_model_ft[5],
                male_model_ft[3], modality, male_model_ft[6], predict_cohort = clin_common_test_male['source'])
        female_output, female_score, female_pred = predict_fusionmodel(test_fusion_female, female_model_ft[0], female_model_ft[1], female_model_ft[2], female_model_ft[4], female_model_ft[5],
                female_model_ft[3], modality, female_model_ft[6], predict_cohort = clin_common_test_female['source'])

        output = merged_output + male_output + female_output
        output = pd.DataFrame(output).T
        output.columns = ['merged_auc', 'merged_accuracy', 'merged_f1', 'merged_cohort_auc', "merged_ga_w", 'male_auc', 'male_accuracy', 'male_f1', 'male_cohort_auc', 'male_ga_w', 'female_auc', 'female_accuracy', 'female_f1', 'female_cohort_auc', 'female_ga_w']
        output['modality'] = modality
        merged_score_df = pd.DataFrame({"sample": clin_common_test.index,  "pred": merged_pred, "modality": modality, "sex": "all", "score": merged_score})
        male_score_df = pd.DataFrame({"sample": clin_common_test_male.index, "pred": male_pred,"modality": modality, "sex": "male", "score": male_score})
        female_score_df = pd.DataFrame({"sample": clin_common_test_female.index, "pred": female_pred,"modality": modality, "sex": "female", "score": female_score})

        output['repeat']= args.repeat

        score_df = pd.concat([merged_score_df, male_score_df, female_score_df], axis = 0)
        score_df['repeat'] = [args.repeat]*score_df.shape[0]
        score_df['resp'] = score_df['sample'].map(clin_common_test['resp'])

        all_outputs.append(output)
        all_scores.append(score_df)



    final_score = pd.concat(all_scores, axis = 0, ignore_index=True)
    final_output = pd.concat(all_outputs, axis=0, ignore_index=True)
    final_output.to_csv(outdir)
    final_score.to_csv(score_outdir)
    with open(f"model/tcga_models_pretrain_{args.pretrain}.pkl", "wb") as f:
        pickle.dump(all_models, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mutation Attention")
    parser.add_argument("--hidden_dim", type = int, default = 32, help = "Self-attention input dimension")
    parser.add_argument("--feature_dim", type = int, default = 16, help = "Hidden dimension")
    parser.add_argument("--lr", type = float, default = 1e-4, help = "Learning rate")
    parser.add_argument("--epsilon", type = float, default = 1e-05, help = "Weight decay")
    parser.add_argument("--num_epochs", type = int, default = 20)
    parser.add_argument("--batch_size", type = int, default = 32)
    parser.add_argument("--temperature", type = float, default = 1)
    parser.add_argument("--repeat", type = int, help = "Repeatition")
    parser.add_argument("--TRT", type = str, default = "full", help = "treatment")
    parser.add_argument("--cancer", type = str, default = 'melanoma', help = "melanoma or allcancer")
    parser.add_argument("--interaction", type = bool, default = False, help = 'feature interact with gender, trt, cancer type')
    parser.add_argument("--pretrain", type = str, default = "None")
    args = parser.parse_args()
    main(args)
