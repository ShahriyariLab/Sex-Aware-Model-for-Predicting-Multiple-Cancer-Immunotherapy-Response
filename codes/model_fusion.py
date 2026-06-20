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
import torch.utils.data as du
import random
os.environ["SCIPY_ARRAY_API"] = "1"

from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, f1_score, confusion_matrix, precision_recall_curve, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold, cross_validate, RepeatedStratifiedKFold
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import pickle

import numpy as np
import requests
import warnings, copy
import concurrent.futures
import json

from itertools import cycle

from layers import convert_to_sn, remove_all_normalization_layers

import argparse
from processing import get_all_score, set_seed, zscore_norm, get_2dpos, prepare_mut_data, encode_domain, remove_nonmutate, prepare_test_data, _to_tensor_df, best_threshold_macro_f1, quantile_transformation, cohort_align_zscore, quantile_norm, combat_correction, cohort_align_minmax
from selection import feature_selection
from model_utils import SelfAttention, gene_model, mut_model, tme_extractor, Classifier_fusion, DomainDiscriminator, compute_group_attributions_fusion, mmd_loss, BalancedBatchSampler, EarlyStopping, ResidualModel
warnings.filterwarnings("ignore")

device = (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
print("device: ", device)


def fusion_model_train(gene, pos, clin, num_epochs, batch_size, learning_rate, epsilon, hidden_dim, feature_dim, temperature, top_tme_scores,
                        interaction = True, modality = "gene+mut+tme", cancer = "allcancer", monosex = None, trt = None, lodo = False):

    fold_metrics = []
    scores_all = {"logit":[], "modality": []}
    score_ind_all = []
    all_thresholds = []
    label = 'resp'

    monocancer = False
    if (cancer == 'allcancer') or ("+" in cancer):
        monocancer = False
    
    monotrt = False
    if trt == 'PD1':
        monotrt = True

    test_size = 0.2
    if monosex == 'Female':
        test_size = 0.3
    for pack in prepare_mut_data(pos, clin, label_id=label, k_fold = 1, test_size=test_size, gene=gene, tme_feature = top_tme_scores):

        X_train_df, y_train, gene_train_df, dom_train1 = pack['X_train'], pack['y_train'], pack['gene_train'], pack['dom_train1']
        X_tr, y_tr, _, gene_tr = _to_tensor_df(X_train_df, y_train, dom_train1, device, gene_train_df)

        target, target_response, gene_target, target_dom = pack['X_val'], pack['y_val'], pack['gene_val'], pack['dom_val1']
        print("validation distribution: ", target_response.value_counts())
        target, target_response, target_dom, gene_target = _to_tensor_df(target, target_response, target_dom, device, gene_target)

        feature_extractor_gene = gene_model(gene_tr.shape[1], hidden_dim, feature_dim, interaction = interaction, monosex = monosex, 
                                monocancer = monocancer, monotrt = monotrt)
        feature_extractor_mut = mut_model(X_tr.shape[1], hidden_dim, feature_dim, interaction = interaction, monosex = monosex, 
                                monocancer = monocancer, monotrt = monotrt)

        feature_extractor_gene = feature_extractor_gene.to(device)
        feature_extractor_mut = feature_extractor_mut.to(device)
        n_domain = 1
        if cancer == 'allcancer':
            n_domain = len(clin['cancer'].unique())
        classifier = Classifier_fusion(hidden_dim, feature_dim, modality = modality, dropout = 0.3, n_cancer = n_domain, sex = monosex).to(device)

        sampler = BalancedBatchSampler(y_tr, batch_size=batch_size)
        source_loader1 = du.DataLoader(du.TensorDataset(X_tr, y_tr, gene_tr), batch_sampler=sampler)

        optimizer = optim.Adam(list(feature_extractor_gene.parameters()) + list(feature_extractor_mut.parameters()) +
                                list(classifier.parameters()), lr=learning_rate, eps = epsilon, weight_decay=1e-03)

        # Loss functions
        pos_class = (y_tr == 1).sum()
        neg_class = (y_tr == 0).sum()
        pos_weight = neg_class / (pos_class + 1e-8)   # avoid division by zero
        pos_weight = torch.tensor(pos_weight, dtype=torch.float32).to(device)
        classification_criterion = nn.BCEWithLogitsLoss(pos_weight = pos_weight)

    # -----------------------------
    # Training Loop
    # -----------------------------
        best_val_auc = -float("inf")
        best_f1 = -float("inf")
        best_scores = None
        early_stopper = EarlyStopping(patience=5, mode="max", min_delta=1e-4)
        if monosex == 'Female':
            num_epochs = 100
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
            
            feature_extractor_gene.eval()
            feature_extractor_mut.eval()
            classifier.eval()
            
            with torch.no_grad():
                outputs_mut, outputs_sex, _ = feature_extractor_mut(target)
                outputs_gene = feature_extractor_gene(gene_target)

                outputs = classifier(outputs_gene, outputs_mut, outputs_sex)
                probs = outputs
                probs = outputs/temperature
                probs = torch.sigmoid(probs).squeeze()
                
                probs1 = probs.cpu().numpy()
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

            f1 = f1_score(y_true, y_pred, average='macro')
            if f1 > best_f1:
                best_f1 = f1
                best_threshold_f1 = optimal_threshold

            print(f"----- Validation: Epoch: {epoch + 1}, Iteration Loss: {total_loss:.4f}, ACC: {accuracy:.4f}, AUC: {auc:.4f}, F1 score: {f1:.4f}, threshold: {optimal_threshold:.4f}")
            
    score_df = None
    threshold = optimal_threshold
          
    return  feature_extractor_gene, feature_extractor_mut, classifier, threshold, score_df


def predict_fusionmodel(predict_data, feature_extractor_gene, feature_extractor_mut, classifier, 
                        threshold, temperature = 1, predict_cohort = None):

    target_mutation, target_response, target_gene1 = predict_data[0], predict_data[1], predict_data[3]
    target_mutation, target_gene1 = target_mutation.to(device), target_gene1.to(device)

    feature_extractor_gene.eval()
    feature_extractor_mut.eval()
    classifier.eval()

    with torch.no_grad():

        outputs_mut, outputs_sex, outputs_cancer = feature_extractor_mut(target_mutation)
        outputs_gene = feature_extractor_gene(target_gene1)
        features = torch.cat([outputs_gene, outputs_mut], axis = 1)
        outputs_tme = None
        features = torch.nan_to_num(features, nan=1e-06)

        outputs = classifier(outputs_gene, outputs_mut, outputs_tme, outputs_sex, outputs_cancer).view(-1, 1)
        final_logits = outputs
        probs = final_logits/temperature
        probs = torch.sigmoid(probs).squeeze()

        fpr, tpr, thresholds = roc_curve(target_response, probs.cpu().numpy())

        if threshold == np.inf:
            preds = (probs >= 0.5).long()
        else:
            preds = (probs >= threshold).long()

        y_true = target_response
        y_pred = preds.detach().cpu().numpy()
        y_prob = probs.cpu().numpy()
        print(y_prob[:10])

    p_pos = probs[target_response==1].mean()
    p_neg = probs[target_response==0].mean()
    print("predicted prob for pos and neg", p_pos, p_neg)
    accuracy = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    print(f"Confusion Matrix → TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")
    f1 = f1_score(y_true, y_pred, average='macro')
    print("AUC: {:.4f}, accuracy: {:.4f}, f1 score: {:.4f}, threshold: {:.4f}".format(auc, accuracy, f1, threshold))
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
    return output, y_prob, y_pred #score.detach().cpu().tolist()


def main(args):
    print(f"------ repeat {args.repeat} ------")
    set_seed(args.repeat)
    outdir = "output/fusion_out_" + args.TRT + "_int_" + str(args.interaction) + "_" + args.cancer + "_" + str(int(args.repeat)) + ".csv"
    score_outdir = "output/fusion_score_" + args.TRT + "_" + str(int(args.repeat)) + ".csv"

    gene_norm_union_minmax = pd.read_csv("data/sample_expression.csv", index_col = 0)
    gene_norm_clin_pd1 = pd.read_csv("data/combined_clin_tme.csv", index_col = 0)
    if args.cancer != "allcancer":
        if "+" in args.cancer:
            selected_cancer = [i.strip() for i in args.cancer.split("+")]
            gene_norm_clin_pd1 = gene_norm_clin_pd1[gene_norm_clin_pd1['cancer'].isin(selected_cancer)]
        else:
            gene_norm_clin_pd1 = gene_norm_clin_pd1[gene_norm_clin_pd1['cancer'] == args.cancer]
    gene_norm_clin_pd1 = gene_norm_clin_pd1[~gene_norm_clin_pd1['Gender'].isna()]

    gene_pos_grouped_pd1 = pd.read_csv("data/sample_mutation.csv", index_col = 0)
    gene_norm_union_minmax = gene_norm_union_minmax.loc[:,gene_norm_union_minmax.columns.isin(gene_norm_clin_pd1.index)]
    gene_pos_grouped_pd1 = gene_pos_grouped_pd1[gene_pos_grouped_pd1.index.isin(gene_norm_clin_pd1.index)]

    if args.correction:
        batch = gene_norm_clin_pd1.loc[gene_norm_union_minmax.columns]['source']
        gene_norm_union_minmax = combat_correction(gene_norm_union_minmax, batch)

    gene_pos_grouped_pd1 = remove_nonmutate(gene_pos_grouped_pd1)
    pd1_samples = list(set(gene_pos_grouped_pd1.index) & set(gene_norm_clin_pd1.index))
    pd1_genesamples = list(set(gene_norm_union_minmax.columns) & set(gene_norm_clin_pd1.index))
    gene_pos_grouped_pd1 = gene_pos_grouped_pd1.loc[pd1_samples]
    gene_norm_union_minmax = gene_norm_union_minmax[pd1_genesamples]
    gene_norm_clin = gene_norm_clin_pd1.loc[pd1_samples]
    gene_pos_grouped = gene_pos_grouped_pd1
    gene_pos_grouped = gene_pos_grouped.fillna(0)

    gene_norm_clin_gene = gene_norm_clin_pd1.loc[pd1_genesamples]
    gene_norm_union = gene_norm_union_minmax
    gene_norm_union = gene_norm_union.fillna(0)
    gene_norm_union = gene_norm_union.T


    ### prepare training and heldout set

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

    gene_pos_grouped['sex'] = gene_norm_clin['Gender'].map({'M': 0, 'F':1})
    gene_pos_grouped['total_count'] = gene_norm_clin['tm_norm']
    gene_pos_grouped['msi'] = gene_norm_clin['msi_status']

    gene_norm_clin = gene_norm_clin.rename(columns={"MET": "MET_TUMOR"})

    gene_pos_grouped['cancer_type'] = gene_norm_clin['cancer'].map({
        "melanoma": 0,
        "ccrcc": 1,
        "nsclc": 2,
        "bladder": 3
    })
    gene_norm_union['cancer_type'] = gene_norm_clin_gene['cancer'].map({
        "melanoma": 0,
        "ccrcc": 1,
        "nsclc": 2,
        "bladder": 3
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

    gene_norm_clin = encode_domain(gene_norm_clin, 'trt')
    gene_pos_grouped['trt1'] = gene_norm_clin['domain_labels1']
    gene_norm_union['sex'] = gene_norm_clin_gene['Gender'].map({'M': 0, 'F':1})
    gene_norm_union['msi'] = gene_norm_clin_gene['msi_status']
    gene_norm_clin_gene = encode_domain(gene_norm_clin_gene, 'trt')
    gene_norm_union['trt1'] = gene_norm_clin_gene['domain_labels1']
    gene_norm_clin_gene = gene_norm_clin_gene.rename(columns={"MET": "MET_TUMOR"})
    gene_norm_union = gene_norm_union.fillna(0)
    gene_pos_grouped = gene_pos_grouped.fillna(0)

    sample_genepos = gene_norm_union.index.intersection(gene_pos_grouped.index)
    clin_common = gene_norm_clin.loc[sample_genepos]


    with open("data/combined_heldout_samples.json") as f:
        heldout = json.load(f)

    male_heldout_fusion = heldout['male_fusion_heldout']
    male_heldout_fusion = list(set(male_heldout_fusion) & set(clin_common.index))
    female_heldout_fusion = heldout['female_fusion_heldout']
    female_heldout_fusion = list(set(female_heldout_fusion) & set(clin_common.index))
    heldouts = male_heldout_fusion + female_heldout_fusion
    heldouts = random.sample(heldouts, len(heldouts)) # shuffle male and female
    heldouts = [i for i in heldouts if i in clin_common.index]

    cols_to_move = ['total_count', 'mut_count', 'mut_max', 'mut_mean', 'cancer_type', 'trt1', 'sex']
    cols_to_move1 = ['cancer_type', 'trt1', 'sex']

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
    if args.norm == 'zscore':
        gene_common_train_norm, gene_common_test_norm = zscore_norm(gene_common_train.drop(columns = cols_to_move1), gene_common_test.drop(columns = cols_to_move1))
        gene_common_train_norm_male, gene_common_test_norm_male = zscore_norm(gene_common_train_male.drop(columns = cols_to_move1), gene_common_test_male.drop(columns = cols_to_move1))
        gene_common_train_norm_female, gene_common_test_norm_female = zscore_norm(gene_common_train_female.drop(columns = cols_to_move1), gene_common_test_female.drop(columns = cols_to_move1))
        gene_common_train_norm_female1 = zscore_norm(gene_common_train_female1)
    elif args.norm == 'minmax':
        gene_common_train_norm, gene_common_test_norm = zscore_norm(gene_common_train.drop(columns = cols_to_move1), gene_common_test.drop(columns = cols_to_move1), method = 'minmax')
        gene_common_train_norm_male, gene_common_test_norm_male = zscore_norm(gene_common_train_male.drop(columns = cols_to_move1), gene_common_test_male.drop(columns = cols_to_move1), method = 'minmax')
        gene_common_train_norm_female, gene_common_test_norm_female = zscore_norm(gene_common_train_female.drop(columns = cols_to_move1), gene_common_test_female.drop(columns = cols_to_move1), method = 'minmax')
        gene_common_train_norm_female1 = zscore_norm(gene_common_train_female1, method = 'minmax')    
    elif args.norm == 'quantile':
        gene_common_train_norm, gene_common_test_norm = quantile_norm(gene_common_train.drop(columns = cols_to_move1), gene_common_test.drop(columns = cols_to_move1))
        gene_common_train_norm_male, gene_common_test_norm_male = quantile_norm(gene_common_train_male.drop(columns = cols_to_move1), gene_common_test_male.drop(columns = cols_to_move1))
        gene_common_train_norm_female, gene_common_test_norm_female = quantile_norm(gene_common_train_female.drop(columns = cols_to_move1), gene_common_test_female.drop(columns = cols_to_move1))
        gene_common_train_norm_female1 = quantile_norm(gene_common_train_female1)            
    elif args.norm == 'cohort_zscore':    
        gene_common_train_norm, gene_common_test_norm = cohort_align_zscore(gene_common_train.drop(columns = cols_to_move1), clin_common_train, gene_common_test.drop(columns = cols_to_move1), clin_common_test)
        gene_common_train_norm_male, gene_common_test_norm_male = cohort_align_zscore(gene_common_train_male.drop(columns = cols_to_move1), clin_common_train_male, gene_common_test_male.drop(columns = cols_to_move1), clin_common_test_male)
        gene_common_train_norm_female, gene_common_test_norm_female = cohort_align_zscore(gene_common_train_female.drop(columns = cols_to_move1), clin_common_train, gene_common_test_female.drop(columns = cols_to_move1), clin_common_test_female)
        gene_common_train_norm_female1 = cohort_align_zscore(gene_common_train_female1, clin_common_train_female1)
    elif args.norm == 'cohort_quantile':
        gene_common_train_norm, gene_common_test_norm = quantile_transformation(gene_common_train.drop(columns = cols_to_move1), clin_common_train, gene_common_test.drop(columns = cols_to_move1), clin_common_test)
        gene_common_train_norm_male, gene_common_test_norm_male = quantile_transformation(gene_common_train_male.drop(columns = cols_to_move1), clin_common_train_male, gene_common_test_male.drop(columns = cols_to_move1), clin_common_test_male)
        gene_common_train_norm_female, gene_common_test_norm_female = quantile_transformation(gene_common_train_female.drop(columns = cols_to_move1), clin_common_train, gene_common_test_female.drop(columns = cols_to_move1), clin_common_test_female)
        gene_common_train_norm_female1 = quantile_transformation(gene_common_train_female1, clin_common_train_female1)  
    elif args.norm == 'cohort_minmax':
        gene_common_train_norm, gene_common_test_norm = cohort_align_minmax(gene_common_train.drop(columns = cols_to_move1), clin_common_train, gene_common_test.drop(columns = cols_to_move1), clin_common_test)
        gene_common_train_norm_male, gene_common_test_norm_male = cohort_align_minmax(gene_common_train_male.drop(columns = cols_to_move1), clin_common_train_male, gene_common_test_male.drop(columns = cols_to_move1), clin_common_test_male)
        gene_common_train_norm_female, gene_common_test_norm_female = cohort_align_minmax(gene_common_train_female.drop(columns = cols_to_move1), clin_common_train, gene_common_test_female.drop(columns = cols_to_move1), clin_common_test_female)
        gene_common_train_norm_female1 = cohort_align_minmax(gene_common_train_female1, clin_common_train_female1)             
    else:
        gene_common_train_norm, gene_common_test_norm = gene_common_train.drop(columns = cols_to_move1), gene_common_test.drop(columns = cols_to_move1)
        gene_common_train_norm_male, gene_common_test_norm_male = gene_common_train_male.drop(columns = cols_to_move1), gene_common_test_male.drop(columns = cols_to_move1)
        gene_common_train_norm_female, gene_common_test_norm_female = gene_common_train_female.drop(columns = cols_to_move1), gene_common_test_female.drop(columns = cols_to_move1)
        gene_common_train_norm_female1 = gene_common_train_female1  

    gene_common_train = pd.concat([gene_common_train_norm, gene_common_train[cols_to_move1]], axis = 1)
    gene_common_train_male = pd.concat([gene_common_train_norm_male, gene_common_train_male[cols_to_move1]], axis = 1)
    gene_common_train_female = pd.concat([gene_common_train_norm_female, gene_common_train_female[cols_to_move1]], axis = 1)
    gene_common_train_female1 = gene_common_train_norm_female1
    

    gene_common_test = pd.concat([gene_common_test_norm, gene_common_test[cols_to_move1]], axis = 1)
    gene_common_test_male = pd.concat([gene_common_test_norm_male, gene_common_test_male[cols_to_move1]], axis = 1)
    gene_common_test_female = pd.concat([gene_common_test_norm_female, gene_common_test_female[cols_to_move1]], axis = 1)

    mutation_common_train = gene_pos_grouped.loc[train_samples_fusion]
    mutation_common_test = gene_pos_grouped.loc[heldouts]

    mutation_common_train_male, mutation_common_train_female = mutation_common_train.loc[clin_common_train_male.index], mutation_common_train.loc[clin_common_train.index]
    mutation_common_test_male, mutation_common_test_female = mutation_common_test.loc[clin_common_test_male.index], mutation_common_test.loc[clin_common_test_female.index]

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

    selected_gene_features = list(dict.fromkeys(x for x in merged_feature if x in gene_common_train.columns))
    selected_mutation_features = [x for x in merged_feature if x in mutation_common_train.columns]
    selected_gene_features_male = list(dict.fromkeys(x for x in male_feature if x in gene_common_train.columns))
    selected_mutation_features_male = [x for x in male_feature if x in mutation_common_train.columns]
    selected_gene_features_female = list(dict.fromkeys(x for x in female_feature if x in gene_common_train.columns))
    selected_mutation_features_female = [x for x in female_feature if x in mutation_common_train.columns]


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

    feature_names = ['CYT1', 'CYT2',
       'IFNr', 'TLS', 'TIS', 'TIP Hot', 'TIP Cold', 'CS Polarity', 'IMPRES', 'SIA']

    modalities = ['gene+mut', 'gene', 'mut'] # 'gene+mut+tme'
    all_outputs = []
    all_scores = []
    all_models = {}
    for modality in modalities:
        print("modality: ", modality)
        merged_model = fusion_model_train(gene_common_train, mutation_common_train, clin_common_train, args.num_epochs, args.batch_size, args.lr, args.epsilon, 
                        args.hidden_dim, args.feature_dim, args.temperature, top_tme_scores = feature_names, interaction = args.interaction, modality = modality, cancer = args.cancer,
                        monosex = None, trt = args.TRT, lodo = True)

        print("---- training male model ----")
        male_model = fusion_model_train(gene_common_train_male, mutation_common_train_male, clin_common_train_male, args.num_epochs, args.batch_size, args.lr, args.epsilon, 
                        args.hidden_dim, args.feature_dim, args.temperature, top_tme_scores = feature_names, interaction = args.interaction, modality = modality, cancer = args.cancer,
                        monosex = 'Male', trt = args.TRT, lodo = True)

        print("---- training female model ----")

        female_model = fusion_model_train(gene_common_train_female, mutation_common_train_female, clin_common_train_female, args.num_epochs, args.batch_size, args.lr, args.epsilon, 
                        args.hidden_dim, args.feature_dim, args.temperature, top_tme_scores = feature_names, interaction = args.interaction, modality = modality, cancer = args.cancer,
                        monosex = 'Female', trt = args.TRT, lodo = True)

        all_models[modality] = {
            'merged': merged_model,
            'male': male_model,
            'female': female_model
        }

        test_fusion = prepare_test_data(mutation_common_test, clin_common_test, gene = gene_common_test)
        test_fusion_male = prepare_test_data(mutation_common_test_male, clin_common_test_male, gene = gene_common_test_male)
        test_fusion_female = prepare_test_data(mutation_common_test_female, clin_common_test_female, gene = gene_common_test_female)


        merged_output, merged_score, merged_pred = predict_fusionmodel(test_fusion, merged_model[0], merged_model[1], merged_model[2], merged_model[3],
                                                                       args.temperature, predict_cohort = clin_common_test['source'])
        male_output, male_score, male_pred = predict_fusionmodel(test_fusion_male, male_model[0], male_model[1], male_model[2], male_model[3], loc_encoding_gene_male1, loc_encoding_mut_male1, 
                                                                 args.temperature, predict_cohort = clin_common_test_male['source'])
        female_output, female_score, female_pred = predict_fusionmodel(test_fusion_female, female_model[0], female_model[1], female_model[2], female_model[3], loc_encoding_gene_female1, loc_encoding_mut_female1,
                                                                       args.temperature, predict_cohort = clin_common_test_female['source'])

        output = merged_output + male_output + female_output
        output = pd.DataFrame(output).T
        output.columns = ['merged_auc', 'merged_accuracy', 'merged_f1', 'merged_cohort_auc', 'male_auc', 'male_accuracy', 'male_f1', 'male_cohort_auc','female_auc', 'female_accuracy', 'female_f1', 'female_cohort_auc']
        output['modality'] = modality
        merged_score_df = pd.DataFrame({"sample": clin_common_test.index, "score": merged_score, "pred": merged_pred, "modality": modality, "sex": "all"})
        male_score_df = pd.DataFrame({"sample": clin_common_test_male.index, "score": male_score, "pred": male_pred, "modality": modality, "sex": "male"})
        female_score_df = pd.DataFrame({"sample": clin_common_test_female.index, "score": female_score, "pred": female_pred, "modality": modality, "sex": "female"})

 
        train_merged = prepare_test_data(mutation_common_train, clin_common_train, 'resp', gene_common_train)
        train_male = prepare_test_data(mutation_common_train_male, clin_common_train_male, 'resp', gene_common_train_male)
        train_female = prepare_test_data(mutation_common_train_female, clin_common_train_female, 'resp', gene_common_train_female)

        results_merged = compute_group_attributions_fusion(merged_model[0], merged_model[1], merged_model[2], x_batch=train_merged)
        results_male = compute_group_attributions_fusion(male_model[0], male_model[1], male_model[2], x_batch=train_male)
        results_female = compute_group_attributions_fusion(female_model[0], female_model[1], female_model[2], x_batch=train_female)

        gene_name = gene_common_train.columns
        mut_name = mutation_common_train.columns

        top_features_merged = {"gene": [], "gene_score": [], "mut": [], "mut_score": []}
        top_features_male = {"gene": [], "gene_score": [], "mut": [], "mut_score": []}
        top_features_female = {"gene": [], "gene_score": [], "mut": [], "mut_score": []}
        for key, vals_tensor in results_merged.items():
            if not isinstance(vals_tensor, torch.Tensor) or vals_tensor.numel() == 0:
                continue
            vals, idx = torch.topk(vals_tensor, k=min(100, vals_tensor.numel()))
            idx = idx.tolist()
            vals = vals.detach().cpu().tolist()
            if key == "gene_cont":
                valid_idx = [i for i in idx if i < len(gene_name)]
                top_features_merged["gene"].extend([gene_name[i] for i in valid_idx])
                top_features_merged["gene_score"].extend(vals)
            elif key == "mut_cont":
                valid_idx = [i for i in idx if i < len(mut_name)]
                top_features_merged["mut"].extend([mut_name[i] for i in valid_idx])
                top_features_merged["mut_score"].extend(vals)

        for key, vals_tensor in results_male.items():
            if not isinstance(vals_tensor, torch.Tensor) or vals_tensor.numel() == 0:
                continue
            vals, idx = torch.topk(vals_tensor, k=min(100, vals_tensor.numel()))
            idx = idx.tolist()
            vals = vals.detach().cpu().tolist()
            if key == "gene_cont":
                valid_idx = [i for i in idx if i < len(gene_name)]
                top_features_male["gene"].extend([gene_name[i] for i in valid_idx])
                top_features_male["gene_score"].extend(vals)
            elif key == "mut_cont":
                valid_idx = [i for i in idx if i < len(mut_name)]
                top_features_male["mut"].extend([mut_name[i] for i in valid_idx])
                top_features_male["mut_score"].extend(vals)

        for key, vals_tensor in results_female.items():
            if not isinstance(vals_tensor, torch.Tensor) or vals_tensor.numel() == 0:
                continue
            vals, idx = torch.topk(vals_tensor, k=min(100, vals_tensor.numel()))
            idx = idx.tolist()
            vals = vals.detach().cpu().tolist()
            if key == "gene_cont":
                valid_idx = [i for i in idx if i < len(gene_name)]
                top_features_female["gene"].extend([gene_name[i] for i in valid_idx])
                top_features_female["gene_score"].extend(vals)
            elif key == "mut_cont":
                valid_idx = [i for i in idx if i < len(mut_name)]
                top_features_female["mut"].extend([mut_name[i] for i in valid_idx])
                top_features_female["mut_score"].extend(vals)

        output['merged_feature_gene'], output['male_feature_gene'], output['female_feature_gene'] = [list(top_features_merged["gene"])], [list(top_features_male["gene"])], [list(top_features_female["gene"])]
        output['merged_feature_mut'], output['male_feature_mut'], output['female_feature_mut'] = [list(top_features_merged["mut"])], [list(top_features_male["mut"])], [list(top_features_female["mut"])]
        output['merged_gene_score'], output['male_gene_score'], output['female_gene_score'] = [list(top_features_merged["gene_score"])], [list(top_features_male["gene_score"])], [list(top_features_female["gene_score"])]
        output['merged_mut_score'], output['male_mut_score'], output['female_mut_score'] = [list(top_features_merged["mut_score"])], [list(top_features_male["mut_score"])], [list(top_features_female["mut_score"])]
        
        output['repeat']= args.repeat
        all_outputs.append(output)

        score_df = pd.concat([merged_score_df, male_score_df, female_score_df], axis = 0)
        score_df['repeat'] = [args.repeat]*score_df.shape[0]
        score_df['resp'] = score_df['sample'].map(clin_common_test['resp'])
        all_scores.append(score_df)

    final_score = pd.concat(all_scores, axis = 0, ignore_index=True)
    final_score.to_csv(score_outdir)
    final_output = pd.concat(all_outputs, axis=0, ignore_index=True)
    final_output.to_csv(outdir)
    with open(f"model/fusion_models_final.pkl", "wb") as f:
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
    parser.add_argument("--correction", type = bool, default = False)
    parser.add_argument("--norm", type = str, default = 'zscore')
    args = parser.parse_args()
    main(args)
