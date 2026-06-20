import pandas as pd 
import numpy as np 
import json

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
from sklearn.tree import DecisionTreeClassifier
from collections import Counter


def get_deg_feature(gene_train, mut_train, clin_train, label_col, top_n = 50):


    group1 = gene_train.loc[clin_train[(clin_train[label_col] == 1)].index]
    group0 = gene_train.loc[clin_train[(clin_train[label_col] == 0)].index]

    pvals = []
    logFC = []

    for gene in gene_train.columns:
        stat,p = mannwhitneyu(group1[gene], group0[gene])
        pvals.append(p)
        logFC.append(group1[gene].mean() - group0[gene].mean())

    # FDR correction
    pvals = np.array(pvals, dtype=float)
    valid_mask = ~np.isnan(pvals)
    adj_pvals = np.full_like(pvals, np.nan)
    pvals = np.array(pvals, dtype=float)
    _, adj_valid, _, _ = multipletests(pvals[valid_mask], method="fdr_bh")
    adj_pvals[valid_mask] = adj_valid

    results = pd.DataFrame({
        "gene":gene_train.columns,
        "logFC":logFC,
        "pvalue":pvals,
        "adj_pvalue":adj_pvals
    })
    results['adj_pvalue'] = results['adj_pvalue'].fillna(1)
    num_sig = len(results[results['adj_pvalue'] < 0.1])
    selected_genes = results.nsmallest(max(top_n, num_sig), 'adj_pvalue')['gene']

    clin_mut_train = clin_train[clin_train.index.isin(mut_train.index)]
    group1 = mut_train.loc[clin_mut_train[(clin_mut_train[label_col] == 1)].index]
    group0 = mut_train.loc[clin_mut_train[(clin_mut_train[label_col] == 0)].index]

    pvals = []
    logFC = []

    for gene in mut_train.columns:
        stat,p = mannwhitneyu(group1[gene], group0[gene]) 
        pvals.append(p)
        logFC.append(group1[gene].mean() - group0[gene].mean())

    # FDR correction
    valid_mask = ~np.isnan(pvals)
    adj_pvals = np.full_like(pvals, np.nan)
    pvals = np.array(pvals, dtype=float)
    _, adj_valid, _, _ = multipletests(pvals[valid_mask], method="fdr_bh")
    adj_pvals[valid_mask] = adj_valid

    results_mut = pd.DataFrame({
        "gene":mut_train.columns,
        "logFC":logFC,
        "pvalue":pvals,
        "adj_pvalue":adj_pvals
    })
    results_mut['adj_pvalue'] = results_mut['adj_pvalue'].fillna(1)

    num_sig = len(results_mut[results_mut['adj_pvalue'] < 0.1])
    selected_mut_genes = results_mut.nsmallest(max(top_n, num_sig), 'adj_pvalue')['gene']

    selected = {'expr': list(selected_genes), 'mutation': list(selected_mut_genes)}

    for k,v in selected.items():
        selected[k] = [i for i in v if i not in ['cancer_type', 'msi', 'trt1', 'total_count']]
    return selected



def get_lasso_genes(gene_train, mut_train, clin_train, label_col, top_n, male_gene = None, merged_gene = None):

    X = gene_train
    y = clin_train
    strat = y['source'].astype(str) + "_" + y[label_col].astype(str)
    Z = mut_train

    n_iter = 30

    counter = Counter()
    sss = StratifiedShuffleSplit(
        n_splits=n_iter,
        test_size=0.3,
        random_state=123
    )

    for i, (train_idx, val_idx) in enumerate(sss.split(X, strat), start=1):
        print(f"Running split {i}/{n_iter}")

        X_train = X.iloc[train_idx]
        Z_train = Z.loc[Z.index.isin(X_train.index)]
        y_train = y.iloc[train_idx]

        selected = get_deg_feature(
            X_train,
            Z_train,
            y_train,
            top_n=top_n,
            label_col=label_col
        )
        deg_genes = selected['expr']

        if male_gene is not None:
            deg_genes = list((set(selected['expr']) | set(male_gene) | set(merged_gene)) & set(X_train.columns))

        X_deg = X_train[deg_genes]

        model = LogisticRegression(
                penalty='elasticnet',
                l1_ratio=0.5,
                solver='saga',
                C=0.5,
                max_iter=5000,
                random_state=123
            )

        model.fit(X_deg, y_train[label_col])
        coef = model.coef_[0]
        selected_genes = np.array(deg_genes)[coef != 0]
        counter.update(selected_genes)


    freq_df = pd.DataFrame({'gene': list(counter.keys()), 'selection_freq': [counter[g] / n_iter for g in counter.keys()]})
    freq_df = freq_df.sort_values(
        'selection_freq',
        ascending=False
    )
    print(freq_df.head(top_n))


    stable_genes = freq_df[freq_df['selection_freq'] >= 0.5]['gene'].tolist()

    if len(stable_genes) < top_n:
        stable_genes = (
            freq_df
            .head(top_n)['gene']
            .tolist()
        )

    selected['expr'] = stable_genes
    return selected


def get_cohort_stable_genes(gene_train, clin_train, gene_list, label_col = 'resp', cohort_col = 'source'):

    genes = gene_list
    results = []
    for gene in genes:
        for cohort in clin_train[cohort_col].unique():
            idx = clin_train[cohort_col] == cohort
            y = clin_train.loc[idx, label_col]
            x = gene_train.loc[idx, gene]

            # remove NA
            valid = ~(x.isna() | y.isna())
            x = x[valid]
            y = y[valid]

            # need both classes
            if len(np.unique(y)) < 2:
                continue
            # at least three samples in each class
            class_counts = pd.Series(y).value_counts()
            if (class_counts < 3).any():
                continue

            auc = roc_auc_score(y, x)
            # fold change direction
            mean_resp = x[y == 1].mean()
            mean_nonresp = x[y == 0].mean()
            direction = mean_resp - mean_nonresp
            results.append({
                'gene': gene,
                'cohort': cohort,
                'auc': auc,
                'direction': direction,
                'mean_resp': mean_resp,
                'mean_nonresp': mean_nonresp
            })

    results_df = pd.DataFrame(results)
    results_df = results_df.groupby('gene')['auc'].agg(['mean','std','min','max'])
    stable_cohort_genes = results_df[(results_df['min'] > min(np.median(results_df['min']), 0.5))] # & (results_df['mean'] > min(np.median(results_df['mean']), 0.5))] 
    stable_cohort_genes = list(stable_cohort_genes.index)

    return stable_cohort_genes


def feature_selection(gene_train, mut_train, clin_train, cohort_col = 'source', label_col = 'resp', top_n = 50, male_gene = None, merged_gene = None):
    if len(clin_train['Gender'].unique()) > 1:
        gender = 'merged'
    elif (clin_train['Gender'] == 'M').any():
        gender = 'male'
    else:
        gender = 'female'

    if gender == 'female':
        feature_dict = get_lasso_genes(gene_train, mut_train, clin_train, label_col = label_col, top_n = top_n, male_gene = male_gene, merged_gene = merged_gene)
        # cohort_features = get_cohort_stable_genes(gene_train, clin_train, feature_dict['expr'], label_col = label_col, cohort_col = cohort_col)
        # feature_dict['expr'] = cohort_features
    else:
        feature_dict = get_lasso_genes(gene_train, mut_train, clin_train, label_col = label_col, top_n = top_n)
      
    for k,v in feature_dict.items():
        feature_dict[k] = [str(i) for i in v if str(i) not in ['cancer_type', 'msi', 'trt1', 'total_count', 'source1']]
    print(f"--------  {len(feature_dict['expr'])} selected for {gender} model  ---------")
    print(f"selected gene features {feature_dict['expr']}")

    return feature_dict

