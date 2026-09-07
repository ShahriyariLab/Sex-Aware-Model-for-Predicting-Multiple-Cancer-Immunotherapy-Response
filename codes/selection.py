import pandas as pd 
import numpy as np 
import json

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from scipy.stats import mannwhitneyu, norm
from statsmodels.stats.multitest import multipletests
from sklearn.tree import DecisionTreeClassifier
from collections import Counter
from kneed import KneeLocator


def per_dataset_deg(gene_df, clin_df, label_col='resp', source_col='source', min_n=8):
    """
    Run Mann-Whitney per dataset for each gene, return per-dataset p-values
    and effect direction (rank-biserial correlation, sign-comparable across cohorts).
    """
    sources = clin_df[source_col].unique()
    genes = gene_df.columns
    records = []

    for s in sources:
        mask = clin_df[source_col] == s
        y = clin_df.loc[mask, label_col]
        if y.nunique() < 2 or min(y.value_counts()) < 3:
            continue  # skip cohorts with too few samples in one arm

        g1 = gene_df.loc[mask & (clin_df[label_col]==1)]
        g0 = gene_df.loc[mask & (clin_df[label_col]==0)]
        n1, n0 = len(g1), len(g0)

        for gene in genes:
            x1, x0 = g1[gene].dropna(), g0[gene].dropna()
            if len(x1) < 3 or len(x0) < 3:
                continue
            stat, p = mannwhitneyu(x1, x0, alternative='two-sided')
            # rank-biserial correlation: effect size + direction, comparable across cohorts
            rbc = 1 - (2*stat) / (n1*n0)
            records.append({'gene': str(gene), 'source': str(s), 'pvalue': float(p), 'effect': float(rbc), 'n1': n1, 'n0': n0})

    return pd.DataFrame(records)


def meta_combine(deg_long, min_datasets=3, consistent_direction=True):
    """
    Combine per-dataset results per gene via Stouffer's method (sample-size weighted),
    require consistent direction across contributing cohorts.
    """
    results = []
    for gene, sub in deg_long.groupby('gene'):
        if len(sub) < min_datasets:
            continue  # gene not testable in enough cohorts

        # direction consistency check
        signs = np.sign(sub['effect'])
        if consistent_direction and signs.nunique() > 1:
            # mixed direction across cohorts -> penalize/exclude
            consistent = False
        else:
            consistent = True

        # Stouffer's Z, weighted by sqrt(n) per cohort
        z_scores = norm.isf(sub['pvalue']/2) * np.sign(sub['effect'])  # signed z
        weights = np.sqrt(sub['n1'] + sub['n0'])
        z_meta = np.sum(z_scores * weights) / np.sqrt(np.sum(weights**2))
        p_meta = 2 * norm.sf(abs(z_meta))

        results.append({
            'gene': gene,
            'n_datasets': len(sub),
            'p_meta': p_meta,
            'direction_consistent': consistent,
            'mean_effect': sub['effect'].mean()
        })

    out = pd.DataFrame(results)
    out = out[~out['gene'].isin(['cancer_type', 'msi', 'trt1', 'total_count'])]
    valid = out['direction_consistent']  # only keep genes with consistent direction
    out.loc[valid, 'adj_p_meta'] = np.nan
    out.loc[valid, 'adj_p_meta'] = multipletests(out.loc[valid, 'p_meta'], method='fdr_bh')[1]
    return out.sort_values('adj_p_meta')



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



def get_lasso_genes(gene_train, mut_train, clin_train, label_col, top_n, ref_genes = None, male_gene = None, merged_gene = None):

    X = gene_train
    y = clin_train
    strat = y['source'].astype(str) + "_" + y[label_col].astype(str)
    Z = mut_train

    n_iter = 5

    counter = Counter()
    sss = StratifiedShuffleSplit(
        n_splits=n_iter,
        test_size=0.1,
        random_state=123
    )

    for i, (train_idx, val_idx) in enumerate(sss.split(X, strat), start=1):
        print(f"Running split {i}/{n_iter}")

        X_train = X.iloc[train_idx]
        Z_train = Z.loc[Z.index.isin(X_train.index)]
        y_train = y.iloc[train_idx]

        deg_long = per_dataset_deg(X_train, y_train)
        meta_results = meta_combine(deg_long)
        meta_results = meta_results[(meta_results['direction_consistent'] == True) & (meta_results['adj_p_meta'] < 0.1)]

        if ref_genes is not None:
            deg_genes = list(set(list(meta_results['gene']) + ref_genes) & set(X.columns))
        else:
            deg_genes = list(set(list(meta_results['gene'])) & set(X.columns))

        # selected = get_deg_feature(
        #     X_train,
        #     Z_train,
        #     y_train,
        #     top_n=top_n,
        #     label_col=label_col
        # )
        # deg_genes = selected['expr']

        if male_gene is not None:
            deg_genes = list((set(meta_results.index) | set(male_gene) | set(merged_gene)) & set(X_train.columns))
        
        X_deg = X_train[deg_genes]

        # model = LogisticRegression(
        #         penalty='elasticnet',
        #         l1_ratio=0.5,
        #         solver='saga',
        #         C=0.5,
        #         max_iter=5000,
        #         random_state=123
        #     )
        model = LogisticRegression(penalty='l1', solver='saga', C=2, max_iter=5000, random_state=123)
        model.fit(X_deg, y_train[label_col])
        coef = model.coef_[0]
        selected_genes = np.array(deg_genes)[coef != 0]
        counter.update(selected_genes)


    freq_df = pd.DataFrame({'gene': list(counter.keys()), 'selection_freq': [counter[g] / n_iter for g in counter.keys()]})
    freq_df = freq_df.sort_values(
        'selection_freq',
        ascending=False
    )

    freq_df_sorted = freq_df.sort_values('selection_freq', ascending=False).reset_index(drop=True)
    freq_df_sorted['rank'] = np.arange(1, len(freq_df_sorted) + 1)
    x = freq_df_sorted['rank'].values
    y = freq_df_sorted['selection_freq'].values
    print(freq_df_sorted.head(30))
    # # Elbow detection: curve is decreasing, likely convex (fast drop then long flat tail)
    # kl = KneeLocator(x, y, curve='convex', direction='decreasing', S=1.0)
    # elbow_rank = kl.knee

    stable_genes = freq_df_sorted[freq_df_sorted['selection_freq'] >= 0.5]['gene'].astype(str).tolist() #freq_df_sorted.iloc[:elbow_rank]['gene'].tolist() 
    print("stable_genes: ", stable_genes)

    # if len(stable_genes) < top_n:
    #     stable_genes = (
    #         freq_df
    #         .head(top_n)['gene']
    #         .tolist()
    #     )

    return stable_genes


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


def feature_selection(gene_train, mut_train, clin_train, ref_genes = None, cohort_col = 'source', label_col = 'resp', top_n = 30, male_gene = None, merged_gene = None):
    if len(clin_train['Gender'].unique()) > 1:
        gender = 'merged'
    elif (clin_train['Gender'] == 'M').any():
        gender = 'male'
    else:
        gender = 'female'

    if gender == 'female':
        feature_genes = get_lasso_genes(gene_train, mut_train, clin_train, ref_genes = ref_genes, label_col = label_col, top_n = top_n, male_gene = male_gene, merged_gene = merged_gene)
    else:
        feature_genes = get_lasso_genes(gene_train, mut_train, clin_train, ref_genes = ref_genes, label_col = label_col, top_n = top_n)
      
    feature_genes = [str(i) for i in feature_genes if str(i) not in ['cancer_type', 'msi', 'trt1', 'total_count', 'source1']]
    print(f"--------  {len(feature_genes)} selected for {gender} model  ---------")
    #print(f"selected gene features {feature_genes}")

    return feature_genes

