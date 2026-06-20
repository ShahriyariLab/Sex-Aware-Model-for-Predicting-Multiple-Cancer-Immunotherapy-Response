
import pandas as pd
import os
import torch
os.environ["SCIPY_ARRAY_API"] = "1"
from imblearn.over_sampling import RandomOverSampler, SMOTE, SMOTENC
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold, cross_validate, RepeatedStratifiedKFold, LeaveOneOut
import numpy as np
import requests
import json
import warnings
import random
from TMEImmune import TME_score
from tidepy.pred import TIDE
import gseapy as gp
from scipy.stats import gmean
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, QuantileTransformer
from inmoose.pycombat import pycombat_norm, pycombat_seq


warnings.filterwarnings("ignore")

# ## map symbol to entrez id
symbol2id = pd.read_csv("/home/qiluzhou_umass_edu/mutation_attention/data/symbol2entrez.csv")
symbol2id1 = symbol2id[~symbol2id['entrez_id'].isna()]
symbol2id1['entrez_id'] = symbol2id1['entrez_id'].apply(lambda x: int(x))

def map_index_id(df, merge_df, id2symbol = False, transcript = False):
    if not transcript:
        if not id2symbol:
            df1 = df.merge(merge_df, left_index = True, right_on = 'symbol', how = "inner")
            df1.index = df1['entrez_id']
            df1 = df1.drop(columns = ['symbol', 'entrez_id'])
        else:
            df1 = df.merge(merge_df, left_index = True, right_on = 'entrez_id', how = "inner")
            df1.index = df1['symbol']
            df1 = df1.drop(columns = ['symbol', 'entrez_id', 'transcripts'])
    return df1


# length of each chromosome
# https://www.ncbi.nlm.nih.gov/grc/human/data?asm=GRCh38.p13
chrom_len = {"chr1": 248956422, "chr2": 242193529, "chr3": 198295559, "chr4": 190214555, "chr5": 181538259,
    "chr6": 170805979, "chr7": 159345973, "chr8": 145138636, "chr9": 138394717, "chr10": 133797422,
    "chr11": 135086622, "chr12": 133275309, "chr13": 114364328, "chr14": 107043718, "chr15": 101991189,
    "chr16": 90338345, "chr17": 83257441, "chr18": 80373285, "chr19": 58617616, "chr20": 64444167,
    "chr21": 46709983, "chr22": 50818468, "chrX": 156040895, "chrY": 57227415}

def get_2dpos(position_df, chromosome = 'Chromosome', start_pos = 'Start', end_pos = 'End'):
    df1 = position_df.copy()
    n = df1.shape[0]
    df1['pos2d'] = [None]*n
    for i in range(n):
        chromo = position_df.loc[i, chromosome]
        chr_str = "chr" + str(chromo)
        if chromo == "X":
            chromo = 23
        elif chromo == "Y":
            chromo = 24
        else:
            try:
                chromo = int(chromo)
            except ValueError:
                continue
        chr_len = chrom_len[chr_str]
        start_norm = round(df1.loc[i, start_pos]/chr_len, 16)
        end_norm = round(df1.loc[i, end_pos]/chr_len, 16)
        pos2d_array = np.array([chromo, start_norm, end_norm])
        df1['pos2d'][i] = pos2d_array
    return df1


def safe_transform(le, values):
    mapping = {c: i for i, c in enumerate(le.classes_)}
    unknown = len(le.classes_)
    return [mapping.get(v, unknown) for v in values]

def prepare_mut_data(pos = None, clin = None, label_id = 'CB', test_size = 0.3, k_fold = 5, gene = None, tme_feature = None, 
                        train_domains=None, test_domains=None, lodo = False): # labels is a dataframe with id and response info
    full_data = clin.merge(pos, left_index = True, right_index = True, how = 'inner').copy()
    source_labels = clin.loc[full_data.index]['source'].values
    source_labels1 = clin.loc[full_data.index]['source'].reset_index(drop=True)
    cancer_labels = clin.loc[full_data.index]['cancer'].reset_index(drop=True)

    if train_domains is not None:
        full_data.reset_index(drop=True, inplace=True)
        pos = full_data.drop(columns=clin.columns)

        # ---- gene / TME ----
        if gene is not None:
            gene = gene.loc[full_data.index]
            gene.reset_index(drop=True, inplace=True)

        dom = full_data['source']

        # ---- define LODO split ----
        train_mask = dom.isin(train_domains)
        test_mask  = dom.isin(test_domains)

        X_train_df = pos.loc[train_mask]
        y_train    = full_data.loc[train_mask, label_id]

        X_val_df   = pos.loc[test_mask]
        y_val      = full_data.loc[test_mask, label_id]

        dom_train1 = full_data['domain_labels1'].loc[train_mask].reset_index(drop=True)
        dom_val1   = full_data['domain_labels1'].loc[test_mask].reset_index(drop=True)

        if gene is not None:
            gene_train_df = gene.loc[train_mask].reset_index(drop=True)
            gene_val_df   = gene.loc[test_mask].reset_index(drop=True)
        else:
            gene_train_df = gene_val_df = None

        # ---- oversample TRAIN only ----
        ros = RandomOverSampler(random_state=123)
        if X_train_df is not None and y_train.nunique() > 1:
            X_tr_rs, y_tr_rs = ros.fit_resample(X_train_df.reset_index(drop=True), y_train.reset_index(drop=True))
        else:
            X_tr_rs = X_train_df.reset_index(drop=True)
            y_tr_rs = y_train.reset_index(drop=True)

        idx = ros.sample_indices_
        dom_tr_rs1 = dom_train1.iloc[idx].reset_index(drop=True)

        if gene is not None:
            gene_train_out = gene_train_df.iloc[idx].reset_index(drop=True)
        else:
            gene_train_out = None

        # ---- output ----
        yield {
            'fold': f"LODO_test={test_domains}",
            'X_train': pd.DataFrame(X_tr_rs, columns=X_train_df.columns),
            'y_train': pd.Series(y_tr_rs, name=label_id),
            'gene_train': gene_train_out,
            'dom_train1': dom_tr_rs1,
            'X_val': X_val_df.reset_index(drop=True),
            'y_val': y_val.reset_index(drop=True),
            'gene_val': gene_val_df,
            'dom_val1': dom_val1
        }

    else:
        if gene is not None:
            gene = gene.loc[full_data.index]
            gene = gene.reset_index(drop = True)

        full_data.reset_index(drop=True, inplace=True)
        pos = full_data.drop(columns = clin.columns)
        ros = RandomOverSampler(random_state=123)
        gene_train_out = None
        gene_val_out = None

        if test_size != 0:

            X_all   = pos.reset_index(drop = True)
            y_all    = full_data[label_id].reset_index(drop = True)

            dom_labels1_all = full_data['domain_labels1']
            if gene is not None:
                gene_all = gene.reset_index(drop=True)
            strat = full_data["source"].astype(str) + "_" + full_data[label_id].astype(str)

            if k_fold == 1:
                idx = np.arange(len(X_all))

                if strat is not None:
                    t_tr_idx, t_val_idx = train_test_split(
                        idx, test_size=test_size, random_state=123, stratify=strat
                    )
                else:
                    t_tr_idx, t_val_idx = train_test_split(
                        idx, test_size=test_size, random_state=123
                    )
                split_iter = [(t_tr_idx, t_val_idx)]
            else:
                # Original K-fold logic
                if strat is not None:
                    splitter = RepeatedStratifiedKFold(n_splits=k_fold, n_repeats=3, random_state=123)
                    split_iter = splitter.split(np.zeros(len(X_all)), strat.values)
                else:
                    splitter = KFold(n_splits=k_fold, shuffle=True, random_state=123)
                    split_iter = splitter.split(np.zeros(len(X_all)))

            ros = RandomOverSampler(random_state=123)

            for fold_idx, (t_tr_idx, t_val_idx) in enumerate(split_iter, start=1):
                # Target train/val by iloc
                X_tr = X_all.iloc[t_tr_idx]
                X_val = X_all.iloc[t_val_idx]
                y_tr = y_all.iloc[t_tr_idx]
                y_val = y_all.iloc[t_val_idx]

                # ---- build TRAIN = all source + target-train-fold (+ optional concat) ----
                X_train_df = X_tr
                y_train = y_tr

                # domain labels for training rows (before any resampling)
                dom_train1_out = dom_labels1_all.iloc[t_tr_idx].reset_index(drop=True)
                if gene is not None:
                    gene_tr = gene.iloc[t_tr_idx]

                X_res_list = []
                y_res_list = []
                gene_res_list = []
                gene_train_out = None

                if gene is not None:
                    gene_tr = gene_tr.add_prefix("gene_")
                    X_tr = pd.concat([X_tr, gene_tr], axis=1).fillna(0)
                
                # skip if too small or single class
                if len(y_tr) < 5 or len(y_tr.unique()) < 2:
                    X_res_list.append(X_tr.drop(columns=gene_tr.columns))
                    y_res_list.append(y_tr)
                    if gene is not None:
                        gene_res_list.append(gene_tr)
                    continue
                
                class_counts = np.bincount(y_tr)
                minority_class_size = np.min(class_counts)
                if minority_class_size > 1:
                    k_neighbors = min(3, minority_class_size - 1)
                    smote = SMOTE(
                        k_neighbors=k_neighbors,
                        random_state=123
                    )
                    X_c_res, y_c_res = smote.fit_resample(X_tr, y_tr)
                else:
                    X_c_res, y_c_res = X_tr, y_tr
                
                # preserve index (optional but useful)
                X_c_res = pd.DataFrame(X_c_res, columns=X_tr.columns)
                if gene is not None:
                    #gene_c_res = X_c_res.filter(like="gene_")
                    gene_c_res = X_c_res[gene_tr.columns]
                    X_c_res = X_c_res.drop(columns=gene_c_res.columns)
                else:
                    X_c_res = X_c_res

                y_c_res = pd.Series(y_c_res)
                
                X_res_list.append(X_c_res)
                gene_res_list.append(gene_c_res)
                y_res_list.append(y_c_res)

                # combine all cohorts
                X_train_out = pd.concat(X_res_list).reset_index(drop=True)
                perm = np.random.permutation(len(X_train_out))
                if gene is not None:
                    gene_train_out = pd.concat(gene_res_list).reset_index(drop=True)
                    gene_train_out = gene_train_out.iloc[perm].reset_index(drop=True)
                y_train_out = pd.concat(y_res_list).reset_index(drop=True)

                X_train_out = X_train_out.iloc[perm].reset_index(drop=True)
                y_train_out = y_train_out.iloc[perm].reset_index(drop=True)

                # ---- VALIDATION = target-val-fold only (no resampling) ----
                X_val_out = X_val.reset_index(drop=False)  # keep original index to map domain labels accurately if needed
                # map domain labels for val by original index
                dom_val1_out = dom_labels1_all.loc[X_val_out['index']].reset_index(drop=True)
                if gene is not None:
                    gene_val_out = gene.loc[X_val_out['index']].reset_index(drop=True)
                X_val_out = X_val_out.drop(columns=['index']).reset_index(drop=True)
                y_val_out = y_val.reset_index(drop=True)

                yield {
                        'fold': fold_idx,
                        'X_train': X_train_out,
                        'y_train': y_train_out,
                        'gene_train': gene_train_out,
                        'dom_train1': dom_train1_out,
                        'X_val': X_val_out,
                        'y_val': y_val_out,
                        'gene_val': gene_val_out,
                        'dom_val1': dom_val1_out,
                        'train_idx_tgt': t_tr_idx,
                        'val_idx_tgt': t_val_idx, # idx_balanced
                    }

        else:
            X_all   = pos.reset_index(drop = True)
            y_all    = full_data[label_id].reset_index(drop = True)

            dom_labels1_all = full_data['domain_labels1']
            if gene is not None:
                gene_all = gene.reset_index(drop=True)

            strat = (y_all.astype(str) + "_" + full_data['source'].astype(str)) if y_all.nunique() > 1 else None
            if k_fold == 1:
                # Single "fold": train = all, val = all
                idx = np.arange(X_all.shape[0])
                split_iter = [(idx, idx)]
            else:
                if lodo:
                    unique_domains = np.unique(source_labels)
                    split_iter = []
                    for d in unique_domains:
                        train_idx = np.where(source_labels != d)[0]
                        val_idx = np.where(source_labels == d)[0]
                        # Optional: skip if too small
                        if len(val_idx) < 10:
                            continue
                        if len(np.unique(y_all[val_idx])) < 2:
                            continue
                        split_iter.append((train_idx, val_idx))

                else:
                    #splitter = StratifiedKFold(n_splits=k_fold, shuffle=True, random_state=123)
                    splitter = RepeatedStratifiedKFold(n_splits=k_fold, n_repeats=3, random_state=123)
                    split_iter = splitter.split(np.zeros(X_all.shape[0]), strat.values)

            for fold_idx, (t_tr_idx, t_val_idx) in enumerate(split_iter, start=1):
                X_tr = X_all.iloc[t_tr_idx]
                y_tr = y_all.iloc[t_tr_idx]
                source_labels_tr = source_labels1.iloc[t_tr_idx]
                X_val = X_all.iloc[t_val_idx]
                y_val = y_all.iloc[t_val_idx]
                dom_train1_out = dom_labels1_all.iloc[t_tr_idx].reset_index(drop=True)

                if gene is not None:
                    gene_tr = gene.iloc[t_tr_idx]

                X_res_list = []
                y_res_list = []
                gene_res_list = []
                gene_train_out = None
                for source in source_labels_tr.unique():
                    idx = source_labels_tr[source_labels_tr == source].index
                    
                    X_c = X_tr.loc[idx]
                    if gene is not None:
                        gene_c = gene_tr.loc[idx]
                        gene_c = gene_c.add_prefix("gene_")
                        X_c = pd.concat([X_c, gene_c], axis=1).fillna(0)
                    y_c = y_tr.loc[idx]
                    
                    # skip if too small or single class
                    if len(y_c) < 5 or len(y_c.unique()) < 2:
                        X_res_list.append(X_c.drop(columns=gene_c.columns))
                        y_res_list.append(y_c)
                        if gene is not None:
                            gene_res_list.append(gene_c)
                        continue
                    
                    class_counts = np.bincount(y_c)
                    minority_class_size = np.min(class_counts)
                    if minority_class_size > 1:
                        k_neighbors = min(3, minority_class_size - 1)
                        smote = SMOTE(
                            k_neighbors=k_neighbors,
                            random_state=123
                        )
                        X_c_res, y_c_res = smote.fit_resample(X_c, y_c)
                    else:
                        X_c_res, y_c_res = X_c, y_c
                    
                    # preserve index (optional but useful)
                    X_c_res = pd.DataFrame(X_c_res, columns=X_c.columns)
                    if gene is not None:
                        #gene_c_res = X_c_res.filter(like="gene_")
                        gene_c_res = X_c_res[gene_c.columns]
                        X_c_res = X_c_res.drop(columns=gene_c_res.columns)
                    else:
                        X_c_res = X_c_res

                    y_c_res = pd.Series(y_c_res)
                    
                    X_res_list.append(X_c_res)
                    gene_res_list.append(gene_c_res)
                    y_res_list.append(y_c_res)

                # combine all cohorts
                X_train_out = pd.concat(X_res_list).reset_index(drop=True)
                perm = np.random.permutation(len(X_train_out))
                if gene is not None:
                    gene_train_out = pd.concat(gene_res_list).reset_index(drop=True)
                    gene_train_out = gene_train_out.iloc[perm].reset_index(drop=True)
                y_train_out = pd.concat(y_res_list).reset_index(drop=True)

                X_train_out = X_train_out.iloc[perm].reset_index(drop=True)
                y_train_out = y_train_out.iloc[perm].reset_index(drop=True)

                # ---- VALIDATION = target-val-fold only (no resampling) ----
                X_val_out = X_val.reset_index(drop=False)  # keep original index to map domain labels accurately if needed
                # map domain labels for val by original index
                dom_val1_out = dom_labels1_all.loc[X_val_out['index']].reset_index(drop=True)
                if gene is not None:
                    gene_val_out = gene.loc[X_val_out['index']].reset_index(drop=True)
                X_val_out = X_val_out.drop(columns=['index']).reset_index(drop=True)
                y_val_out = y_val.reset_index(drop=True)

                yield {
                        'fold': fold_idx,
                        'X_train': X_train_out,
                        'y_train': y_train_out,
                        'gene_train': gene_train_out,
                        'dom_train1': dom_train1_out,
                        'X_val': X_val_out,
                        'y_val': y_val_out,
                        'gene_val': gene_val_out,
                        'dom_val1': dom_val1_out,
                        #'train_idx_tgt': idx,
                        'train_idx_tgt': t_tr_idx,
                        'val_idx_tgt': t_val_idx, # idx_balanced
                        'val_domain': clin.iloc[t_val_idx]['source'][0]
                    }




def prepare_test_data(pos = None, clin = None, label_id = 'resp', gene = None):
    if pos is None:
        if gene is None:
            raise ValueError("gene must be provided when pos is None")
        full_data = clin.merge(gene, left_index=True, right_index=True, how="inner").copy()
        pos_cols = None
    else:
        full_data = clin.merge(pos, left_index = True, right_index = True, how = 'inner').copy()
        pos_cols = pos.columns
    gene_train = None
    if gene is not None:
        gene_samples = gene.index.intersection(full_data.index)
        gene = gene.loc[gene_samples]
        full_data = full_data.loc[gene_samples]
        gene = gene.reset_index(drop=True)
    full_data.reset_index(drop=True, inplace=True)
    if pos_cols is not None:
        pos = full_data[pos_cols]
    else:
        pos = None

    y_train_s = full_data[label_id]
    dom_train1 = None
    if pos is not None:
        X_train_df = pos
        if 'domain_labels1' in full_data.columns:
            dom_train1 = full_data.loc[pos.index, 'domain_labels1'].reset_index(drop=True)
    else:
        X_train = None
        X_train_df = y_train_s.index  # for gene alignment only
        if 'domain_labels1' in full_data.columns:
            dom_train1 = full_data.loc[X_train_df, 'domain_labels1'].reset_index(drop=True)
    if gene is not None:
        gene_train = gene.loc[y_train_s.index].reset_index(drop = True)
        gene_train = torch.tensor(gene_train.values, dtype = torch.float32)
    domain_labels_train2 = None

    X_train = torch.tensor(X_train_df.values, dtype=torch.float32)

    y_train = torch.tensor(y_train_s.values, dtype=torch.long)
    domain_labels_train1 = None
    if 'domain_labels1' in full_data.columns:
        domain_labels_train1 = torch.tensor(dom_train1.values, dtype=torch.float32)  # numeric, unchanged
    cohort = full_data['source']
    # domain_labels_train2 = torch.tensor(dom_tr_rs1.values, dtype=torch.float32)  # numeric, unchanged

    X_test, y_test, domain_labels_test1, domain_labels_test2, gene_test = None, None, None, None, None
    return (X_train, y_train, domain_labels_train1, gene_train, cohort, domain_labels_train2, 
        X_test, y_test, domain_labels_test1, domain_labels_test2)



def encode_domain(df, col):
    vc1 = df[col].value_counts(dropna=True)       
    order1 = vc1.index.tolist()            
    df['domain_labels1'] = pd.Categorical(df[col], categories=order1, ordered=True).codes
    return df


def remove_nonmutate(df, thr = 0.01):
    n = df.shape[0]
    df_filtered = df.loc[:, (df == 1).sum(axis=0) >= thr*n]
    return df_filtered

def _to_tensor_df(X_df, y_vec_or_df, dom_series, device, gene = None):
    if X_df is not None:
        X = torch.tensor(X_df.values, dtype=torch.float32, device=device)
    else:
        X = None
    if gene is not None:
        gene = torch.tensor(gene.values, dtype = torch.float32, device=device)
    # y can be Series (classification) or DataFrame (survival) — handle both
    if hasattr(y_vec_or_df, "values"):
        y_np = y_vec_or_df.values
    else:
        y_np = np.asarray(y_vec_or_df)

    is_continuous = (
        np.issubdtype(y_np.dtype, np.floating) and
        len(np.unique(y_np)) > 10  # heuristic: many unique values
    )
    # classification expects shape (N,1) float; survival you'd adapt accordingly
    if y_np.ndim == 1:
        if is_continuous:
            y = torch.tensor(y_np, dtype=torch.float32, device=device)
        else:
            y = torch.tensor(y_np, dtype=torch.long, device=device)#.view(-1, 1)
    else:
        y = torch.tensor(y_np, dtype=torch.long, device=device)
    dom = torch.tensor(dom_series.values, dtype=torch.long, device=device)
    return X, y, dom, gene


def get_geomean_score(df, sig, name = None):
    missing = set(sig) - set(df.index)
    if not missing:
        sig_df = df.loc[sig]
    else:
        common_sig = list(set(sig) & set(df.index))
        sig_df = df.loc[common_sig]
        print(f"{name}: {missing} not in dataframe")
    
    gmeans = sig_df.apply(lambda row: gmean(row), axis=0)
    return gmeans


def get_avgmean_score(df, sig, name = None):
    missing = set(sig) - set(df.index)
    if not missing:
        sig_df = df.loc[sig]
    else:
        common_sig = list(set(sig) & set(df.index))
        sig_df = df.loc[common_sig]
        print(f"{name}: {missing} not in dataframe")
    
    avgmeans = sig_df.mean(axis = 0)
    return avgmeans

def get_ratio_score(df, sig, name = None):
    numerator = sig[1]
    denominator = sig[0]
    if denominator not in df.index:
        print(f"{name}: denominator does not exist")
        ratio_score = pd.Series(None, index=df.columns, dtype=int)
        #raise ValueError("denominator does not exist")
    elif numerator not in df.index:
        ratio_score = pd.Series(0, index=df.columns, dtype=int)
    else:
        ratio_score = df.loc[numerator]/df.loc[denominator]
    return ratio_score

def get_impres_score(df, sig, name = None):
    Gene1 = sig['Gene1']
    Gene2 = sig['Gene2']
    missing = set(Gene1 + Gene2) - set(df.index)
    scores = pd.Series(0, index=df.columns, dtype=int)
    if missing:
        print(f"{name}: these genes are missing: {missing}")
    pairs_in = 0
    for g1, g2 in zip(Gene1, Gene2):
        g1_in = g1 in df.index
        g2_in = g2 in df.index
        if g1_in and g2_in:
            scores += (df.loc[g1] > df.loc[g2]).astype(int)
            pairs_in += 1
        else:
            continue
    scores = scores * 15/pairs_in
    return scores

def get_ssgsea_score(df, sig):
    ss = gp.ssgsea(data=df,gene_sets=sig,min_size=1,outdir=None,verbose=True,sample_norm_method = "rank")
    score = ss.res2d[['Name', 'NES']]
    score.index = score['Name']
    score = score.drop(columns = 'Name')
    return score

## geometric mean

CYT1 = ['GZMA', 'PRF1']
CYT2 = ['B2M', 'HLA-A', 'HLA-B', 'HLA-C', 'CASP8']
IFNr = ['CD3D', 'IDO1', 'CIITA', 'CD3E', 'CCL5', 'GZMK', 'CD2', 'HLA-DRA', 'CXCL3', 'IL2RG', 'NKG7', 'HLA-E', 'CXCR6', 'LAG3', 'TAGAP', 'CXCL10', 'STAT1', 'GZMB']
TLS = ['BCL6', 'CD86', 'CXCR4', 'LAMP3', 'SELL', 'CCR7', 'CXCL13', 'CCL21', 'CCL19']
TIS = ['CD276', 'HLA-DQA1', 'CD274', 'IDO1', 'HLA-DRB1', 'HLA-E', 'CMKLR1', 'PDCD1LG2', 'PSMB10', 'LAG3', 'CXCL9', 'STAT1', 'CD8A', 'CCL5', 'NKG7', 'TIGIT', 'CD27', 'CXCR6']

## average mean
TIP_hot = ['CXCL9', 'CXCL10', 'CXCL11', 'CXCR3', 'CD3','CD4','CD8A','CD8B', 'CD274', 'PDCD1', 'CXCE4', 'CCL5']
TIP_cold = ['CXCL1','CXCL2', 'CCL20']

## ratio
CS_polarity = ['SPP1', 'CXCL9']

## IMPRES
IMPRES = {'Gene1': ['PDCD1','CD27', 'CTLA4', 'CD40', 'CD86', 'CD28', 'CD80', 'CD274', 'CD86', 'CD40', 'CD86', 
                'CD40', 'CD28', 'CD40', 'TNFRSF14'],
            'Gene2': ['TNFSF4', 'PDCD1', 'TNFSF4', 'CD28', 'TNFSF4', 'CD86', 'TNFSF9', 'VSIR', 'HAVCR2', 'PDCD1',
                      'CD200', 'CD80', 'CD276', 'CD274', 'CD86']}

## ssGSEA
TGFb = {'TGFb':['SLC20A1', 'XIAP', 'TGFBR1', 'BMPR2','FKBP1A', 'SKIL']}

def get_all_score(df, clin, cancer = "Other", response_col = "resp"):
    common_samples = df.columns.intersection(clin.index)
    df = df[common_samples]
    if not all(isinstance(idx, str) for idx in df.index):
        df = map_index_id(df, symbol2id1, id2symbol = True)
        df = df.loc[~df.index.duplicated(keep='first')]
    cyt1_score = get_geomean_score(df, CYT1, "CYT1")
    cyt2_score = get_geomean_score(df, CYT2, "CYT2")
    ifnr_score = get_geomean_score(df, IFNr, "IFNr")
    tls_score = get_geomean_score(df, TLS, "TLS")
    tis_score = get_geomean_score(df, TIS, "TIS")
    tip_hotscore = get_avgmean_score(df, TIP_hot, "TIP Hot")
    tip_coldscore = get_avgmean_score(df, TIP_cold, "TIP Cold")
    cs_polarity_score = get_ratio_score(df, CS_polarity, "CS Polarity")
    impres_score = get_impres_score(df, IMPRES, "IMPRES")
    score_df = pd.concat([cyt1_score, cyt2_score, ifnr_score, tls_score, tis_score, 
                          tip_hotscore, tip_coldscore, cs_polarity_score, impres_score], axis = 1)#, tgfb_score], axis = 1)
    score_df.columns = ['CYT1', 'CYT2','IFNr','TLS','TIS','TIP Hot','TIP Cold', 'CS Polarity','IMPRES']#,'TGFb']
    ## the other scores in tmeimmune
    tmeimmune_score = TME_score.get_score(df, method = ['ISTME', 'ESTIMATE', 'SIA', 'NetBio'], clin = clin, test_clinid = response_col)
    df_log2 = np.log2(df + 1)
    row_means = df_log2.mean(axis=1)
    df_centered = df_log2.sub(row_means, axis=0)
    tide_score = TIDE(df_centered, cancer, pretreat = True, vthres = 0.)['TIDE']
    score_df = pd.concat([score_df, tmeimmune_score, tide_score], axis = 1)
    score_df_scaled = score_df.copy()
    for cohort in clin['source'].unique():
        idx = clin[clin['source'] == cohort].index
        idx = idx.intersection(score_df.index)
        cohort_df = score_df.loc[idx].copy()
        # replace inf
        cohort_df = cohort_df.replace(
            [np.inf, -np.inf],
            np.nan)
        cohort_df = cohort_df.clip(-1e6, 1e6)
        # fill NaN
        cohort_df = cohort_df.fillna(cohort_df.median())
        scaler = MinMaxScaler(feature_range=(0, 1))
        score_df_scaled.loc[idx] = scaler.fit_transform(
            cohort_df
        )
    return score_df_scaled


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def optimal_cost_threshold(y_true, y_prob, c_fp=5, c_fn=1):
    thresholds = np.linspace(0.01, 0.99, 99)
    best_t, best_cost = 0.5, np.inf

    for t in thresholds:
        y_hat = (y_prob >= t).astype(int)
        if len(np.unique(y_hat)) < 2:
            continue
        tn, fp, fn, tp = confusion_matrix(y_true, y_hat).ravel()
        cost = c_fp * fp + c_fn * fn

        if cost < best_cost:
            best_cost, best_t = cost, t

    return best_t


def zscore_norm(df_train, df_test = None, method = 'zscore'):
    if method == 'zscore':
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler(feature_range=(-1,1))
    X_train_scaled = scaler.fit_transform(df_train)
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=df_train.columns, index=df_train.index)

    if df_test is not None:
        X_test_scaled = scaler.transform(df_test)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=df_test.columns, index=df_test.index)

        return X_train_scaled, X_test_scaled

    return X_train_scaled


def best_threshold_macro_f1(y_true, y_score, thresholds=None):
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)

    f1s = []
    for t in thresholds:
        y_pred = (y_score >= t).astype(int)
        f1s.append(f1_score(y_true, y_pred, average="macro"))

    best_idx = np.argmax(f1s)
    return thresholds[best_idx], f1s[best_idx]



def bucketize_tensor(X):
    X_bucket = torch.zeros_like(X, dtype=torch.long)

    X_bucket[X == 0] = 0
    X_bucket[X == 1] = 1
    X_bucket[X >= 2] = 2

    return X_bucket


def quantile_transformation(gene_train, clin_train, gene_test = None, clin_test = None):

    qt_global = QuantileTransformer(
        output_distribution='uniform',
        n_quantiles=min(1000, len(gene_train)),
        random_state=123
    )

    qt_global.fit(gene_train)
    aligned_train = []
    aligned_test = []
    cohort_qt = {}

    for cohort in clin_train['source'].unique():
        idx = clin_train['source'] == cohort
        X_cohort = gene_train.loc[idx]
            # fit transformer on cohort
        qt_cohort = QuantileTransformer(
            output_distribution='uniform',
            n_quantiles=min(1000, len(X_cohort)),
            random_state=123
        )
        qt_cohort.fit(X_cohort)
        cohort_qt[cohort] = qt_cohort
        percentile = qt_cohort.transform(X_cohort)
        train_aligned = qt_global.inverse_transform(percentile)
        train_aligned = pd.DataFrame(
            train_aligned,
            index=X_cohort.index,
            columns=X_cohort.columns
        )
        aligned_train.append(train_aligned)
    gene_train_aligned = pd.concat(aligned_train).loc[
        gene_train.index
    ]

    if clin_test is not None:
        for cohort in clin_test['source'].unique():
            idx = clin_test['source'] == cohort
            X_cohort = gene_test.loc[idx]
            qt_cohort = cohort_qt[cohort]
            qt_cohort.transform(X_cohort)
            percentile = qt_cohort.transform(X_cohort)
            test_aligned = qt_global.inverse_transform(percentile)
            test_aligned = pd.DataFrame(
                test_aligned,
                index=X_cohort.index,
                columns=X_cohort.columns
            )
            aligned_test.append(test_aligned)
        gene_test_aligned = pd.concat(aligned_test).loc[
            gene_test.index
        ]
        return gene_train_aligned, gene_test_aligned
    return gene_train_aligned




def cohort_align_zscore(X_train, cohort_train, X_test=None, cohort_test=None, cohort_col = 'source'):

    train_aligned = X_train.copy()
    if X_test is not None:
        test_aligned = X_test.copy()
    feature_cols = X_train.columns
    global_mean = X_train.mean()
    global_std = X_train.std()
    global_std = global_std.replace(0, 1)
    cohort_stats = {}

    # ---- TRAIN ----
    for cohort in cohort_train[cohort_col].unique():
        idx = cohort_train[cohort_col] == cohort
        cohort_data = X_train.loc[idx, feature_cols]
        cohort_mean = cohort_data.mean()
        cohort_std = cohort_data.std()
        cohort_std = cohort_std.replace(0, 1)
        cohort_stats[cohort] = (
            cohort_mean,
            cohort_std
        )
        aligned = (
            (cohort_data - cohort_mean)
            / cohort_std
        ) * global_std + global_mean

        train_aligned.loc[idx, feature_cols] = aligned.values

    # ---- TEST ----
    if X_test is not None:
        for cohort in cohort_test[cohort_col].unique():
            idx = cohort_test[cohort_col] == cohort
            cohort_data = X_test.loc[idx, feature_cols]
            cohort_mean, cohort_std = cohort_stats[cohort]

            aligned = (
                (cohort_data - cohort_mean)
                / cohort_std
            ) * global_std + global_mean

            test_aligned.loc[idx, feature_cols] = aligned.values
        return train_aligned, test_aligned
    return train_aligned


def cohort_align_minmax(
    X_train,
    cohort_train,
    X_test=None,
    cohort_test=None,
    cohort_col='source'
):
    train_aligned = X_train.copy()
    if X_test is not None:
        test_aligned = X_test.copy()
    feature_cols = X_train.columns
    global_min = X_train.min()
    global_max = X_train.max()
    global_range = (global_max - global_min).replace(0, 1)

    cohort_stats = {}
    # ---- TRAIN ----
    for cohort in cohort_train[cohort_col].unique():
        idx = cohort_train[cohort_col] == cohort
        cohort_data = X_train.loc[idx, feature_cols]
        cohort_min = cohort_data.min()
        cohort_max = cohort_data.max()
        cohort_range = (cohort_max - cohort_min).replace(0, 1)
        cohort_stats[cohort] = (
            cohort_min,
            cohort_max
        )
        aligned = (
            (cohort_data - cohort_min)
            / cohort_range
        ) * global_range + global_min
        train_aligned.loc[idx, feature_cols] = aligned.values

    # ---- TEST ----
    if X_test is not None:
        for cohort in cohort_test[cohort_col].unique():
            idx = cohort_test[cohort_col] == cohort
            cohort_data = X_test.loc[idx, feature_cols]
            cohort_min, cohort_max = cohort_stats[cohort]
            cohort_range = (cohort_max - cohort_min).replace(0, 1)
            aligned = (
                (cohort_data - cohort_min)
                / cohort_range
            ) * global_range + global_min
            test_aligned.loc[idx, feature_cols] = aligned.values
        return train_aligned, test_aligned
    return train_aligned


def quantile_norm(X_train, X_test = None):
    qt = QuantileTransformer(
    output_distribution='uniform',
    random_state=123
    )

    X_train_qt = qt.fit_transform(X_train)
    X_train_qt = pd.DataFrame(X_train_qt, columns=X_train.columns, index=X_train.index)
    if X_test is not None:
        X_test_qt = qt.transform(X_test)
        X_test_qt = pd.DataFrame(X_test_qt, columns=X_test.columns, index=X_test.index)
        return X_train_qt, X_test_qt
    return X_train_qt


def combat_correction(gene_df, batch, covar = None):
    gene_df1 = gene_df.fillna(1e-08)
    gene_df1 = gene_df1.replace(0, 1e-08)
    gene_df2 = pycombat_norm(gene_df1, np.array(batch), covar)
    return gene_df2