## ISAFN: A Sex-Aware Model for Predicting Multiple Cancer Immunotherapy Response

This repository contains the source code required to train, validate, and apply the ISAFN models.

### Model Training

We provide three versions of ISAFN:

\dot ISAFN: the original model
\dot ISAFN-EMS: ISAFN ensembled with multiple machine learning models
\dot ISAFN-TCGA: ISAFN pretrained on TCGA data

#### ISAFN

To train the original ISAFN model, run:

```bash
sbatch codes/model_fusion.sh
```

#### Ensemble ISAFN and Pretrained ISAFN

To train the ensemble or pretrained versions, run:
```bash
sbatch codes/model_pretrain_ems.sh
```

The model version is controlled by the pretrain parameter in the script:

\dot pretrain="None": trains the ensemble ISAFN
\dot pretrain="TCGA": trains the TCGA-pretrained ISAFN

### Model Prediction

Trained ISAFNs can be loaded as follows:

```bash
import pickle
with open("/model/fusion_models_final.pkl", "rb") as f:
    fusion_models = pickle.load(f)
isafn_a = fusion_models['gene+mut']['merged']
isafn_m = fusion_models['gene+mut']['male']
isafn_f = fusion_models['gene+mut']['female']
```

There are three modalities users can choose from, 'gene+mut' for fusion model, 'gene' for gene model, and 'mut' for mutation model. After loading the models, users can apply them to new samples for ICI response prediction.