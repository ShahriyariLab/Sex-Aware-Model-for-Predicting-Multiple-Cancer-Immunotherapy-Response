#!/bin/bash
#SBATCH -c 4  # Number of Cores per Task
##SBATCH --nodes=1
##SBATCH --gpus=2080ti:1
#SBATCH --mem=8192  # Requested Memory
#SBATCH -p cpu          
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=FAIL,END
#SBATCH -t 15:00:00  # Job time limit
#SBATCH -o slurm-%j.out  # %j = job ID

module load python/3.11.7
module load cuda/11.8

python -m venv ~/venvs/ici311

source ~/venvs/ici311/bin/activate
# python -m pip install "setuptools<70"
# python -m pip show setuptools
# python -m pip install pandas numpy scipy scikit-learn torch torch_geometric positional_encodings timm imblearn lifelines xgboost lightgbm catboost


TRT="PD1"
n_reps=1
cancer="allcancer"
interaction=True
combat=False
norm="minmax" #["zscore", "quantile", "cohort_zscore", "cohort_quantile", "cohort_minmax","raw", "minmax"]

for i in $(seq 1 "$n_reps"); do
    python -u codes/model_fusion.py --num_epochs 50 --lr 1e-04 --batch_size 16 --epsilon 1e-08 --temperature 0.5 --repeat "$i" --TRT "$TRT" --cancer "$cancer" \
    --hidden_dim 16 --feature_dim 8 --interaction "$interaction" --correction "$combat" --norm "$norm"

done

shopt -s nullglob

merged="output/fusion_result_${TRT}_int_${interaction}_${cancer}_${n_reps}.csv"
score="output/fusion_scores_${TRT}_${task}_${n_reps}.csv"
first="output/fusion_out_${TRT}_int_${interaction}_${cancer}_1.csv"
score_first="output/fusion_score_${TRT}_${task}_1.csv"

cp "$first" "$merged"
for f in output/fusion_out_${TRT}_int_${interaction}_${cancer}_*.csv; do
  [[ "$f" == "$first" ]] && continue
  tail -n +2 "$f" >> "$merged"
done

cp "$score_first" "$score"
for f in output/fusion_score_${TRT}_*.csv; do
  [[ "$f" == "$score_first" ]] && continue
  tail -n +2 "$f" >> "$score"
done

# done
rm -f output/fusion_out_${TRT}_int_${interaction}_${cancer}_*.csv
rm -f output/fusion_score_${TRT}_*.csv
