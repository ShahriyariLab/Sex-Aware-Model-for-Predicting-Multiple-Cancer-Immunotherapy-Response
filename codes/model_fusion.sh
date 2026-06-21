

TRT="PD1"
n_reps=1
cancer="allcancer"
interaction=True
combat=False
norm="minmax" #["zscore", "quantile", "cohort_zscore", "cohort_quantile", "cohort_minmax","raw", "minmax"]

for i in $(seq 1 "$n_reps"); do
    python -u codes/model_fusion.py --num_epochs 1 --lr 1e-04 --batch_size 16 --epsilon 1e-08 --temperature 0.5 --repeat "$i" --TRT "$TRT" --cancer "$cancer" \
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
