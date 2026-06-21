

TRT="PD1"
n_reps=10
cancer="allcancer"
interaction=True
pretrain='None' #["None", "TCGA"]

for i in $(seq 1 "$n_reps"); do
    python -u codes/model_pretrain_ems.py --num_epochs 50 --lr 1e-04 --batch_size 16 --epsilon 1e-08 --temperature 0.5 --repeat "$i" --TRT "$TRT" --cancer "$cancer" \
    --hidden_dim 16 --feature_dim 8 --da "$da" --interaction "$interaction" --gene_expr --pretrain "$pretrain"

done


shopt -s nullglob

merged="output/pretrain_${pretrain}_result_${TRT}_int_${interaction}_${cancer}_${n_reps}.csv"
score="output/pretrain_${pretrain}_scores_${TRT}_${task}_${n_reps}.csv"
first="output/pretrain_${pretrain}_out_${TRT}_int_${interaction}_${cancer}_1.csv"
score_first="output/pretrain_${pretrain}_score_${TRT}_${task}_1.csv"

cp "$first" "$merged"
for f in output/pretrain_${pretrain}_out_${TRT}_int_${interaction}_${cancer}_*.csv; do
  [[ "$f" == "$first" ]] && continue
  tail -n +2 "$f" >> "$merged"
done

cp "$score_first" "$score"
for f in output/pretrain_${pretrain}_score_${TRT}_*.csv; do
  [[ "$f" == "$score_first" ]] && continue
  tail -n +2 "$f" >> "$score"
done

# done
rm -f output/pretrain_${pretrain}_out_${TRT}_int_${interaction}_${cancer}_*.csv
rm -f output/pretrain_${pretrain}_score_${TRT}_*.csv
