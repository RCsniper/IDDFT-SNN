trial_lst=("1" "2" "3" "4" "5")

for trial in "${trial_lst[@]}"; do
  echo "trial=$trial"
  nohup python -u main.py \
    --trial_num "$trial" \
    --lr 0.1 \
    --T_step 10 \
    --n_epochs 400 \
    --training_strategy auSF \
    --bfunc gaussian \
    --amp 2.5 \
    --phase -1.0 \
    --device cuda:0 \
    > "trial_${trial}.log" 2>&1 &
  pid=$!
  wait "$pid"
done