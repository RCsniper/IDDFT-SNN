trial_lst=("1" "2" "3" "4" "5")

for trial in "${trial_lst[@]}"; do
  echo "trial=$trial"
  nohup python -u main.py \
    --trial_num "$trial" \
    --lr 0.1 \
    --T_step 4 \
    --n_epochs 400 \
    --training_strategy auSF \
    --bfunc gaussian \
    --amp 3.0 \
    --phase -1.5 \
    --device cuda:0 \
    > "trial_${trial}.log" 2>&1 &
  pid=$!
  wait "$pid"
done