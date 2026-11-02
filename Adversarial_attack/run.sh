trial_lst=("1" "2" "3" "4" "5")

for trial in "${trial_lst[@]}"; do
  echo "trial=$trial"
  nohup python -u main_dvs.py \
    --trial_num "$trial" \
    --attack all \
    --steps 7 \
    --bfunc gaussian \
    --amp 2.5 \
    --phase -1.0 \
    --device cuda:0 \
    > "trial_${trial}.log" 2>&1 &
  pid=$!
  wait "$pid"
done