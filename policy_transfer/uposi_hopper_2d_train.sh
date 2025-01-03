
scripts="python policy_transfer/ppo/hopper_2d/main.py"  

for seed in 10 20 30 40 50 60 70 80 90 100
do
    for ((i = 0; i < ${#scripts[@]}; i++))
    do  
        sbatch sbatch_4d.sh ${scripts[$i]} --seed=$seed --env Hopper-v3 --min_timesteps 10000 --xml_path ./hopper.xml --run_path ./runs_2d_hopper --order NOUPN
        echo ${scripts[$i]} --seed=$seed --env Hopper-v3 --min_timesteps 10000 --xml_path ./hopper.xml --run_path runs_2d_hopper --order NOUPN --env_seed variable --epochs 10 --configs=27 --hiddensize 64 
    done
done