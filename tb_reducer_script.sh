#!/bin/bash

# =========== for local Mac ===========
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate Drone-NetworkX-PyTorch
path=$(pwd)"/data"

# =========== for ARC linux ============
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=18
#SBATCH -t 00-01:00:00
#SBATCH -p normal_q
#SBATCH --account=zelin1
#SBATCH --export=NONE # this makes sure the compute environment is clean
# uncomment below for ARC linux
#module load Anaconda3
#source activate Drone-NetworkX-PyTorch
#path="/projects/zelin1/Drone_DRL_HT/data"
# ======================================

echo "load environment: DT-DRL-ENV"


env_discrete_version_set=(1 2 3 4 5 6 7)  #(1 2 3)
agent_name_set=('DT' 'random') #('DT' 'random' 'DQN' 'PPO' 'A2C')
env_name='CustomCartPole' # pick from ['CustomCartPole', 'CustomPendulum']

# Get number of CPUs on device. Manually set cpu_count if you want to use less cpu
cpu_count=10

# run tb-reducer for each agent. Each tb-reducer run in parallel matching cpu number
for env_discrete_version in "${env_discrete_version_set[@]}"
do
    for agent_name in "${agent_name_set[@]}"
    do
        # Run tb-reducer in background and limit the number of parallel jobs
        echo "Starting tb-reducer for agent ${agent_name} with env_discrete_version ${env_discrete_version}"
        (
          tb-reducer $path/$env_name/agent_${agent_name}/env_discrete_ver_${env_discrete_version}/* -o $path/tb_reduce/$env_name/agent_${agent_name}/env_discrete_ver_${env_discrete_version}/ -r mean --handle-dup-steps 'mean' --lax-step --lax-tags
        ) &

        # If the number of background processes matches CPU count, wait for all to complete
        if [[ $(jobs -r -p | wc -l) -ge $cpu_count ]]; then
            wait -n
        fi
    done
done

# Wait for any remaining background processes
wait

echo "tb_reducer_script.sh finished"

