#!/usr/bin/env bash
#SBATCH --time=06:59:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=3
#SBATCH --constraint=a100
#SBATCH --mem=48GB
#SBATCH --cpus-per-gpu=4
#SBATCH --partition=batch
#SBATCH --job-name=imotion_3
#SBATCH --output=/home/felembaa/logs/%j-%x.out
#SBATCH --error=/home/felembaa/errs/%j-%x.err
##SBATCH --mail-user=abdulwahab.felemban@kaust.edu.sa
#SBATCH --mail-type=ALL
#SBATCH --account=conf-icl-2025.09.24-elhosemh

# Get the config file path from the first argument
CFG_PATH="$1"
if [ -z "$CFG_PATH" ]; then
  echo "Error: No config file specified."
  echo "Usage: sbatch run_.sh /path/to/config.yaml"
  exit 1
fi

# Extract the job name from the config filename (without extension)
JOB_NAME=$(basename "$CFG_PATH" .yaml)

# Rename the current Slurm job
if [ -n "$SLURM_JOB_ID" ]; then
  scontrol update JobId=$SLURM_JOB_ID JobName=$JOB_NAME
  echo "Renamed job to: $JOB_NAME"
fi

source ~/miniconda3/bin/activate /ibex/project/c2253/felembaa/envs/imotion_mtr03

# Optional: dynamically pick a free port if needed
read LOWERPORT UPPERPORT < /proc/sys/net/ipv4/ip_local_port_range
while :
do
    PORT="`shuf -i $LOWERPORT-$UPPERPORT -n 1`"
    ss -lpn | grep -q ":$PORT " || break
done

PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')

torchrun --nproc-per-node 3 --master_port ${PORT} \
    /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train.py \
    --cfg-path "$CFG_PATH"

    # 

# #!/usr/bin/env bash
# #SBATCH --time=04:00:00
# #SBATCH --nodes=1
# #SBATCH --gpus-per-node=3
# #SBATCH --constraint=a100
# #SBATCH --mem=48GB
# #SBATCH --cpus-per-gpu=4
# #SBATCH --partition=batch
# #SBATCH --job-name=imotion
# #SBATCH --output=/home/felembaa/logs/%j-%x.out
# #SBATCH --error=/home/felembaa/errs/%j-%x.err
# #SBATCH --mail-user=abdulwahab.felemban@kaust.edu.sa
# #SBATCH --mail-type=ALL
# ##SBATCH --account conf-iccv-2025.03.08-elhosemh

# source ~/miniconda3/bin/activate /ibex/project/c2253/felembaa/envs/imotion_mtr03

# # Get the config file path from the first argument passed to this script
# CFG_PATH="$1"
# if [ -z "$CFG_PATH" ]; then
#   echo "Error: No config file specified."
#   echo "Usage: sbatch run.sh /path/to/config.yaml"
#   exit 1
# fi

# # Optional: dynamically pick a free port if needed
# read LOWERPORT UPPERPORT < /proc/sys/net/ipv4/ip_local_port_range
# while :
# do
#     PORT="`shuf -i $LOWERPORT-$UPPERPORT -n 1`"
#     ss -lpn | grep -q ":$PORT " || break
# done

# PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')

# torchrun --nproc-per-node 3 --master_port ${PORT} \
#     /home/felembaa/projects/iMotion-LLM-ICLR/trajgpt/train.py \
#     --cfg-path "$CFG_PATH"