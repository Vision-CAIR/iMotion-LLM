
read LOWERPORT UPPERPORT < /proc/sys/net/ipv4/ip_local_port_range
while :
do
        PORT="`shuf -i $LOWERPORT-$UPPERPORT -n 1`"
        ss -lpn | grep -q ":$PORT " || break
done

# export CUDA_VISIBLE_DEVICES=0,3

conda activate imotion

module load cuda/12.2

torchrun --nproc-per-node 4 --master_port ${PORT} <legacy_repo_root>/trajgpt/train.py --cfg-path <legacy_repo_root>/trajgpt/train_configs_new/eval05/p.yaml