#!/bin/bash
# Standalone MN5/BSC launch script for Kimi-K2.7-Code with vLLM + Ray.
# No vector-inference repo checkout is required on the submitter side.
#
# Change --account if your allocation is not bsc70. The rest of the paths use
# $USER at runtime, so they work for another BSC user with access to the shared
# model and Singularity image paths below.
#SBATCH --job-name=Kimi-K2.7-Code-vec-inf
#SBATCH --partition=acc
#SBATCH --account=bsc70
#SBATCH --qos=acc_debug
#SBATCH --time=02:00:00
#SBATCH --nodes=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=80
#SBATCH --ntasks-per-node=1
#SBATCH --output=Kimi-K2.7-Code.%j.out
#SBATCH --error=Kimi-K2.7-Code.%j.err

module load singularity

unset LD_PRELOAD
unset SINGULARITYENV_LD_PRELOAD
unset APPTAINERENV_LD_PRELOAD
unset SINGULARITYENV_LD_LIBRARY_PATH
unset APPTAINERENV_LD_LIBRARY_PATH

MODEL_NAME="Kimi-K2.7-Code"
MODEL_PATH="/gpfs/scratch/bsc70/hpai/storage/projects/heka/models/Kimi-K2.7-Code"
IMAGE_PATH="/gpfs/scratch/bsc70/singularity/vllm_openai_0.20.1-cu129-deepgemm.sif"
WORK_DIR="/gpfs/scratch/bsc70/users/$USER/vec-inf-work"
CACHE_DIR="$WORK_DIR/.vec-inf-cache"
RUNTIME_DIR="$WORK_DIR/.vec-inf-runtime-$SLURM_JOB_ID"

mkdir -p "$WORK_DIR" "$CACHE_DIR" "$RUNTIME_DIR"
cd "$WORK_DIR" || exit 1

find_available_port() {
    local host="$1"
    local start_port="$2"
    local end_port="$3"
    local port

    for port in $(seq "$start_port" "$end_port"); do
        if ! timeout 1 bash -c "cat < /dev/null > /dev/tcp/$host/$port" >/dev/null 2>&1; then
            echo "$port"
            return 0
        fi
    done

    echo "No available port found in range $start_port-$end_port on $host" >&2
    return 1
}

vllm_config_path="$RUNTIME_DIR/kimi_multi_vllm_config.yaml"
cat > "$vllm_config_path" <<'VLLM_CONFIG'
async-scheduling: false
max-cudagraph-capture-size: 256
stream-interval: 100
max-num-batched-tokens: 2048
max-num-seqs: 8

compilation-config: >
  {"mode":"VLLM_COMPILE",
   "cudagraph_mode":"FULL_AND_PIECEWISE",
   "cudagraph_num_of_warmups":1,
   "pass_config":{"fuse_allreduce_rms":true}}
VLLM_CONFIG

export SINGULARITY_BINDPATH="${SINGULARITY_BINDPATH:+$SINGULARITY_BINDPATH,}/dev,/tmp,$WORK_DIR,$CACHE_DIR:$HOME/.cache,$CACHE_DIR:/cache,$MODEL_PATH,/home/bsc/$USER"

# Get list of nodes.
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

export VLLM_HOST_IP=$head_node_ip
export HOST_IP=$head_node_ip
export RAY_NODE_IP_ADDRESS=$head_node_ip
export RAY_OVERRIDE_NODE_IP_ADDRESS=$head_node_ip
export SINGULARITYENV_VLLM_HOST_IP=$head_node_ip
export SINGULARITYENV_HOST_IP=$head_node_ip
export SINGULARITYENV_RAY_NODE_IP_ADDRESS=$head_node_ip
export SINGULARITYENV_RAY_OVERRIDE_NODE_IP_ADDRESS=$head_node_ip
export APPTAINERENV_VLLM_HOST_IP=$head_node_ip
export APPTAINERENV_HOST_IP=$head_node_ip
export APPTAINERENV_RAY_NODE_IP_ADDRESS=$head_node_ip
export APPTAINERENV_RAY_OVERRIDE_NODE_IP_ADDRESS=$head_node_ip

# Pick RDMA/NCCL settings based on host capabilities.
CONTAINER_NCCL_ENV=""
if ! command -v ibv_devices >/dev/null 2>&1; then
    echo "ibv_devices not found; forcing TCP. (No RDMA userland on host?)"
    export NCCL_IB_DISABLE=1
    export SINGULARITYENV_NCCL_IB_DISABLE=1
    export APPTAINERENV_NCCL_IB_DISABLE=1
    CONTAINER_NCCL_ENV=",NCCL_IB_DISABLE=1"
else
    if ibv_devinfo 2>/dev/null | grep -q "link_layer:.*Ethernet"; then
        export NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX:-3}
    else
        export NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX:-0}
    fi
    export SINGULARITYENV_NCCL_IB_GID_INDEX=$NCCL_IB_GID_INDEX
    export APPTAINERENV_NCCL_IB_GID_INDEX=$NCCL_IB_GID_INDEX
    CONTAINER_NCCL_ENV=",NCCL_IB_GID_INDEX=$NCCL_IB_GID_INDEX"
fi

# Ray head address. Avoid Ray's default worker port range.
head_node_port=$(find_available_port "$head_node_ip" 20000 65535)
ray_head=$head_node_ip:$head_node_port
echo "Ray Head IP: $ray_head"

export RAY_ADDRESS=$ray_head
export SINGULARITYENV_RAY_ADDRESS=$ray_head
export APPTAINERENV_RAY_ADDRESS=$ray_head

server_port_number=$(find_available_port "$head_node_ip" 8080 65535)
server_address="http://${head_node_ip}:${server_port_number}/v1"
echo "vLLM server address: $server_address"

container_env="XDG_CACHE_HOME=/cache,TMPDIR=/tmp,TMP=/tmp,TEMP=/tmp,TORCHINDUCTOR_CACHE_DIR=/cache/torchinductor,TRITON_CACHE_DIR=/cache/triton,VLLM_HOST_IP=${VLLM_HOST_IP:-},HOST_IP=${HOST_IP:-},RAY_ADDRESS=${RAY_ADDRESS:-},RAY_NODE_IP_ADDRESS=${RAY_NODE_IP_ADDRESS:-},RAY_OVERRIDE_NODE_IP_ADDRESS=${RAY_OVERRIDE_NODE_IP_ADDRESS:-}${CONTAINER_NCCL_ENV}"

head_ready_file="$RUNTIME_DIR/ray-head-ready"
workers_started_file="$RUNTIME_DIR/ray-workers-started"
rm -f "$head_ready_file" "$workers_started_file"

head_launch_script="$RUNTIME_DIR/head_vllm.sh"
cat > "$head_launch_script" <<'VEC_INF_HEAD_SCRIPT'
#!/bin/bash
set -euo pipefail

head_node_ip="$1"
head_node_port="$2"
cpus_per_task="$3"
head_ready_file="$4"
workers_started_file="$5"
server_port_number="$6"
vllm_config_path="$7"

export VLLM_HOST_IP="$head_node_ip"
export HOST_IP="$head_node_ip"
export RAY_NODE_IP_ADDRESS="$head_node_ip"
export RAY_OVERRIDE_NODE_IP_ADDRESS="$head_node_ip"
export RAY_ADDRESS="$head_node_ip:$head_node_port"

echo "Starting Ray HEAD at $head_node_ip:$head_node_port"
ray start --head --node-ip-address="$head_node_ip" --port="$head_node_port" --num-cpus "$cpus_per_task" --num-gpus 4

touch "$head_ready_file"
echo "Waiting for Ray workers to be started..."
while [ ! -f "$workers_started_file" ]; do sleep 1; done

echo "Starting vLLM on Ray head"
vllm serve /gpfs/scratch/bsc70/hpai/storage/projects/heka/models/Kimi-K2.7-Code \
    --served-model-name Kimi-K2.7-Code \
    --host "$head_node_ip" \
    --port "$server_port_number" \
    --distributed-executor-backend ray \
    --trust-remote-code \
    --max-model-len 100000 \
    --gpu-memory-utilization 0.8 \
    --tensor-parallel-size 16 \
    --enable-auto-tool-choice \
    --tool-call-parser kimi_k2 \
    --reasoning-parser kimi_k2 \
    --config "$vllm_config_path" \
    ${VEC_INF_API_KEY:+--api-key} ${VEC_INF_API_KEY:+"$VEC_INF_API_KEY"}
VEC_INF_HEAD_SCRIPT
chmod +x "$head_launch_script"

echo "Starting HEAD and vLLM at $head_node"
srun --nodes=1 --ntasks=1 -w "$head_node" \
    singularity exec --nv --cleanenv --env "$container_env" --containall "$IMAGE_PATH" \
    bash "$head_launch_script" "$head_node_ip" "$head_node_port" "$SLURM_CPUS_PER_TASK" "$head_ready_file" "$workers_started_file" "$server_port_number" "$vllm_config_path" &
head_vllm_pid=$!

for _ in $(seq 1 120); do
    if [ -f "$head_ready_file" ]; then break; fi
    sleep 1
done

if [ ! -f "$head_ready_file" ]; then
    echo "[ERROR] Ray head did not become ready." >&2
    exit 1
fi

worker_num=$((SLURM_JOB_NUM_NODES - 1))
for ((i = 1; i <= worker_num; i++)); do
    node_i=${nodes_array[$i]}
    echo "Starting WORKER $i at $node_i"
    node_ip=$(srun --nodes=1 --ntasks=1 -w "$node_i" hostname --ip-address)
    echo "Node ip $node_ip"
    srun --nodes=1 --ntasks=1 -w "$node_i" \
        singularity exec --nv --cleanenv --env "$container_env" --containall "$IMAGE_PATH" \
        bash -lc '
            export VLLM_HOST_IP="$2"
            export HOST_IP="$2"
            export RAY_NODE_IP_ADDRESS="$2"
            export RAY_OVERRIDE_NODE_IP_ADDRESS="$2"
            ray start --address "$1" --node-ip-address="$2" --num-cpus "$3" --num-gpus 4 --block
        ' bash "$ray_head" "$node_ip" "$SLURM_CPUS_PER_TASK" &
    sleep 5
done

touch "$workers_started_file"
wait "$head_vllm_pid"
