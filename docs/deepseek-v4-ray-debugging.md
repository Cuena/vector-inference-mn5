# DeepSeek V4 / vLLM 0.20.1 Ray Debugging Notes

Date: 2026-05-08

This document records the debugging work done around DeepSeek V4 Pro, Kimi
K2.6, Gemma 4, vLLM 0.20.1, Ray, and the MN5 Singularity SLURM launcher. It is
intended to explain why the current repo contains experimental changes and what
each change was trying to prove.

## Summary

The DeepSeek V4 Pro launch is not working yet.

The main unresolved failure is vLLM 0.20.1 failing to attach its EngineCore
process to the Ray cluster started by the SLURM script:

```text
ConnectionError: Failed to connect to Ray cluster at <head_ip>:<port>
RuntimeError: No node info found matching attributes: ...
```

The failure happens before model weights are loaded. It is not yet a DeepSeek
weight, DeepGEMM, or GPU-memory failure.

The current state of the repo includes several attempted fixes. Some were useful
and should probably stay, while others are diagnostic or experimental and should
be reviewed before keeping.

## External References Used

- vLLM distributed troubleshooting:
  https://docs.vllm.ai/en/v0.20.1/serving/distributed_troubleshooting/
- vLLM cluster launch example:
  https://docs.vllm.com.cn/en/latest/examples/online_serving/run_cluster/
- Ray `ray.init` API:
  https://docs.ray.io/en/latest/ray-core/api/doc/ray.init.html

These references led to the hypothesis that vLLM/Ray was selecting or resolving
the local node IP incorrectly in the containerized multi-node SLURM setup.

## Image / Version Comparison

Remote image inspection found:

```text
/gpfs/scratch/bsc70/singularity/vllm_openai_0.18.0.sif
  vLLM 0.18.0
  Ray 2.54.1
  Torch 2.10.0+cu129

/gpfs/scratch/bsc70/singularity/vllm_openai_0.19.1.sif
  vLLM 0.19.1
  Ray 2.55.0
  Torch 2.10.0+cu129
  external deep_gemm import/support present

/gpfs/scratch/bsc70/singularity/vllm_openai_0.20.1-cu129.sif
  vLLM 0.20.1
  Ray 2.55.1
  Torch 2.11.0+cu129
  DeepSeek V4 modules present
  vendored DeepGEMM support present

/gpfs/scratch/bsc70/singularity/vllm_openai_0.20.1-cu129-deepgemm.sif
  vLLM 0.20.1
  Ray 2.55.1
  Torch 2.11.0+cu129
  DeepSeek V4 modules present
  external and vendored DeepGEMM support present
```

The older Kimi run on vLLM 0.18.0 reached checkpoint loading. The vLLM 0.20.1
runs failed during Ray initialization before model loading.

## Gemma 4 Work

Initial Gemma launch failed because:

```text
vllm serve: error: argument --limit-mm-per-prompt:
Value image=0,audio=0 cannot be converted to <function loads ...>
```

Decision:

- vLLM expected JSON for `--limit-mm-per-prompt`.
- Changed the config from `image=0,audio=0` to `{"image":0,"audio":0}`.
- Added shell quoting for JSON-shaped engine arg values so JSON survives shell
  parsing.

Second Gemma launch failed because:

```text
ValueError: Chunked MM input disabled but max_tokens_per_mm_item (2496)
is larger than max_num_batched_tokens (2048). Please increase
max_num_batched_tokens.
```

Decision:

- Add a Gemma single-agent config rather than piling all tuning flags into
  `models.yaml`.
- The target use case was a single sequential agent, not high parallelism.
- User wanted thinking enabled and context length was acceptable.

Created:

```text
compilation_configs/gemma4_single_agent.yaml
```

Key settings included:

```yaml
async-scheduling: true
performance-mode: interactivity
stream-interval: 8
max-num-batched-tokens: 16384
max-num-seqs: 1
cudagraph-capture-sizes: [1]

speculative-config:
  model: /gpfs/scratch/bsc70/hpai/storage/projects/heka/models/gemma-4-31B-it-assistant
  num_speculative_tokens: 4
```

Reasoning:

- `max-num-batched-tokens: 16384` avoids the multimodal encoder budget failure.
- `max-num-seqs: 1` matches the sequential agent use case.
- The assistant model path is local because compute nodes do not have internet.

Status:

- This was config work for Gemma, not part of the DeepSeek Ray failure.
- It has local tests, but should still be validated by an actual Gemma launch.

## DeepSeek V4 Pro Config Work

Added a DeepSeek V4 Pro MN5 profile and an initial config file:

```text
compilation_configs/deepseek_v4_pro_mn5_tp16_ep.yaml
```

After job `40247684`, the active experiment was changed to:

```text
compilation_configs/deepseek_v4_pro_mn5_tp24_ep.yaml
```

with `num_nodes: 6` and `--tensor-parallel-size: 24`.

After job `40250971`, the active experiment was changed again to:

```text
compilation_configs/deepseek_v4_pro_mn5_tp8_dp3_ep.yaml
```

with `num_nodes: 6`, `--tensor-parallel-size: 8`, and
`--data-parallel-size: 3`.

After job `40253904`, the model preset also pins `--api-server-count: 1`.

After job `40254155`, the model preset also adds
`--data-parallel-backend: ray` and `VLLM_RAY_DP_PACK_STRATEGY=span`.

After job `40255510`, the launcher stopped exporting `VLLM_HOST_IP` from the
outer SLURM environment. After job `40256658`, it now sets node-local
`VLLM_HOST_IP` only for `ray start`.

After job `40262452`, the launcher now injects
`--data-parallel-address "$head_node_ip"` for Ray DP runs when the user did not
set a DP address explicitly.

After job `40263543`, the launcher now waits for Ray to report all allocated
nodes as alive before allowing the head-node script to start `vllm serve`.

Important config settings:

```yaml
block-size: 256
max-model-len: 100000
max-num-seqs: 1
max-num-batched-tokens: 512
no-enable-flashinfer-autotune: true
attention-config: '{"use_fp4_indexer_cache": true}'

compilation-config: >
  {"mode":0,
   "cudagraph_mode":"FULL_DECODE_ONLY",
   "pass_config":{"fuse_allreduce_rms":false}}
```

Speculative decoding was removed from the active DeepSeek experiment after
`40250971`; the next run should prove the base model boots before adding MTP
back.

Job `40253904` showed that leaving `api_server_count` unset makes vLLM default
it to `data_parallel_size` in internal-DP mode. It spawned three API server
processes and those child processes failed during CUDA/NVML capability checks:

```text
NVMLError_InvalidArgument: Invalid Argument
```

The next experiment pins one API server because this is a single-agent use case
and avoids vLLM's API server scale-out path.

Job `40254155` showed that this was not enough. vLLM still used the local
internal-DP core launcher (`DPLBAsyncMPClient`), so the DP core child processes
started from the head-node API process and failed during CUDA/NVML capability
checks:

```text
vllm.third_party.pynvml.NVMLError_InvalidArgument: Invalid Argument
```

According to vLLM's data-parallel deployment docs, DP mode can use Ray by
setting `--data-parallel-backend=ray`. The same docs say that when a single DP
group spans multiple nodes, `VLLM_RAY_DP_PACK_STRATEGY=span` should be set.
That matches this experiment because each DP rank is `TP8`, and each MN5 node
has only 4 GPUs.

Job `40255510` showed that Ray DP did start and vLLM created
`DPMoEEngineCoreActor` processes, but those actors failed because the launcher
exported `VLLM_HOST_IP` on the head node. vLLM copies `VLLM_*` environment
variables into Ray actors, so remote actors inherited the head IP and vLLM
reported duplicate node IPs:

```text
Every node should have a unique IP address ... 1 unique IP addresses
{'10.2.104.70'}.
```

The launcher now leaves `VLLM_HOST_IP` unset in the vLLM driver environment.
The script still passes the HTTP bind address with `--host "$head_node_ip"` and
still sets Ray node addresses via `ray start --node-ip-address`.

Job `40256658` showed that fully removing `VLLM_HOST_IP` was too aggressive.
The duplicate head-IP failure was gone, but vLLM workers could not infer their
own IP and fell back to `0.0.0.0`:

```text
Failed to get the IP address, using 0.0.0.0 by default.
Every node should have a unique IP address ... {'0.0.0.0'}.
```

The launcher now sets `VLLM_HOST_IP` only in the `ray start` process
environment for each node. The head script does not export it before
`vllm serve`, so the driver should not copy a single head IP into remote actors.

Job `40262452` showed that node-local `VLLM_HOST_IP` for Ray workers was not
enough because the vLLM driver itself still could not infer the ray-based DP
master address and selected `0.0.0.0`:

```text
Using host IP 0.0.0.0 as ray-based data parallel address
AssertionError: The DP master node (ip: 0.0.0.0) is missing or dead
```

The launcher now adds an explicit `--data-parallel-address "$head_node_ip"` for
Ray DP jobs. This keeps `VLLM_HOST_IP` out of the driver environment while still
giving vLLM a concrete DP master address.

Job `40263543` showed that the explicit DP address worked. vLLM moved to DP
placement group creation but saw only 5 live Ray nodes, not the allocated 6:

```text
AssertionError: Not enough total available nodes (5) and devices per node (4)
to satisfy required world size 8 and data parallel size 3
```

The stdout showed the last worker was still starting when vLLM began. The
launcher now polls `ray.nodes()` from inside the container and waits until the
alive Ray node count reaches `$SLURM_JOB_NUM_NODES` before starting the vLLM
driver.

Important correction:

- The original attention config attempt used a dotted form that vLLM parsed
  incorrectly.
- Job `40208068` failed with:

```text
argument --attention-config/-ac: 1 validation error for AttentionConfig
use_fp4_indexer_cache
  Input should be a valid boolean, unable to interpret input
  input_value='--compilation-config'
```

Decision:

- Use JSON for `attention-config`:

```yaml
attention-config: '{"use_fp4_indexer_cache": true}'
```

Status:

- That parse error was fixed.
- Later DeepSeek jobs got past argument parsing and model config resolution.
- They did not reach model weight loading because Ray initialization failed.

## Kimi / vLLM 0.20.1 Comparison

The user reported:

- Kimi works with the default image.
- Kimi fails with both vLLM 0.20.1 images.
- Kimi 2.6 was the immediate target at one point, then DeepSeek became the
  target again.

Compared last Kimi jobs:

```text
40226438
  Kimi-K2.6
  default / older image, vLLM 0.18.0
  status: CANCELLED by user
  result: Ray cluster connected, placement group created, checkpoint shards
          started loading

40226922
  Kimi-K2.6
  vLLM 0.20.1
  status: FAILED
  result: Ray head/workers started, then EngineCore failed during ray.init
```

Important finding:

```text
Tensor parallel size (16) exceeds available GPUs (4)
RuntimeError: No node info found matching attributes: ''
ConnectionError: Failed to connect to Ray cluster at <head_ip>:<port>
```

Interpretation:

- The warning about 16 TP vs 4 local GPUs is expected in a 4-node x 4-GPU Ray
  setup before Ray cluster resources are attached.
- The fatal problem is the Ray driver attachment, not necessarily TP=16.
- This same failure pattern later appeared with DeepSeek V4 Pro.

## DeepSeek Job Timeline

### Job 40208068

Failure:

```text
argument --attention-config/-ac: 1 validation error for AttentionConfig
use_fp4_indexer_cache
  Input should be a valid boolean, unable to interpret input
```

Decision:

- Fix `attention-config` by passing a JSON object string.

### Job 40228026

Failure:

```text
RuntimeError: No node info found matching attributes: ''
ConnectionError: Failed to connect to Ray cluster at 10.2.101.31:31642
```

Observed:

- Ray head and three workers started.
- vLLM 0.20.1 parsed DeepSeek arguments.
- DeepGEMM was detected:

```text
Detected quantization_config.scale_fmt=ue8m0; enabling UE8M0 for DeepGEMM.
```

Decision:

- Hypothesis: vLLM/Ray did not receive the explicit Ray head address.
- Add `RAY_ADDRESS=$ray_head` and pass it into Singularity/Apptainer env.

### Job 40228967

Failure:

```text
RuntimeError: No node info found matching attributes: ''
ConnectionError: Failed to connect to Ray cluster at 10.2.101.132:52144
```

Result:

- Explicit `RAY_ADDRESS` was present.
- Failure did not change.

Decision:

- Hypothesis: Ray worker nodes and/or the driver had inconsistent local node IPs.
- Add:

```bash
RAY_NODE_IP_ADDRESS=$head_node_ip
RAY_OVERRIDE_NODE_IP_ADDRESS=$head_node_ip
```

- Pass `--node-ip-address="$node_ip"` to Ray workers, not only to the head.

### Job 40229333

Failure:

```text
RuntimeError: No node info found matching attributes: ''
ConnectionError: Failed to connect to Ray cluster at 10.2.101.145:25738
```

Result:

- Worker node IPs were correct.
- Failure still had empty matching attributes.

Decision:

- Hypothesis: vLLM calls `ray.init(address=...)` without Ray's
  `_node_ip_address`, and Ray 2.55.1 cannot infer the local node from inside
  this SLURM/Singularity context.
- Add `vec_inf/sitecustomize.py` as an opt-in Python auto-import shim to patch
  `ray.init(..., _node_ip_address=VLLM_HOST_IP)`.

### Job 40229541

Failure:

```text
RuntimeError: No node info found matching attributes: ''
```

Result:

- `sitecustomize.py` did not load.
- The generated script had host-level `PYTHONPATH`, but final `vllm serve` ran
  with `singularity exec --cleanenv`, so the container did not receive it.

Decision:

- Add `SINGULARITYENV_PYTHONPATH` and `APPTAINERENV_PYTHONPATH`.

### Job 40229864

Failure:

```text
RuntimeError: No node info found matching attributes: ''
```

Result:

- The submitted `singularity exec --env ...` did include `PYTHONPATH` and patch
  env vars.
- `sitecustomize.py` still did not reliably patch the vLLM EngineCore path.

Decision:

- Stop relying only on Python auto-import behavior.
- Add explicit wrapper:

```text
vec_inf/vllm_with_ray_patch.py
```

- Change multi-node vLLM launch from:

```bash
vllm serve ...
```

to:

```bash
python3.12 -m vllm_with_ray_patch serve ...
```

### Job 40230235

Failure:

```text
/usr/local/bin/python3.12: No module named vllm_with_ray_patch
```

Result:

- The wrapper command was present.
- The wrapper module was not visible inside the container.
- `PYTHONPATH` pointed at `/gpfs/home/.../vec_inf`, but `--containall` did not
  bind that source directory into the container.

Decision:

- Add `src_dir` to the Singularity bind path.

### Job 40230342

Failure changed:

```text
RuntimeError: No node info found matching attributes: '10.2.104.1'
ConnectionError: Failed to connect to Ray cluster at 10.2.104.1:59739
```

Important result:

- The traceback now includes:

```text
/gpfs/home/bsc/bsc070916/repos/vector-inference-mn5/vec_inf/sitecustomize.py
```

- This proves the patch loaded and `ray.init` received `_node_ip_address`.
- The failure changed from empty attributes to the explicit head IP.

Interpretation:

- The original missing environment/import problem is fixed.
- Ray still cannot map the vLLM driver process to a local Ray node at that IP.
- Ray's logic appears to require discovering a local raylet process. In this
  launcher layout, Ray head/workers are started in background `srun` steps and
  the final vLLM driver is launched in a separate `singularity exec --containall`
  process.
- `--containall` likely isolates the process namespace enough that Ray cannot
  discover the raylet process even though TCP connectivity to the Ray head exists.

Status:

- This is the latest known state.
- No successful DeepSeek launch has been reached.

## Code Changes Made

### `vec_inf/client/_slurm_script_generator.py`

Changes include:

- Shell-quote non-boolean engine arg values with `shlex.quote`.
- Add runtime env forwarding for container commands:

```bash
PYTHONPATH=${PYTHONPATH:-}
VEC_INF_PATCH_RAY_INIT_NODE_IP=${VEC_INF_PATCH_RAY_INIT_NODE_IP:-}
VLLM_HOST_IP=${VLLM_HOST_IP:-}
HOST_IP=${HOST_IP:-}
RAY_ADDRESS=${RAY_ADDRESS:-}
RAY_NODE_IP_ADDRESS=${RAY_NODE_IP_ADDRESS:-}
RAY_OVERRIDE_NODE_IP_ADDRESS=${RAY_OVERRIDE_NODE_IP_ADDRESS:-}
```

- Launch multi-node vLLM through:

```bash
python3.12 -m vllm_with_ray_patch
```

instead of `vllm`.

Assessment:

- JSON shell quoting is likely worth keeping.
- Container env forwarding is related to the Ray patch and should be reviewed.
- The explicit vLLM wrapper is experimental and has not solved the Ray failure.

### `vec_inf/client/_slurm_templates.py`

Changes include:

- Add `src_dir` to the container bind path.
- Export `PYTHONPATH`, `SINGULARITYENV_PYTHONPATH`, and
  `APPTAINERENV_PYTHONPATH`.
- Export Ray address and node IP vars:

```bash
RAY_ADDRESS
RAY_NODE_IP_ADDRESS
RAY_OVERRIDE_NODE_IP_ADDRESS
VEC_INF_PATCH_RAY_INIT_NODE_IP
```

- Pass `--node-ip-address="$node_ip"` to Ray workers.

Assessment:

- Passing worker `--node-ip-address` matches vLLM's documented cluster example
  and is probably reasonable.
- Binding `src_dir` and exporting Python path were needed for the wrapper, but
  may be unnecessary if the wrapper is removed.
- The Ray env exports did not solve the failure alone.

### `vec_inf/sitecustomize.py`

Purpose:

- Opt-in auto-import patch for `ray.init`.
- When `VEC_INF_PATCH_RAY_INIT_NODE_IP=1`, inject
  `_node_ip_address=VLLM_HOST_IP` into `ray.init(address=...)`.

Result:

- Eventually proved to load in job `40230342`.
- Changed the Ray error from empty attributes to an explicit IP.
- Did not solve the Ray failure.

Assessment:

- Diagnostic value: high.
- Production value: questionable unless combined with a structural launcher fix.

### `vec_inf/vllm_with_ray_patch.py`

Purpose:

- Explicitly patch `ray.init` before calling vLLM's normal CLI.
- Avoid relying on Python's automatic `sitecustomize` import.

Result:

- Initial job `40230235` could not import it because `src_dir` was not bound.
- Later job `40230342` proved the patch path was active, but Ray still failed.

Assessment:

- Experimental and not proven useful as a final fix.

### `vec_inf/config/marenostrum5/models.yaml`

Changes include:

- Gemma 4 moved to vLLM 0.20.1 and the single-agent config.
- DeepSeek V4 Pro profile added.
- Kimi K2.6 profile added/updated during the Kimi comparison work.
- Some image paths were changed while the user was also editing the same config.

Assessment:

- This file needs a careful final review because both user edits and assistant
  edits happened concurrently.

### `compilation_configs/gemma4_single_agent.yaml`

Purpose:

- Gemma 4 single-agent inference config, including MTP speculative decoding.

Assessment:

- Reasonable but unvalidated by a successful launch in this debugging sequence.

### `compilation_configs/deepseek_v4_pro_mn5_tp8_dp3_ep.yaml`

Purpose:

- DeepSeek V4 Pro TP8 + DP3 + expert parallel config for 6 x 4-GPU nodes.

Assessment:

- Argument parsing is now past the original attention-config error.
- Job `40247684` proved the Ray attach issue was fixed and TP16 reached model
  construction.
- TP16 then failed with:

```text
ValueError: Weight input_size_per_partition = 192 is not divisible by
weight quantization block_k = 128.
```

- Job `40250971` tested TP24 on 6 nodes and failed earlier, during speculative
  decoding config validation:

```text
Value error, Total number of attention heads (128) must be divisible by
tensor parallel size (24).
```

- TP24 is therefore invalid for DeepSeek V4's 128 heads, at least with MTP and
  likely for attention sharding generally.
- TP8 satisfies both known constraints:
  - `128 % 8 == 0` for attention heads.
  - the known 3072-wide shared-expert partition becomes `3072 / 8 = 384`, and
    `384 % 128 == 0` for FP8 block quantization.
- DP3 uses the remaining 24-GPU allocation with expert parallelism
  (`EP_SIZE = TP_SIZE x DP_SIZE = 24`) instead of increasing TP to an invalid
  value.
- Speculative decoding is disabled in this config. It can be reintroduced only
  after the base model serves.
- Job `40253904` reached vLLM internal DP startup with TP8/DP3 but failed after
  vLLM defaulted `api_server_count` to 3. The active preset now sets
  `--api-server-count: 1` to avoid spawning extra API server processes.
- Job `40254155` still failed because internal DP core workers were local
  multiprocessing children. The active preset now sets
  `--data-parallel-backend: ray` plus `VLLM_RAY_DP_PACK_STRATEGY=span` so Ray
  allocates remote DP ranks and each TP8 replica may span two 4-GPU nodes.
- Job `40255510` proved Ray DP placement starts, but failed because
  `VLLM_HOST_IP` was copied into remote Ray actors. The launcher no longer
  exports `VLLM_HOST_IP`.
- Job `40256658` proved vLLM still needs a node-local `VLLM_HOST_IP` inside Ray
  workers. The launcher now injects it only into each node's `ray start`
  environment, not into the vLLM driver environment.
- Job `40262452` then failed because the driver chose `0.0.0.0` as the DP
  master address. The launcher now injects `--data-parallel-address
  "$head_node_ip"` for Ray DP jobs.
- Job `40263543` then failed because vLLM started before all Ray workers had
  joined. The launcher now waits for the full Ray node count.

### Tests

Tests were added/updated for:

- JSON argument shell quoting.
- MN5 Gemma profile.
- MN5 DeepSeek profile.
- MN5 Kimi K2.6 profile.
- Multi-node Ray env exports.
- Wrapper selection for multi-node vLLM.
- `sitecustomize.py` Ray patch behavior.

Focused tests passed at the end of the debugging sequence:

```text
45 passed
```

These are local generation/unit tests only. They do not prove the SLURM launch
works.

## Current Best Explanation

The vLLM 0.20.1 / Ray 2.55.1 stack tries to attach the vLLM EngineCore process
to an existing local Ray node. Ray's internal logic scans for local raylet
processes and then queries GCS for a matching node.

In our SLURM script:

1. Ray head is started in one background `srun` + `singularity exec --containall`.
2. Ray workers are started in separate background `srun` + `singularity exec --containall`.
3. The vLLM server is started later in another separate `singularity exec --containall`.

The vLLM driver can see the Ray head address, but Ray cannot discover a local
raylet process from the vLLM driver's container/process context. This explains
why:

- the old image could work or get further,
- the Ray cluster appears to start,
- vLLM parses all DeepSeek config,
- but `ray.init` fails before placement group/model loading.

The latest error with explicit attributes:

```text
No node info found matching attributes: '10.2.104.1'
```

means the driver was told the correct head IP, but Ray still could not discover
or match a local raylet for that driver process.

## Likely Next Directions

These are not implemented in a proven way.

### Option 1: Run vLLM Driver in the Same `srun` / Container Context as Ray Head

Most likely structural fix.

Instead of:

```bash
srun ... singularity exec ... ray start --head --block &
...
singularity exec ... vllm serve ...
```

the head node container could run a shell that starts Ray head in the background
and then runs vLLM in the same container/process namespace:

```bash
srun -w "$head_node" singularity exec ... bash -lc '
  ray start --head --node-ip-address="$head_node_ip" --port="$head_node_port"
  python3.12 -m vllm.entrypoints.openai.api_server ...
'
```

This should allow Ray's process scan to find the local raylet from the vLLM
driver.

Tradeoff:

- More invasive launcher template change.
- Need to coordinate readiness, JSON endpoint writing, and cleanup carefully.

### Option 2: Avoid `--containall`

If `--containall` is isolating process discovery, removing it for multi-node
Ray jobs may allow Ray to see the local raylet.

Tradeoff:

- Less isolation.
- May reintroduce host environment leakage that `--cleanenv` and `--containall`
  were trying to avoid.

### Option 3: Use vLLM's Native Multi-Node Arguments Instead of Manual Ray Start

DeepSeek's example command uses vLLM multi-node flags:

```bash
--tensor-parallel-size 16
--nnodes 2
--node-rank 0
--master-addr "$HEAD_IP"
```

The current launcher manually starts Ray and then calls a single vLLM server.
It may be worth testing a vLLM-native multi-node launch mode if vLLM 0.20.1
supports the required DeepSeek topology.

Tradeoff:

- Significant launcher path change.
- May require per-rank server invocation logic.

### Option 4: Revert Experimental Ray Patch Code

If the next person wants a clean baseline, consider reverting:

```text
vec_inf/sitecustomize.py
vec_inf/vllm_with_ray_patch.py
```

and the wrapper-specific launcher code.

Keep or separately review:

- JSON engine arg quoting.
- Gemma JSON `--limit-mm-per-prompt`.
- DeepSeek JSON `attention-config`.
- Worker `--node-ip-address`.

## Practical Cleanup Notes

Before merging or relying on this branch:

1. Review `vec_inf/config/marenostrum5/models.yaml` manually because user edits
   and assistant edits overlapped.
2. Decide whether to keep the wrapper approach or switch to a structural
   same-container-head-node launcher.
3. If reverting the wrapper, also remove the tests that assert
   `python3.12 -m vllm_with_ray_patch`.
4. If keeping the wrapper, add explicit logging that is visible in stdout or
   stderr before vLLM starts.
5. Do not treat the local passing tests as evidence of a successful remote
   launch.

## Final State at Time of Writing

Known latest failed job:

```text
40230342
```

Latest observed failure:

```text
RuntimeError: No node info found matching attributes: '10.2.104.1'
ConnectionError: Failed to connect to Ray cluster at 10.2.104.1:59739
```

This is progress only in the diagnostic sense: the Ray patch was loaded and
changed the error. The model still does not serve.

## Follow-Up Experiment: Remove `--containall`

After the notes above were written, the next experiment was prepared:

- Remove the unproven Ray shim/wrapper from the active launcher path.
- Keep low-risk Ray address and worker node-IP settings.
- Disable `--containall` only for multi-node vLLM jobs.
- Keep `--containall` for other containerized launches.

Rationale:

- Ray 2.55 still failed after receiving `_node_ip_address`.
- The remaining suspected issue is that separate `singularity exec --containall`
  invocations isolate process discovery.
- Ray's local-node attach path appears to need visibility of the local raylet
  process. Removing `--containall` is the least invasive way to test that before
  restructuring the launcher to run Ray head and vLLM in the same container
  process context.

Expected rendered multi-node vLLM shape for the next run:

```bash
singularity exec --nv --cleanenv ... <image> \
vllm serve ...
```

not:

```bash
singularity exec --nv --cleanenv --containall ... <image> \
vllm serve ...
```

Expected interpretation:

- If Ray gets past `ray.init`, `--containall` was likely the critical blocker.
- If the same Ray attach error persists, the next serious experiment should be
  the structural same-container head-node launcher.
- If new host-environment conflicts appear, `--containall` was masking those and
  the structural launcher may be safer than globally relaxing isolation.

## Result: Removing `--containall` Failed

Job:

```text
40245498
```

The submitted script did omit `--containall`, but it failed before reaching the
previous Ray attach point:

```text
RuntimeError: Unable to open /dev/urandom
```

Ray workers also aborted with:

```text
absl::lts_20230802::SeedGenException
Failed generating seed-material for URBG.
```

Conclusion:

- Plainly removing `--containall` is not viable on MN5 with this image.
- The launcher should keep `--containall`.
- The next experiment should be the structural fix: run `ray start --head` and
  `vllm serve` inside the same head-node `srun` + container process context.

## Follow-Up Experiment: Same-Container Ray Head and vLLM Driver

After job `40245498`, the active launcher was changed again:

- Restore `--containall`.
- Do not use the Ray shim/wrapper path.
- For multi-node containerized vLLM, make `_generate_launch_cmd()` create a
  head-node script in the job work directory.
- Run that script through:

```bash
srun --nodes=1 --ntasks=1 -w "$head_node" \
  singularity exec --nv --cleanenv ... --containall <image> \
  bash "$head_launch_script" ...
```

Inside that same container process, the script does:

```bash
ray start --head --node-ip-address="$head_node_ip" --port="$head_node_port" ...
vllm serve ...
```

Worker nodes are then started from the parent SLURM script with separate
`srun ... ray start --address "$ray_head" --block` commands.

Rationale:

- Keep the container behavior that gives access to `/dev/urandom`.
- Put the vLLM driver in the same container/process context as the Ray head
  raylet, which is what Ray 2.55 appears to require for local node discovery.

Expected interpretation for the next run:

- If it passes `ray.init`, this confirms that separate head/vLLM container
  contexts were the blocker.
- If it still fails at Ray attach, the next step is to inspect process
  visibility inside the head container or use vLLM's native multi-node launch
  mode instead of the manually started Ray cluster.

## Cleanup After Abandoning DeepSeek V4 Pro

After job `40263543`, the active DeepSeek V4 Pro work was abandoned because the
fix path had become too invasive for a model-specific experiment.

Repository cleanup decision:

- Keep the generic fixes needed for the newer vLLM/Ray image path:
  - shell quoting for JSON-like engine argument values;
  - `--containall` remains enabled;
  - multi-node containerized vLLM starts Ray head and `vllm serve` inside the
    same head-node container;
  - Ray workers are started with explicit `--node-ip-address`;
  - the Kimi K2.6 new-image preset remains available.
- Remove the active DeepSeek V4 Pro preset from `marenostrum5/models.yaml`.
- Remove the untracked DeepSeek TP8/DP3 experiment config from
  `compilation_configs/`.
- Remove the active test coverage that expected DeepSeek V4 Pro to be present
  in the MN5 model profile.
- Remove the launcher changes that were only for the DeepSeek TP8/DP3 Ray-DP
  experiment, including explicit Ray-DP address injection and the full Ray node
  count barrier.

The failed DeepSeek attempts are intentionally left documented above so the
same path does not need to be rediscovered later.
