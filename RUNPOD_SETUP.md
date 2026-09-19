# RunPod setup for the LoRA-placement campaign

Written 2026-09-20. Sizes come from measuring the artifacts on disk, not from estimates.
Prices are RunPod's published figures at the time of writing and should be re-checked in
the console before committing.

---

## 0. Two decisions before you click anything

**Region.** The network volume and the pod must live in the same region, and a volume
cannot be moved. Filter the GPU list for L4 availability **first**, note the region, and
create the volume there. Getting this wrong means deleting the volume and re-uploading.

**Volume size: 50 GB.** Measured from what has to live there:

| What | Size | Note |
|---|---|---|
| Existing adapters, `final_adapter/` only | 1.50 GB | the 3.95 GB of `checkpoint-*/` optimiser states are not needed |
| `data_cache/` | 0.34 GB | includes the fixed evaluation index files, which must be preserved |
| `checkpoints/` (the `.pkl` run records) | 0.001 GB | tiny, and they are what makes the campaign resumable |
| `results/`, `logs/` | 0.03 GB | |
| **Upload total** | **~1.9 GB** | |
| Hugging Face weights, package B | 14.5 GB | Qwen 1.5, Falcon 1.0, two Granite 3B at 6.0 each |
| New training runs (72 anchor + 8 Granite) | ~13 GB | measured at 151 MB per anchor run |
| New per-instance outputs, full splits | ~1 GB | |
| Benchmark datasets at full split | ~1.5 GB | |
| **Working total** | **~32 GB** | |

50 GB leaves comfortable headroom. Go to 80 GB only if you add the scale steps
(package D). At $0.07/GB/month, 50 GB is about **$3.50/month**.

---

## 1. Create the network volume

Storage → **+ New Network Volume**.

- Region: the one you chose in step 0.
- Size: 50 GB.
- Name: something you will recognise in a month, e.g. `lora-placement`.

It mounts at `/workspace` on any pod you attach it to.

> **It bills continuously, whether or not a pod exists.** There is no auto-cleanup.
> Delete it from the Storage console when the campaign is finished, after downloading
> the results.

---

## 2. Deploy the pod

Pods → **+ Deploy**, filtered to your region.

| Setting | Value | Why |
|---|---|---|
| GPU | **NVIDIA L4 24 GB** | about $0.39/h on Community Cloud; 24 GB fits 3B with LoRA |
| Network Volume | the one from step 1 | **must be selected here.** A volume cannot be attached to an existing pod; you would have to delete the pod and redeploy |
| Template | a RunPod **PyTorch 2.x / CUDA 12.x** image | ships a CUDA-enabled torch, which `mamba-ssm` needs to build against |
| Container disk | 30 GB | ephemeral. Big enough for pip caches and compilation |
| Ports | leave the defaults (SSH 22, Jupyter 8888) | |

Deploy On-Demand. Community Cloud is cheaper; Secure Cloud is steadier if you plan long
unattended runs.

**A5000 instead of L4** is about a third cheaper and roughly a third slower, so the
total bill is similar. Prefer the L4: fewer hours means fewer chances for a run to be
interrupted.

---

## 3. SSH

Add your public key once, under Settings → SSH Public Keys, before deploying. Then from
the pod's Connect panel copy the SSH command, which looks like:

```bash
ssh root@<pod-id>-<hash>.proxy.runpod.net -i ~/.ssh/id_ed25519
```

On Windows, generate a key with `ssh-keygen -t ed25519` in Git Bash if you do not have
one, and paste `~/.ssh/id_ed25519.pub` into the console.

---

## 4. Upload

From your machine, ~1.9 GB. Upload only what cannot be regenerated:

```bash
cd "C:/Users/HectorBorobia/proyectos/papers/paper-tecnico-3/3-LoRA-Placement/3-LoRA-Placement"

# adapters, without the optimiser states that are only needed to resume a partial run
tar --exclude='checkpoint-*' -czf /tmp/adapters.tgz models
tar -czf /tmp/state.tgz data_cache checkpoints results logs

scp -i ~/.ssh/id_ed25519 /tmp/adapters.tgz /tmp/state.tgz root@<host>:/workspace/
ssh -i ~/.ssh/id_ed25519 root@<host> 'cd /workspace && tar xzf adapters.tgz && tar xzf state.tgz && rm *.tgz'
```

Then the code, which is small and versioned:

```bash
ssh root@<host>
cd /workspace && git clone https://github.com/hecboar/lora-placement-hybrid.git code
```

**Then put the fixed evaluation index files where the driver looks for them.** They were
written to `data_cache/` by the original pipeline, but the new evaluator reads them from
`results/eval_indices/`:

```bash
mkdir -p /workspace/results/eval_indices
cp /workspace/data_cache/indices__*.json /workspace/results/eval_indices/
ls /workspace/results/eval_indices | wc -l     # must print 7
```

This is not cosmetic. Those seven files are what make the new, larger evaluation subsets
supersets of the released ones. Without them the driver draws fresh subsets, and every
new result becomes unpairable with the 36 runs you already have.

The layout the scripts expect ends up as `/workspace/{models,data_cache,checkpoints,results,logs}`
plus `/workspace/code`.

---

## 5. Environment

```bash
cd /workspace/code

# keep every cache on the volume, not on the ephemeral container disk
cat >> ~/.bashrc <<'EOF'
export HF_HOME=/workspace/hf
export HF_DATASETS_CACHE=/workspace/hf/datasets
export TOKENIZERS_PARALLELISM=false
export HF_ALLOW_CODE_EVAL=1
export PYTHONPATH=/workspace/code
EOF
source ~/.bashrc && mkdir -p $HF_HOME

pip install -q -U transformers peft trl datasets accelerate bitsandbytes
```

> **The single most expensive mistake available here** is leaving `HF_HOME` at its
> default. Model weights would go to `~/.cache/huggingface` on the 30 GB container disk,
> which is wiped when the pod is terminated, so you would re-download 14.5 GB every
> session and risk filling the disk mid-run.

Then the fast Mamba-2 kernels, which halve Falcon's training time and are the largest
single saving in the whole plan:

```bash
python -c "import torch;print(torch.__version__, torch.version.cuda)"   # note both
pip install -q causal-conv1d --no-build-isolation
pip install -q mamba-ssm --no-build-isolation
python -c "import mamba_ssm, causal_conv1d; print('fast kernels OK')"
```

`--no-build-isolation` is required: without it pip builds in a clean environment, pulls
a CPU-only torch and the CUDA kernels fail to compile. pip tries a prebuilt wheel first;
if none matches your torch/CUDA pair it compiles, which can take 20–40 minutes. Let it
finish, it is paid for once. **If it fails, do not fight it** — the campaign runs without
it, just slower, and the budget guard accounts for that.

---

## 6. Verify before spending

```bash
cd /workspace/code
for m in evaluator_v2/evaluator.py pipeline_v2/*.py; do
  [ "$(basename $m)" = "paths.py" ] && continue
  echo "== $m"; python $m --self-test | tail -1
done
```

Nine suites, about 150 checks, no GPU needed. Everything must say `all checks passed`.

Then three commands that write nothing:

```bash
python pipeline_v2/evaluate_adapters.py --plan --project-dir /workspace
python pipeline_v2/install_conditions.py --check --discovery-dir /workspace/results/discovery
python pipeline_v2/run_campaign.py --dry-run --project-dir /workspace
```

---

## 7. First real GPU work

```bash
tmux new -s eval        # so a dropped SSH connection does not kill the job
python pipeline_v2/evaluate_adapters.py --run --project-dir /workspace 2>&1 | tee /workspace/logs/eval_v2.log
# detach with Ctrl-b then d; reattach with: tmux attach -t eval
```

About 6 GPU-hours, roughly €2.30. **Stop here and read the numbers.** Compare the GSM8K
accuracies against `reanalysis/result_matrix_corrected.csv`. They should be close for the
same conditions. A wild difference is a batching or prompt bug, not a finding.

Only then:

```bash
python pipeline_v2/install_conditions.py --install --discovery-dir /workspace/results/discovery
tmux new -s train
python pipeline_v2/run_campaign.py --run --project-dir /workspace --budget-hours 40
```

---

## 8. Before you shut down

```bash
cd /workspace && tar -czf results_$(date +%F).tgz checkpoints results logs \
  --exclude='models' && ls -lh results_*.tgz
# then scp it down
```

Everything on `/workspace` survives pod termination, but a local copy costs nothing and
the volume is the single point of failure.

- **Terminate** the pod when not working: GPU billing stops immediately, the volume and
  its contents remain.
- **Delete the volume** only when the campaign is done and downloaded. It bills whether
  or not a pod is attached.

---

## 9. Things that will bite you

| | |
|---|---|
| Volume attached after deploy | Impossible. Select it in the deploy form or start over. |
| Volume in a different region from the GPU | They will not connect. Choose the region first. |
| `HF_HOME` left at its default | 14.5 GB re-downloaded every session, onto a disk that is wiped. |
| Installing `mamba-ssm` without `--no-build-isolation` | Pulls CPU torch, kernels fail to build. |
| Running the campaign over a bare SSH session | A dropped connection kills a 40-hour job. Use `tmux`. |
| Running the notebook's own cells | Its first cell pip-installs transformers, peft and trl from git `main`, which can change library versions mid-campaign. The runner refuses to execute that cell; do not run it by hand either. |
| Forgetting the volume after the campaign | It bills silently, for ever. |
| 3B out of memory | Halve `per_device_train_batch_size` to 2 and double `gradient_accumulation_steps` to 8. The effective batch stays 16 and the protocol is unchanged. |
