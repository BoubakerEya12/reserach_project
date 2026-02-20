# OMBOS GPU Setup (VS Code Remote-SSH)

This guide ensures you run long MTL training on `ombos.synchromedia.ca` (GPU server), not on Narval login nodes.

## Quick Start (single command)

If this is a fresh OMBOS session, you can bootstrap everything with:

```bash
bash scripts/bootstrap_ombos_mtl.sh
```

This script:
- clones the repo if missing
- copies `results/mtl_unsupervised.last.weights.h5` from Narval
- creates `venv_mtl` and installs dependencies
- runs GPU checks

## 1) Add OMBOS host to SSH config

In VS Code:
1. `Ctrl+Shift+P` -> `Remote-SSH: Open SSH Configuration File`
2. Add:

```sshconfig
Host ombos
    HostName ombos.synchromedia.ca
    User <your_login>
    # IdentityFile ~/.ssh/<your_key>   # optional, only if key-based auth
```

If you use password auth, omit `IdentityFile`.

## 2) Connect to OMBOS (not Narval)

In VS Code:
1. `Ctrl+Shift+P` -> `Remote-SSH: Connect to Host...`
2. Choose `ombos`
3. Open terminal and verify:

```bash
hostname
pwd
```

Expected:
- `hostname` should indicate OMBOS (and not `narval*`)
- `pwd` should be your OMBOS home/project path

## 3) Verify GPU visibility

From the same terminal:

```bash
bash scripts/check_ombos_gpu.sh
```

The script checks:
- host guard (warns if you're on Narval)
- `nvidia-smi`
- TensorFlow GPU device visibility

## 4) Run training safely with tmux

Use tmux so training survives SSH disconnects:

```bash
bash scripts/run_mtl_train_ombos.sh
```

This starts/attaches a `tmux` session named `mtl_train` and runs:
- checkpointed training
- resume from `results/mtl_unsupervised.last.weights.h5` if available

Detach from tmux:
- `Ctrl+b` then `d`

Reattach later:

```bash
tmux attach -t mtl_train
```

## 5) Evaluate latest checkpoint

Quick eval without plots:

```bash
bash scripts/run_mtl_eval_ombos.sh
```

or full eval:

```bash
python -m scripts.eval_mtl_unsupervised \
  --weights results/mtl_unsupervised.last.weights.h5 \
  --steps 50 --batch_size 64 --gamma_qos_db 5
```

## 6) Troubleshooting

- If you see `CUDA_ERROR_NO_DEVICE`, you are not on a GPU-capable node/server.
- If host looks like Narval, reconnect to `ombos`.
- If `tmux` is missing, install/use allowed environment tools per lab policy.
