# nn-docker-test

A tiny Dockerized Python project that trains small neural networks to fit the 1D function:

- y = x^2

It includes:

- NumPy implementation (`train.py`)
- PyTorch implementation (`train_torch.py`)
- Saved plots in `out/`

## NumPy Only (Docker)

```bash
docker build -t nn-x2-test .
```

```bash
docker run --rm -v "${PWD}/out:/app/out" nn-x2-test
```

On PowerShell, use:

```powershell
docker run --rm -v "${PWD}\out:/app/out" nn-x2-test
```

With custom noise and number of observations:

```powershell
docker run --rm -e NOISE_STD=0.05 -e NUM_OBS=800 -v "${PWD}\out:/app/out" nn-x2-test
```

## PyTorch Only (Docker)

```bash
docker build -f Dockerfile.torch -t nn-x2-torch-test .
docker run --rm -v "${PWD}/out:/app/out" nn-x2-torch-test
```

## One Command (Docker Compose)

Store outputs in a named Docker volume:

Set the volume name before running compose.

PowerShell example:

```powershell
$env:RESULTS_VOLUME="my-existing-volume"
```

Results are written to `/app/out` inside the container, which maps to that volume.

Tune data settings with env vars (defaults: `NOISE_STD=0.03`, `NUM_OBS=400`):

```powershell
$env:NOISE_STD="0.05"
$env:NUM_OBS="800"
```

NumPy only:

```bash
docker compose up --build
```

NumPy + PyTorch together:

```bash
docker compose --profile torch up --build
```

Check files in the volume:

```powershell
docker run --rm -v my-existing-volume:/data alpine ls -lah /data
```

You should see training loss decrease and plots written under `out/`:

- `out/fit_numpy.png`
- `out/fit_torch.png`
