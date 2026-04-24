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

## PyTorch Only (Docker)

```bash
docker build -f Dockerfile.torch -t nn-x2-torch-test .
docker run --rm -v "${PWD}/out:/app/out" nn-x2-torch-test
```

## One Command (Docker Compose)

NumPy only:

```bash
docker compose up --build
```

NumPy + PyTorch together:

```bash
docker compose --profile torch up --build
```

You should see training loss decrease and plots written under `out/`:

- `out/fit_numpy.png`
- `out/fit_torch.png`
