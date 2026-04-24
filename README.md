# nn-docker-test

A tiny Dockerized Python project that trains a very small neural network to fit the 1D function:

- y = x^2

## Build

```bash
docker build -t nn-x2-test .
```

## Run

```bash
docker run --rm nn-x2-test
```

You should see training loss decrease and sample predictions near x^2 values.
