import numpy as np
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    rng = np.random.default_rng(42)

    # Training data: y = x^2 over [-1, 1]
    x = np.linspace(-1.0, 1.0, 400, dtype=np.float32).reshape(-1, 1)
    y = x**2

    # Tiny 1-hidden-layer neural network
    hidden_size = 16
    w1 = rng.normal(0.0, 0.3, size=(1, hidden_size)).astype(np.float32)
    b1 = np.zeros((1, hidden_size), dtype=np.float32)
    w2 = rng.normal(0.0, 0.3, size=(hidden_size, 1)).astype(np.float32)
    b2 = np.zeros((1, 1), dtype=np.float32)

    lr = 0.05
    epochs = 5000
    n = x.shape[0]

    for epoch in range(1, epochs + 1):
        # Forward
        z1 = x @ w1 + b1
        a1 = np.tanh(z1)
        y_hat = a1 @ w2 + b2

        # MSE loss
        loss = np.mean((y_hat - y) ** 2)

        # Backward
        d_y_hat = (2.0 / n) * (y_hat - y)
        d_w2 = a1.T @ d_y_hat
        d_b2 = np.sum(d_y_hat, axis=0, keepdims=True)

        d_a1 = d_y_hat @ w2.T
        d_z1 = d_a1 * (1.0 - np.tanh(z1) ** 2)
        d_w1 = x.T @ d_z1
        d_b1 = np.sum(d_z1, axis=0, keepdims=True)

        # Gradient step
        w1 -= lr * d_w1
        b1 -= lr * d_b1
        w2 -= lr * d_w2
        b2 -= lr * d_b2

        if epoch % 500 == 0 or epoch == 1:
            print(f"epoch={epoch:4d} loss={loss:.8f}")

    # Quick sanity-check predictions
    test_x = np.array([[-1.0], [-0.5], [0.0], [0.5], [1.0]], dtype=np.float32)
    test_y = test_x**2
    test_pred = np.tanh(test_x @ w1 + b1) @ w2 + b2

    print("\nSample predictions:")
    for x_i, y_i, p_i in zip(test_x.flatten(), test_y.flatten(), test_pred.flatten()):
        print(f"x={x_i:>4.1f} target={y_i:.4f} pred={p_i:.4f}")

    out_dir = Path("out")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "fit_numpy.png"

    plt.figure(figsize=(7, 4))
    plt.plot(x.flatten(), y.flatten(), label="target y=x^2", linewidth=2)
    plt.plot(x.flatten(), y_hat.flatten(), label="numpy model", linestyle="--")
    plt.scatter(test_x.flatten(), test_pred.flatten(), s=20, label="sample preds")
    plt.title("NumPy NN fit to y=x^2")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    print(f"\nSaved plot to {out_path.resolve()}")


if __name__ == "__main__":
    main()
