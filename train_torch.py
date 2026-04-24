from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch


def main() -> None:
    torch.manual_seed(42)

    x = torch.linspace(-1.0, 1.0, 400).unsqueeze(1)
    y = x**2

    model = torch.nn.Sequential(
        torch.nn.Linear(1, 32),
        torch.nn.Tanh(),
        torch.nn.Linear(32, 1),
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    loss_fn = torch.nn.MSELoss()

    epochs = 3000
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        y_hat = model(x)
        loss = loss_fn(y_hat, y)
        loss.backward()
        optimizer.step()

        if epoch % 500 == 0 or epoch == 1:
            print(f"epoch={epoch:4d} loss={loss.item():.8f}")

    test_x = torch.tensor([[-1.0], [-0.5], [0.0], [0.5], [1.0]])
    test_y = test_x**2
    test_pred = model(test_x)

    print("\nSample predictions:")
    for x_i, y_i, p_i in zip(test_x.flatten(), test_y.flatten(), test_pred.flatten()):
        print(f"x={x_i.item():>4.1f} target={y_i.item():.4f} pred={p_i.item():.4f}")

    out_dir = Path("out")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "fit_torch.png"

    with torch.no_grad():
        full_pred = model(x).squeeze(1)

    plt.figure(figsize=(7, 4))
    plt.plot(x.squeeze(1).numpy(), y.squeeze(1).numpy(), label="target y=x^2", linewidth=2)
    plt.plot(x.squeeze(1).numpy(), full_pred.numpy(), label="torch model", linestyle="--")
    plt.scatter(test_x.squeeze(1).numpy(), test_pred.squeeze(1).detach().numpy(), s=20, label="sample preds")
    plt.title("PyTorch NN fit to y=x^2")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    print(f"\nSaved plot to {out_path.resolve()}")


if __name__ == "__main__":
    main()
