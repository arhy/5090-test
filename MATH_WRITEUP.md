# Mathematical Write-Up: Fitting a Noisy x^2 with a 1-Hidden-Layer Neural Net

## 1. Problem Setup

The training data in train.py is generated on [-1, 1] with additive Gaussian noise:

$$
x_i \sim \text{linspace}(-1,1,n), \qquad
\epsilon_i \sim \mathcal{N}(0,\sigma^2), \qquad
y_i = x_i^2 + \epsilon_i
$$

where:

- n = NUM_OBS
- sigma = NOISE_STD

So the network learns a noisy version of the target function f(x)=x^2.

## 2. Model Architecture

The model is a fully connected network with one hidden tanh layer:

$$
z_1 = XW_1 + b_1, \quad
a_1 = \tanh(z_1), \quad
\hat{Y} = a_1W_2 + b_2
$$

Shapes in the implementation:

- X in R^{n x 1}
- W1 in R^{1 x h}, b1 in R^{1 x h}, h=16
- W2 in R^{h x 1}, b2 in R^{1 x 1}
- Y-hat, Y in R^{n x 1}

## 3. Objective

Mean squared error:

$$
\mathcal{L} = \frac{1}{n}\sum_{i=1}^{n}(\hat{y}_i - y_i)^2
= \frac{1}{n}\|\hat{Y}-Y\|_2^2
$$

## 4. Backpropagation Equations

Starting derivative:

$$
\frac{\partial \mathcal{L}}{\partial \hat{Y}} = \frac{2}{n}(\hat{Y}-Y)
$$

Output layer:

$$
\frac{\partial \mathcal{L}}{\partial W_2} = a_1^T\frac{\partial \mathcal{L}}{\partial \hat{Y}}, \qquad
\frac{\partial \mathcal{L}}{\partial b_2} = \sum_{i=1}^{n} \frac{\partial \mathcal{L}}{\partial \hat{Y}_i}
$$

Hidden layer chain rule:

$$
\frac{\partial \mathcal{L}}{\partial a_1}
= \frac{\partial \mathcal{L}}{\partial \hat{Y}}W_2^T
$$

Using tanh derivative:

$$
\frac{\partial \mathcal{L}}{\partial z_1}
= \frac{\partial \mathcal{L}}{\partial a_1} \odot (1-\tanh^2(z_1))
$$

Input layer gradients:

$$
\frac{\partial \mathcal{L}}{\partial W_1} = X^T\frac{\partial \mathcal{L}}{\partial z_1}, \qquad
\frac{\partial \mathcal{L}}{\partial b_1} = \sum_{i=1}^{n}\frac{\partial \mathcal{L}}{\partial z_{1,i}}
$$

## 5. Update Rule

For learning rate eta:

$$
W_1 \leftarrow W_1 - \eta\frac{\partial \mathcal{L}}{\partial W_1}, \quad
b_1 \leftarrow b_1 - \eta\frac{\partial \mathcal{L}}{\partial b_1}
$$

$$
W_2 \leftarrow W_2 - \eta\frac{\partial \mathcal{L}}{\partial W_2}, \quad
b_2 \leftarrow b_2 - \eta\frac{\partial \mathcal{L}}{\partial b_2}
$$

The script repeats this for 5000 epochs.

## 6. Why This Works

- A 1-hidden-layer tanh net is expressive enough to approximate smooth 1D functions.
- x^2 is smooth and low complexity.
- Noise in targets makes the model fit a smoothed trend rather than exact interpolation.
- MSE is natural for Gaussian-like noise.

## 7. Funky Mermaid Diagrams

### A. Training Pipeline

```mermaid
flowchart TD
    A[Set hyperparams n, sigma, h, eta, epochs] --> B[Generate x in -1 to 1]
    B --> C[Sample epsilon ~ N(0,sigma^2)]
    C --> D[Build targets y = x^2 + epsilon]
    D --> E[Initialize W1 b1 W2 b2]
    E --> F[Forward pass z1 tanh y_hat]
    F --> G[Compute MSE]
    G --> H[Backprop gradients]
    H --> I[Gradient step]
    I --> J{More epochs?}
    J -- Yes --> F
    J -- No --> K[Test predictions]
    K --> L[Save fit_numpy.png]

    classDef data fill:#d9f7e8,stroke:#1b5e20,stroke-width:2px;
    classDef model fill:#e8f0fe,stroke:#0d47a1,stroke-width:2px;
    classDef train fill:#fff3cd,stroke:#7a4f00,stroke-width:2px;
    classDef out fill:#fde2e4,stroke:#7f1d1d,stroke-width:2px;

    class B,C,D data;
    class E,F model;
    class G,H,I,J train;
    class K,L out;
```

### B. Computation Graph and Gradient Flow

```mermaid
graph LR
    X[X] --> Z1[Z1 = XW1 + b1]
    W1[W1] --> Z1
    b1[b1] --> Z1
    Z1 --> A1[A1 = tanh(Z1)]
    A1 --> YH[Y_hat = A1W2 + b2]
    W2[W2] --> YH
    b2[b2] --> YH
    Y[Y] --> L[Loss = mean((Y_hat - Y)^2)]
    YH --> L

    L -. dL/dY_hat .-> YH
    YH -. dL/dW2 dL/db2 .-> W2
    YH -. dL/dW2 dL/db2 .-> b2
    YH -. dL/dA1 .-> A1
    A1 -. dL/dZ1 via tanh' .-> Z1
    Z1 -. dL/dW1 dL/db1 .-> W1
    Z1 -. dL/dW1 dL/db1 .-> b1
```

### C. Training Lifecycle

```mermaid
stateDiagram-v2
    [*] --> Init
    Init --> DataReady: sample noisy x^2
    DataReady --> Forward
    Forward --> Loss
    Loss --> Backward
    Backward --> Update
    Update --> Forward: next epoch
    Update --> Evaluate: final epoch
    Evaluate --> SavePlot
    SavePlot --> [*]
```
