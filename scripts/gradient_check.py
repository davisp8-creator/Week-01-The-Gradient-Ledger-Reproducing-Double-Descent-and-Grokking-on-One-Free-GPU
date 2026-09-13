"""Two-layer MLP with a hand-derived backward pass, verified two ways.

Forward pass:  Linear(in, hidden) -> ReLU -> Linear(hidden, out) -> softmax
Loss:          mean cross-entropy over the batch

The backward pass below is computed manually with plain tensor ops
(no autograd involved). It is checked against:
  1. torch.autograd, on the same forward computation.
  2. Central finite differences of the loss itself (ground truth).

Step-size choice for the finite-difference check
--------------------------------------------------
Central differences estimate a derivative as

    g(theta) ~= [f(theta+h) - f(theta-h)] / (2h)

with two competing error sources:
  * truncation error, from dropping higher-order Taylor terms: O(h^2)
    -- shrinks as h shrinks.
  * floating-point cancellation error, from subtracting two nearly
    equal numbers f(theta+h) and f(theta-h): O(eps / h)
    -- grows as h shrinks.

Total error is minimized near h* ~ eps^(1/3). For float64
(eps ~ 2.22e-16), h* ~ 6e-6, so this script uses h = 1e-5. For
float32 (eps ~ 1.19e-7), h* balloons to ~5e-3 -- too coarse to
resolve gradients cleanly, and rounding noise swamps the subtraction
well before h gets that small. So the finite-difference pass casts
its own copy of the parameters and inputs to float64, even though the
model itself trains in float32.

Usage:
    python scripts/gradient_check.py
"""

import torch
import torch.nn.functional as F

torch.manual_seed(0)


class ManualMLP:
    """Two-layer MLP: Linear -> ReLU -> Linear -> softmax cross-entropy."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        dtype=torch.float32,
        device: torch.device | str = "cpu",
    ):
        # Random init always drawn on CPU with a CPU generator, then moved to
        # `device` -- CUDA generators aren't interchangeable with CPU ones,
        # and this keeps the same seed reproducible regardless of device.
        g = torch.Generator().manual_seed(0)
        scale1 = (2.0 / in_dim) ** 0.5
        scale2 = (2.0 / hidden_dim) ** 0.5
        self.W1 = (torch.randn(in_dim, hidden_dim, generator=g, dtype=dtype) * scale1).to(device)
        self.b1 = torch.zeros(hidden_dim, dtype=dtype, device=device)
        self.W2 = (torch.randn(hidden_dim, out_dim, generator=g, dtype=dtype) * scale2).to(device)
        self.b2 = torch.zeros(out_dim, dtype=dtype, device=device)
        self._cache = {}

    def params(self) -> dict[str, torch.Tensor]:
        return {"W1": self.W1, "b1": self.b1, "W2": self.W2, "b2": self.b2}

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        z1 = X @ self.W1 + self.b1
        a1 = torch.clamp(z1, min=0.0)  # ReLU
        z2 = a1 @ self.W2 + self.b2
        self._cache = {"X": X, "z1": z1, "a1": a1, "z2": z2}
        return z2

    def loss(self, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        logits = self.forward(X)
        return F.cross_entropy(logits, y)

    def backward(self, y: torch.Tensor) -> dict[str, torch.Tensor]:
        """Manual backward pass for softmax + mean cross-entropy loss.

        Must be called after forward() (or loss()) on the same batch,
        since it reads the cached activations from that call.
        """
        X, z1, a1, z2 = (self._cache[k] for k in ("X", "z1", "a1", "z2"))
        N = X.shape[0]

        probs = torch.softmax(z2, dim=1)
        one_hot = F.one_hot(y, num_classes=z2.shape[1]).to(z2.dtype)
        dz2 = (probs - one_hot) / N  # d(mean CE loss) / d(logits)

        dW2 = a1.t() @ dz2
        db2 = dz2.sum(dim=0)

        da1 = dz2 @ self.W2.t()
        dz1 = da1 * (z1 > 0).to(z1.dtype)  # ReLU'(z1)

        dW1 = X.t() @ dz1
        db1 = dz1.sum(dim=0)

        return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}


def autograd_grads(model: ManualMLP, X: torch.Tensor, y: torch.Tensor):
    """Recompute the same forward pass under autograd and return d(loss)/d(param)."""
    params = {k: v.clone().detach().requires_grad_(True) for k, v in model.params().items()}
    z1 = X @ params["W1"] + params["b1"]
    a1 = torch.relu(z1)
    z2 = a1 @ params["W2"] + params["b2"]
    loss = F.cross_entropy(z2, y)
    loss.backward()
    return {k: v.grad.clone() for k, v in params.items()}, loss.item()


@torch.no_grad()
def finite_diff_grads(model: ManualMLP, X: torch.Tensor, y: torch.Tensor, h: float = 1e-5):
    """Central finite-difference gradient estimate, evaluated in float64."""
    X64 = X.double()
    params64 = {k: v.clone().double() for k, v in model.params().items()}

    def loss_at(params: dict[str, torch.Tensor]) -> float:
        z1 = X64 @ params["W1"] + params["b1"]
        a1 = torch.clamp(z1, min=0.0)
        z2 = a1 @ params["W2"] + params["b2"]
        return F.cross_entropy(z2, y).item()

    numeric = {}
    for name, p in params64.items():
        grad = torch.zeros_like(p)
        flat, grad_flat = p.view(-1), grad.view(-1)
        for i in range(flat.numel()):
            original = flat[i].item()
            flat[i] = original + h
            loss_plus = loss_at(params64)
            flat[i] = original - h
            loss_minus = loss_at(params64)
            flat[i] = original  # restore before moving to the next element
            grad_flat[i] = (loss_plus - loss_minus) / (2 * h)
        numeric[name] = grad
    return numeric


def relative_error(a: torch.Tensor, b: torch.Tensor) -> float:
    a64, b64 = a.double(), b.double()
    return ((a64 - b64).norm() / (a64.norm() + b64.norm() + 1e-12)).item()


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
    print(f"device = {device} ({device_name})")

    in_dim, hidden_dim, out_dim, batch = 5, 6, 3, 8
    model = ManualMLP(in_dim, hidden_dim, out_dim, device=device)

    X = torch.randn(batch, in_dim, device=device)
    y = torch.randint(0, out_dim, (batch,), device=device)

    manual_loss = model.loss(X, y)
    manual_grads = model.backward(y)
    auto_grads, auto_loss = autograd_grads(model, X, y)
    numeric_grads = finite_diff_grads(model, X, y, h=1e-5)

    print(f"manual loss = {manual_loss.item():.6f}   autograd loss = {auto_loss:.6f}")
    print()
    header = f"{'param':6s} {'manual_vs_autograd':>20s} {'manual_vs_numeric':>20s} {'autograd_vs_numeric':>20s}"
    print(header)
    print("-" * len(header))
    tol = 1e-4
    all_ok = True
    for name in manual_grads:
        e_ma = relative_error(manual_grads[name], auto_grads[name])
        e_mn = relative_error(manual_grads[name], numeric_grads[name])
        e_an = relative_error(auto_grads[name], numeric_grads[name])
        all_ok &= max(e_ma, e_mn, e_an) < tol
        print(f"{name:6s} {e_ma:20.3e} {e_mn:20.3e} {e_an:20.3e}")

    print()
    print(f"tolerance = {tol:.0e}  ->  {'ALL PASS' if all_ok else 'CHECK FAILED'}")


if __name__ == "__main__":
    main()
