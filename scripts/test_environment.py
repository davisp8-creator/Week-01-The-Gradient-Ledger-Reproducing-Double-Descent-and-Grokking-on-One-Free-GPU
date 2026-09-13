"""Verify (and repair) the local Python environment for this project.

Checks Python version and that PyTorch, NumPy, pandas, matplotlib, and
scikit-learn are installed and functional, then exercises each with a
small operation so a broken install (e.g. missing wheels, bad CUDA
build) fails loudly instead of silently at import time.

Any missing package is pip-installed automatically (into the current
interpreter, via `sys.executable -m pip`) before its check runs. Pass
--no-install to just report what's missing instead of installing it.

Python itself cannot be auto-installed here: this script needs a
working interpreter to run at all, so if the *running* interpreter is
too old, the best it can do is install a newer Python alongside it
(via winget/apt/brew) and tell you to re-run with that interpreter.

Usage:
    python scripts/test_environment.py [--no-install]
"""

import shutil
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

MIN_PYTHON = (3, 10)
AUTO_INSTALL = "--no-install" not in sys.argv[1:]

# import name -> pip package name, for packages where they differ
PIP_NAME = {
    "sklearn": "scikit-learn",
}

results: list[tuple[str, bool, str]] = []


def check(name: str, fn):
    try:
        detail = fn()
        results.append((name, True, detail or "ok"))
    except Exception as exc:  # noqa: BLE001 - report any failure, not just expected ones
        results.append((name, False, f"{type(exc).__name__}: {exc}"))
        traceback.print_exc()


def ensure_import(import_name: str):
    """Import a module, pip-installing it first if it's missing."""
    try:
        return __import__(import_name)
    except ImportError:
        if not AUTO_INSTALL:
            raise
        pip_name = PIP_NAME.get(import_name, import_name)
        print(f"'{import_name}' not found - installing '{pip_name}' via pip...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--quiet", pip_name],
            check=True,
        )
        return __import__(import_name)


def check_python_version() -> str:
    if sys.version_info < MIN_PYTHON:
        required = f"{MIN_PYTHON[0]}.{MIN_PYTHON[1]}"
        found = f"{sys.version_info.major}.{sys.version_info.minor}"
        if AUTO_INSTALL:
            installer = _install_newer_python()
            if installer:
                raise RuntimeError(
                    f"Python {required}+ required, found {found}. Installed a newer "
                    f"Python via {installer} - re-run this script using that "
                    f"interpreter (this process can't hot-swap itself)."
                )
        raise RuntimeError(
            f"Python {required}+ required, found {found}. Install a newer Python "
            f"from https://python.org/downloads and re-run this script with it."
        )
    return f"Python {sys.version.split()[0]}"


def _install_newer_python() -> str | None:
    """Best-effort install of a newer Python via the platform package manager."""
    target = f"{MIN_PYTHON[0]}.{MIN_PYTHON[1]}"
    if shutil.which("winget"):
        cmd = ["winget", "install", "--id", f"Python.Python.{target}", "-e", "--silent"]
    elif shutil.which("brew"):
        cmd = ["brew", "install", f"python@{target}"]
    elif shutil.which("apt-get"):
        cmd = ["sudo", "apt-get", "install", "-y", f"python{target}"]
    else:
        return None
    print(f"Attempting to install Python {target} via: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
        return cmd[0]
    except (subprocess.CalledProcessError, OSError) as exc:
        print(f"Automatic install failed ({exc}); install Python manually instead.")
        return None


def check_numpy() -> str:
    np = ensure_import("numpy")

    arr = np.arange(9, dtype=np.float64).reshape(3, 3)
    assert arr.sum() == 36.0
    inv = np.linalg.inv(arr + np.eye(3))
    assert inv.shape == (3, 3)
    return f"numpy {np.__version__}"


def check_pandas() -> str:
    pd = ensure_import("pandas")

    df = pd.DataFrame({"width": [4, 8, 16], "test_error": [0.5, 0.3, 0.4]})
    grouped = df.groupby(df["width"] > 4)["test_error"].mean()
    assert len(grouped) == 2
    return f"pandas {pd.__version__}"


def check_matplotlib() -> str:
    matplotlib = ensure_import("matplotlib")

    matplotlib.use("Agg")  # headless backend, no display required
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [1, 4, 9])
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "smoke_test.png"
        fig.savefig(out)
        assert out.exists() and out.stat().st_size > 0
    plt.close(fig)
    return f"matplotlib {matplotlib.__version__}"


def check_sklearn() -> str:
    sklearn = ensure_import("sklearn")
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    X, y = make_classification(n_samples=200, n_features=10, random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    clf = LogisticRegression(max_iter=1000).fit(X_train, y_train)
    acc = clf.score(X_test, y_test)
    assert 0.0 <= acc <= 1.0
    return f"scikit-learn {sklearn.__version__} (smoke-test accuracy={acc:.2f})"


def check_pytorch() -> str:
    torch = ensure_import("torch")

    x = torch.randn(4, 4, requires_grad=True)
    y = (x ** 2).sum()
    y.backward()
    assert x.grad is not None and x.grad.shape == x.shape

    cuda_info = "CUDA available" if torch.cuda.is_available() else "CUDA not available (CPU only)"
    if torch.cuda.is_available():
        cuda_info += f" - {torch.cuda.get_device_name(0)}"
    return f"torch {torch.__version__} - {cuda_info}"


def main() -> int:
    check("Python version", check_python_version)
    check("NumPy", check_numpy)
    check("pandas", check_pandas)
    check("matplotlib", check_matplotlib)
    check("scikit-learn", check_sklearn)
    check("PyTorch", check_pytorch)

    print("\n" + "=" * 60)
    print("Environment check summary")
    print("=" * 60)
    all_ok = True
    for name, ok, detail in results:
        status = "PASS" if ok else "FAIL"
        all_ok &= ok
        print(f"[{status}] {name:15s} {detail}")
    print("=" * 60)

    if all_ok:
        print("All checks passed. Environment is ready.")
        return 0
    else:
        print("One or more checks failed. See tracebacks above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
