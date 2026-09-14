"""Collect hardware/environment info for the experiment report.

Cross-platform best-effort detection (Windows or Linux, CPU or CUDA), so
a report generated on this repo's local CPU device and one generated on
a Colab GPU runtime are directly comparable and clearly labeled.

Usage:
    python scripts/hardware_info.py
    python scripts/hardware_info.py --out results/hardware_t4gpu.json
"""

import argparse
import json
import os
import platform
import re
import subprocess
from pathlib import Path


def _cpu_model() -> str:
    system = platform.system()
    if system == "Linux":
        try:
            text = Path("/proc/cpuinfo").read_text()
            m = re.search(r"model name\s*:\s*(.+)", text)
            if m:
                return m.group(1).strip()
        except OSError:
            pass
    elif system == "Windows":
        # wmic is deprecated/removed on newer Windows (e.g. Server 2025);
        # Get-CimInstance is the modern replacement and gives the real
        # marketing name instead of platform.processor()'s raw family/model
        # string.
        try:
            out = subprocess.check_output(
                ["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
                text=True,
                stderr=subprocess.DEVNULL,
            )
            name = out.strip()
            if name:
                return name
        except Exception:
            pass
    return platform.processor() or "unknown"


def _ram_gib() -> float | None:
    try:
        import psutil

        return round(psutil.virtual_memory().total / (1024**3), 1)
    except ImportError:
        pass

    system = platform.system()
    if system == "Linux":
        try:
            text = Path("/proc/meminfo").read_text()
            m = re.search(r"MemTotal:\s*(\d+)\s*kB", text)
            if m:
                return round(int(m.group(1)) / (1024**2), 1)
        except OSError:
            pass
    elif system == "Windows":
        try:
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))  # type: ignore[attr-defined]
            return round(stat.ullTotalPhys / (1024**3), 1)
        except Exception:
            pass
    return None


def system_is_windows_vm() -> bool:
    """Best-effort check for a virtualized Windows host (Hyper-V etc.)."""
    if platform.system() != "Windows":
        return False
    try:
        out = subprocess.check_output(
            ["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_ComputerSystem).Model"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return "virtual" in out.strip().lower()
    except Exception:
        return False


def collect() -> dict:
    import torch

    cuda = torch.cuda.is_available()
    info = {
        "os": f"{platform.system()} {platform.release()}",
        "os_version": platform.version(),
        "python_version": platform.python_version(),
        "cpu_model": _cpu_model(),
        "cpu_logical_cores": os.cpu_count(),
        "ram_gib": _ram_gib(),
        "ram_note": (
            "snapshot at run time -- Hyper-V Dynamic Memory can reallocate RAM to/from "
            "this VM between runs, so this figure is not a fixed spec"
            if system_is_windows_vm()
            else None
        ),
        "cuda_available": cuda,
        "gpu_name": torch.cuda.get_device_name(0) if cuda else None,
        "gpu_vram_gib": (
            round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 1) if cuda else None
        ),
        "torch_version": torch.__version__,
    }
    try:
        import matplotlib
        import numpy
        import pandas
        import sklearn

        info["numpy_version"] = numpy.__version__
        info["pandas_version"] = pandas.__version__
        info["matplotlib_version"] = matplotlib.__version__
        info["sklearn_version"] = sklearn.__version__
    except ImportError:
        pass
    return info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    info = collect()
    text = json.dumps(info, indent=2)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
