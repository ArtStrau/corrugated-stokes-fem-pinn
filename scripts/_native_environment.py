"""Windows-only native-library defaults used before numerical imports."""

from __future__ import annotations


WINDOWS_NATIVE_DEFAULTS = {
    "MKL_THREADING_LAYER": "SEQUENTIAL",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
}


def configure_native_library_environment(platform_name: str, environment) -> None:
    """Apply conservative defaults only for Windows, preserving user values."""
    if platform_name == "nt":
        for name, value in WINDOWS_NATIVE_DEFAULTS.items():
            environment.setdefault(name, value)
