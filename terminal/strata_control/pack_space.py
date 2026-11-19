"""Additional disk allocation for a pack that may resume or be rewritten."""
import os
from pathlib import Path


def remaining_pack_gb(folder: Path, total_gb: float) -> float:
    """A partial file is reusable or truncated by iq_pack before its replacement.

    Count allocated bytes, not sparse-file logical length. On an unsupported or
    failed filesystem probe, keep the full estimate. Never follow a link here.
    """
    path = Path(folder) / "experts.bin.tmp"
    try:
        if path.is_symlink() or not path.is_file():
            return total_gb
        info = path.stat()
        if getattr(info, "st_file_attributes", 0) & 0x400:  # reparse point
            return total_gb
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes
            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            probe = kernel.GetCompressedFileSizeW
            probe.argtypes = [wintypes.LPCWSTR, ctypes.POINTER(wintypes.DWORD)]
            probe.restype = wintypes.DWORD
            high = wintypes.DWORD()
            ctypes.set_last_error(0)
            low = probe(str(path), ctypes.byref(high))
            if low == 0xFFFFFFFF and ctypes.get_last_error():
                return total_gb
            allocated = (high.value << 32) | low
        else:
            allocated = getattr(info, "st_blocks", 0) * 512
        return max(0.0, total_gb - min(info.st_size, allocated) / 1e9)
    except (OSError, ValueError):
        return total_gb
