#!/usr/bin/env python3
"""Enumerate the machine's OpenCL platforms by loading OpenCL.dll directly.

The point is only to answer one question: is the Radeon VII / MI50 reachable through OpenCL here?
If it is, the expert kernels can be transcribed almost verbatim from the CUDA originals instead of
being re-derived against a SPIR-V compiler that keeps misresolving dotPacked4x8EXT.
"""

import ctypes
import sys

CL_PLATFORM_NAME = 0x0902
CL_PLATFORM_VENDOR = 0x0903
CL_PLATFORM_VERSION = 0x0901
CL_PLATFORM_EXTENSIONS = 0x0904
CL_DEVICE_TYPE_ALL = 0xFFFFFFFF
CL_DEVICE_NAME = 0x102B
CL_DEVICE_VENDOR = 0x102C
CL_DEVICE_VERSION = 0x102F
CL_DEVICE_EXTENSIONS = 0x1030
CL_DEVICE_GLOBAL_MEM_SIZE = 0x101F
CL_DEVICE_AVAILABLE = 0x1027


def s(cl):
    cl.clGetPlatformIDs.argtypes = [ctypes.c_uint, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint)]
    cl.clGetPlatformIDs.restype = ctypes.c_int
    cl.clGetDeviceIDs.argtypes = [ctypes.c_void_p, ctypes.c_ulonglong, ctypes.c_uint,
                                  ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint)]
    cl.clGetDeviceIDs.restype = ctypes.c_int
    cl.clGetPlatformInfo.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_size_t,
                                     ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
    cl.clGetPlatformInfo.restype = ctypes.c_int
    cl.clGetDeviceInfo.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_size_t,
                                   ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
    cl.clGetDeviceInfo.restype = ctypes.c_int


def pstr(cl, obj, param, dev=False):
    buf = ctypes.create_string_buffer(4096)
    n = ctypes.c_size_t()
    fn = cl.clGetDeviceInfo if dev else cl.clGetPlatformInfo
    if fn(obj, param, 4096, buf, ctypes.byref(n)) != 0:
        return "<err>"
    return buf.value.decode("utf-8", "replace")


def main():
    try:
        cl = ctypes.WinDLL("OpenCL.dll")
    except OSError as exc:
        print(f"OpenCL.dll not loadable: {exc}")
        return 1
    s(cl)

    n_plat = ctypes.c_uint(0)
    rc = cl.clGetPlatformIDs(0, None, ctypes.byref(n_plat))
    print(f"clGetPlatformIDs -> rc={rc} platforms={n_plat.value}")
    if n_plat.value == 0:
        print("NO OPENCL PLATFORMS")
        return 0

    plats = (ctypes.c_void_p * n_plat.value)()
    cl.clGetPlatformIDs(n_plat.value, plats, None)

    for i, p in enumerate(plats):
        print(f"\nplatform[{i}] {pstr(cl, p, CL_PLATFORM_NAME)!r}")
        print(f"  vendor   : {pstr(cl, p, CL_PLATFORM_VENDOR)}")
        print(f"  version  : {pstr(cl, p, CL_PLATFORM_VERSION)}")
        ext = pstr(cl, p, CL_PLATFORM_EXTENSIONS)
        print(f"  plat ext : {ext[:200]}{'...' if len(ext) > 200 else ''}")

        nd = ctypes.c_uint(0)
        rc = cl.clGetDeviceIDs(p, CL_DEVICE_TYPE_ALL, 0, None, ctypes.byref(nd))
        print(f"  clGetDeviceIDs rc={rc} devices={nd.value}")
        if nd.value == 0:
            continue
        devs = (ctypes.c_void_p * nd.value)()
        cl.clGetDeviceIDs(p, CL_DEVICE_TYPE_ALL, nd.value, devs, None)
        for j, d in enumerate(devs):
            gmem = ctypes.c_ulonglong(0)
            cl.clGetDeviceInfo(d, CL_DEVICE_GLOBAL_MEM_SIZE, 8, ctypes.byref(gmem), None)
            avail = ctypes.c_uint(1)
            cl.clGetDeviceInfo(d, CL_DEVICE_AVAILABLE, 4, ctypes.byref(avail), None)
            dext = pstr(cl, d, CL_DEVICE_EXTENSIONS, dev=True)
            print(f"  device[{j}] {pstr(cl, d, CL_DEVICE_NAME, dev=True)!r}"
                  f"  gmem={gmem.value / 2**30:.1f} GiB  available={avail.value}")
            print(f"    vendor  : {pstr(cl, d, CL_DEVICE_VENDOR, dev=True)}")
            print(f"    version : {pstr(cl, d, CL_DEVICE_VERSION, dev=True)}")
            for key in ("cl_khr_subgroups", "cl_khr_fp16", "cl_khr_int64_base_atomics",
                        "cl_amd_media_ops", "cl_khr_il_program", "cl_khr_subgroup_ballot"):
                if key in dext:
                    print(f"    has {key}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
