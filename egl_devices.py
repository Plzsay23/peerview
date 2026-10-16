"""Which EGL device index (MUJOCO_EGL_DEVICE_ID) is which physical GPU.

EGL enumerates devices in its own order, which need not match nvidia-smi's. On the lab server
MUJOCO_EGL_DEVICE_ID=3 rendered on nvidia-smi GPU 0. This prints, per EGL index, the DRM node, the PCI
address behind it and the CUDA index EGL reports, next to nvidia-smi's index -> PCI table.
Needs the same __EGL_VENDOR_LIBRARY_FILENAMES as the experiment. Creates no context, renders nothing.
"""
import ctypes
import os
import subprocess

EGL_DRM_DEVICE_FILE_EXT = 0x3233
EGL_CUDA_DEVICE_NV = 0x323A


def main():
    egl = ctypes.CDLL("libEGL.so.1")
    gpa = egl.eglGetProcAddress
    gpa.restype, gpa.argtypes = ctypes.c_void_p, [ctypes.c_char_p]

    def fn(name, restype, *argtypes):
        ptr = gpa(name)
        if not ptr:
            raise RuntimeError(f"{name.decode()} not available (EGL vendor not loaded?)")
        return ctypes.CFUNCTYPE(restype, *argtypes)(ptr)

    query_devices = fn(b"eglQueryDevicesEXT", ctypes.c_uint, ctypes.c_int, ctypes.POINTER(ctypes.c_void_p),
                       ctypes.POINTER(ctypes.c_int))
    query_string = fn(b"eglQueryDeviceStringEXT", ctypes.c_char_p, ctypes.c_void_p, ctypes.c_int)
    query_attrib = fn(b"eglQueryDeviceAttribEXT", ctypes.c_uint, ctypes.c_void_p, ctypes.c_int,
                      ctypes.POINTER(ctypes.c_long))

    devs, n = (ctypes.c_void_p * 32)(), ctypes.c_int()
    query_devices(32, devs, ctypes.byref(n))

    smi = {}
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,pci.bus_id", "--format=csv,noheader"],
                             capture_output=True, text=True).stdout
        for line in out.strip().splitlines():
            idx, bus = (x.strip() for x in line.split(","))
            smi[bus.lower()[-12:]] = idx  # nvidia-smi "00000000:41:00.0" vs sysfs "0000:41:00.0"
    except FileNotFoundError:
        pass

    print(f"EGL devices: {n.value}")
    print(f"{'MUJOCO_EGL_DEVICE_ID':>20}  {'drm':18s} {'pci':14s} {'egl_cuda':>8}  nvidia-smi")
    for i in range(n.value):
        drm = query_string(devs[i], EGL_DRM_DEVICE_FILE_EXT)
        drm = drm.decode() if drm else None
        pci = None
        if drm:
            pci = os.path.basename(os.path.realpath(f"/sys/class/drm/{os.path.basename(drm)}/device"))
        cuda = ctypes.c_long(-1)
        cuda_idx = cuda.value if query_attrib(devs[i], EGL_CUDA_DEVICE_NV, ctypes.byref(cuda)) else None
        smi_idx = smi.get(pci.lower()[-12:]) if pci else None
        print(f"{i:>20}  {str(drm):18s} {str(pci):14s} {str(cuda_idx):>8}  {smi_idx}")


if __name__ == "__main__":
    main()
