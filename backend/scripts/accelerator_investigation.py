import os
import glob
import subprocess
import onnxruntime as ort

def check_lib(name, patterns):
    print(f"\n--- Checking {name} libraries ---")
    found = False
    for pattern in patterns:
        paths = glob.glob(pattern)
        if paths:
            found = True
            for p in paths:
                print(f"Found: {p}")
    if not found:
        print(f"NOT FOUND: {name}")

def check_device(name, pattern):
    print(f"\n--- Checking {name} device nodes ---")
    paths = glob.glob(pattern)
    if paths:
        for p in paths:
            print(f"Found: {p}")
    else:
        print(f"NOT FOUND: {name}")

def main():
    print("==================================================")
    print("TASK 020: UNO Q ACCELERATOR INVESTIGATION")
    print("==================================================")
    
    # 1. ORT Providers
    print("\n--- ONNX Runtime Execution Providers ---")
    providers = ort.get_available_providers()
    for p in providers:
        print(f"- {p}")
        
    # 2. OpenCL
    check_lib("OpenCL", ["/usr/lib/aarch64-linux-gnu/libOpenCL.so*", "/usr/lib/libOpenCL.so*", "/vendor/lib64/libOpenCL.so*"])
    check_device("GPU/DRI", "/dev/dri/*")
    check_device("KGSL (Adreno)", "/dev/kgsl*")
    
    # 3. Vulkan
    check_lib("Vulkan", ["/usr/lib/aarch64-linux-gnu/libvulkan.so*", "/usr/lib/libvulkan.so*", "/vendor/lib64/vulkan*.so"])
    
    # 4. NNAPI
    check_lib("NNAPI", ["/usr/lib/aarch64-linux-gnu/libneuralnetworks.so*", "/usr/lib/libneuralnetworks.so*", "/system/lib64/libneuralnetworks.so*"])
    
    # 5. QNN / SNPE
    check_lib("QNN", ["/usr/lib/aarch64-linux-gnu/libQnn*.so*", "/usr/lib/libQnn*.so*", "/vendor/lib64/libQnn*.so*", "/usr/lib/libqnn*.so*"])
    check_lib("SNPE", ["/usr/lib/aarch64-linux-gnu/libSNPE*.so*", "/usr/lib/libSNPE*.so*", "/vendor/lib64/libSNPE*.so*"])
    
    # 6. DSP / FastRPC
    check_device("FastRPC", "/dev/fastrpc*")
    check_lib("FastRPC / ADSP", ["/usr/lib/aarch64-linux-gnu/libcdsprpc*.so*", "/usr/lib/libcdsprpc*.so*", "/usr/lib/libadsprpc*.so*", "/vendor/lib64/libadsprpc*.so*"])

    print("\n==================================================")
    print("OS Info:")
    try:
        with open("/etc/os-release", "r") as f:
            print(f.read().strip())
    except:
        pass

if __name__ == "__main__":
    main()
