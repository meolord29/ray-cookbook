# Purpose: 
# 
# To learn how to use basic functions of remote ray and how to use the gpu with ray

# ==========[local package imports]===========

import ray


# ==[Set up packages on the remote ray cluster]==

runtime_env = {
    "pip": {
        "packages": ["torch", "torchvision", "torchaudio", "emoji"]
    }
}

# Init the remote cluster
ray.init(address="ray://192.168.3.179:10001", runtime_env=runtime_env)


print(ray.cluster_resources())


# ===========[ Tasks ]===================

# Validate package installation on the remote ray cluster
@ray.remote
def f():
  import emoji
  return emoji.emojize('Python is :thumbs_up:')

print(ray.get(f.remote()))

# validate usage of the gpu
@ray.remote(num_gpus=1) #num_gpus=1
def g():
    import torch
    import time

    # Set matrix dimensions
    matrix_size = 10000

    # Generate random matrices
    A = torch.randn(matrix_size, matrix_size)
    B = torch.randn(matrix_size, matrix_size)

    # Perform matrix multiplication on CPU
    start_time_cpu = time.time()
    result_cpu = torch.matmul(A, B)
    end_time_cpu = time.time()

    cpu_time = end_time_cpu - start_time_cpu
    print(f"CPU Time: {cpu_time:.6f} seconds")

    # Check if a GPU is available
    if torch.cuda.is_available():
        # Move matrices to GPU
        A_gpu = A.to("cuda")
        B_gpu = B.to("cuda")

        # Warm-up GPU (optional, ensures accurate timing by avoiding startup costs)
        _ = torch.matmul(A_gpu, B_gpu)

        # Perform matrix multiplication on GPU
        torch.cuda.synchronize()  # Ensure previous GPU tasks are done
        start_time_gpu = time.time()
        result_gpu = torch.matmul(A_gpu, B_gpu)
        torch.cuda.synchronize()  # Ensure computation is finished
        end_time_gpu = time.time()

        gpu_time = end_time_gpu - start_time_gpu
        print(f"GPU Time: {gpu_time:.6f} seconds")

        # Check for result consistency
        if torch.allclose(result_cpu, result_gpu.cpu(), atol=1e-6):
            print("The results from the CPU and GPU match!")
        else:
            print("The results from the CPU and GPU do not match!")
    else:
        print("GPU is not available. Cannot perform GPU computation.")

print(ray.get(g.remote()))


# useful information:
# https://askubuntu.com/questions/927199/nvidia-smi-has-failed-because-it-couldnt-communicate-with-the-nvidia-driver-ma
# https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=24.04&target_type=deb_local
# restart the machine when installing the cuda related stuff. then re-run the scripts.
