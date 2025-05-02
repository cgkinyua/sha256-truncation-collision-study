import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import time
import os

# Set the kernel file name for the 32-bit truncated hash
current_dir = os.path.dirname(os.path.abspath(__file__))
kernel_file = os.path.join(current_dir, 'sha256_kernel_32.cubin')
sha256_module = cp.RawModule(path=kernel_file)
sha256_kernel = sha256_module.get_function('sha256_hash')

def generate_keys_gpu(num_keys, key_length):
    # Generate a random binary matrix (0s and 1s) on the GPU
    return cp.random.randint(0, 2, size=(num_keys, key_length), dtype=cp.int32)

def has_collision_gpu(keys, key_length):
    keys_flat = keys.astype(bool).reshape(-1)
    packed = cp.packbits(keys_flat)
    bytes_per_key = (key_length + 7) // 8
    keys_gpu = packed.view(cp.uint8).reshape(keys.shape[0], bytes_per_key)
    
    hashes_gpu = cp.empty(keys.shape[0], dtype=cp.uint64)
    
    block_size = 256
    grid_size = (keys.shape[0] + block_size - 1) // block_size
    
    # Launch the kernel
    sha256_kernel(
        (grid_size,), (block_size,),
        (keys_gpu.data.ptr, hashes_gpu.data.ptr, keys.shape[0], key_length)
    )
    
    collision_exists = cp.any(cp.diff(cp.sort(hashes_gpu)) == 0)
    return int(collision_exists)

def monte_carlo_collision_simulation(num_keys, key_length, num_iterations, batch_size):
    collision_count = 0
    for _ in range(num_iterations):
        total_batches = (num_keys + batch_size - 1) // batch_size
        batch_collision_exists = False
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, num_keys)
            batch_num_keys = end_idx - start_idx
            keys = generate_keys_gpu(batch_num_keys, key_length)
            if has_collision_gpu(keys, key_length):
                batch_collision_exists = True
                break
        collision_count += int(batch_collision_exists)
    estimated_probability = collision_count / num_iterations
    return estimated_probability

if __name__ == "__main__":
    key_lengths = [64, 128, 256]  # key lengths in bits
    num_keys = 10**7             # Total keys for simulation
    num_iterations = 5           # Number of iterations per simulation
    batch_size = 10**5           # Process keys in batches
    
    print("Starting simulation with CUDA-based SHA-256 hashing (32-bit truncated)...\n")
    for key_length in key_lengths:
        start_time = time.time()
        estimated_probability = monte_carlo_collision_simulation(num_keys, key_length, num_iterations, batch_size)
        end_time = time.time()
        print(f"Key Length: {key_length}, Estimated Collision Probability: {estimated_probability:.8f}, Execution Time: {end_time - start_time:.2f} seconds")

