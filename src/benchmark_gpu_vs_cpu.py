# src/benchmark_gpu_vs_cpu.py

import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import time

def generate_keys_cpu(num_keys, key_length):
    return np.random.randint(0, 2, size=(num_keys, key_length), dtype=np.int32)

def generate_keys_gpu(num_keys, key_length):
    return cp.random.randint(0, 2, size=(num_keys, key_length), dtype=np.int32)

def has_collision_cpu(keys):
    keys_bool = keys.astype(bool).reshape(-1)
    packed = np.packbits(keys_bool)
    bits_per_key = keys.shape[1]
    bytes_per_key = (bits_per_key + 7) // 8
    dtype = getattr(np, f"uint{bytes_per_key * 8}")
    key_hashes = packed.view(dtype).reshape(keys.shape[0], -1)[:, 0]
    collision_exists = np.any(np.diff(np.sort(key_hashes)) == 0)
    return int(collision_exists)

def has_collision_gpu(keys):
    keys_bool = keys.astype(bool).reshape(-1)
    packed = cp.packbits(keys_bool)
    bits_per_key = keys.shape[1]
    bytes_per_key = (bits_per_key + 7) // 8
    dtype = getattr(cp, f"uint{bytes_per_key * 8}")
    key_hashes = packed.view(dtype).reshape(keys.shape[0], -1)[:, 0]
    collision_exists = cp.any(cp.diff(cp.sort(key_hashes)) == 0)
    return int(collision_exists)

def monte_carlo_collision_simulation(generate_keys, has_collision, num_keys, key_length, num_iterations):
    collision_count = 0
    for _ in range(num_iterations):
        keys = generate_keys(num_keys, key_length)
        collision_exists = has_collision(keys)
        collision_count += collision_exists
    estimated_probability = collision_count / num_iterations
    return estimated_probability

if __name__ == "__main__":
    num_keys = 10**5
    key_length = 32
    num_iterations = 100
    
    # GPU Simulation
    print("Running GPU simulation...")
    start_time_gpu = time.time()
    gpu_probability = monte_carlo_collision_simulation(
        generate_keys_gpu, has_collision_gpu, num_keys, key_length, num_iterations
    )
    end_time_gpu = time.time()
    print(f"GPU Estimated Probability: {gpu_probability:.8f}, Execution Time: {end_time_gpu - start_time_gpu:.2f} seconds")
    
    # CPU Simulation
    print("Running CPU simulation...")
    start_time_cpu = time.time()
    cpu_probability = monte_carlo_collision_simulation(
        generate_keys_cpu, has_collision_cpu, num_keys, key_length, num_iterations
    )
    end_time_cpu = time.time()
    print(f"CPU Estimated Probability: {cpu_probability:.8f}, Execution Time: {end_time_cpu - start_time_cpu:.2f} seconds")
