# src/monte_carlo_simulation.py

import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import time

def generate_keys_gpu(num_keys, key_length):
    return cp.random.randint(0, 2, size=(num_keys, key_length), dtype=np.int32)

def has_collision_gpu(keys):
    flattened_keys = cp.ascontiguousarray(keys)
    packed = cp.packbits(flattened_keys)
    bits_per_key = keys.shape[1]
    bytes_per_key = (bits_per_key + 7) // 8
    dtype = getattr(cp, f"uint{bytes_per_key * 8}")
    key_hashes = packed.reshape(-1, bytes_per_key).view(dtype)[:, 0]
    # Check if any duplicates exist
    collision_exists = cp.any(cp.diff(cp.sort(key_hashes)) == 0)
    return int(collision_exists)

def monte_carlo_collision_simulation(num_keys, key_length, num_iterations):
    collision_count = 0
    
    for _ in range(num_iterations):
        keys = generate_keys_gpu(num_keys, key_length)
        collision_exists = has_collision_gpu(keys)
        collision_count += collision_exists
    
    estimated_probability = collision_count / num_iterations
    return estimated_probability

if __name__ == "__main__":
    num_keys = 10**5    # 100,000 keys per iteration
    key_length = 32     # 32-bit keys (birthday threshold at ~65,000 keys)
    num_iterations = 100
    
    print(f"Starting Monte Carlo simulation with {num_keys} keys of length {key_length} bits...")
    start_time = time.time()
    estimated_probability = monte_carlo_collision_simulation(num_keys, key_length, num_iterations)
    end_time = time.time()
    
    print(f"Estimated collision probability: {estimated_probability:.8f}")
    print(f"Execution time: {end_time - start_time:.2f} seconds")
