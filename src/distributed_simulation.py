# src/distributed_simulation.py

import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import time

def generate_keys_gpu(num_keys, key_length, device=0):
    with cp.cuda.Device(device):
        return cp.random.randint(0, 2, size=(num_keys, key_length), dtype=np.int32)

def has_collision_gpu(keys, device=0):
    with cp.cuda.Device(device):
        keys_bool = keys.astype(bool).reshape(-1)
        packed = cp.packbits(keys_bool)
        bits_per_key = keys.shape[1]
        bytes_per_key = (bits_per_key + 7) // 8
        dtype = getattr(cp, f"uint{bytes_per_key * 8}")
        key_hashes = packed.view(dtype).reshape(keys.shape[0], -1)[:, 0]
        collision_exists = cp.any(cp.diff(cp.sort(key_hashes)) == 0)
        return int(collision_exists)

def monte_carlo_collision_simulation(num_keys, key_length, num_iterations, batch_size=10**6, num_gpus=1):
    collision_count = 0
    
    for _ in range(num_iterations):
        total_batches = (num_keys + batch_size - 1) // batch_size
        batch_collision_exists = False
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, num_keys)
            batch_num_keys = end_idx - start_idx
            gpu_id = batch_idx % num_gpus  # Distribute batches across GPUs
            
            keys = generate_keys_gpu(batch_num_keys, key_length, device=gpu_id)
            batch_collision = has_collision_gpu(keys, device=gpu_id)
            if batch_collision:
                batch_collision_exists = True
                break  # Stop checking further batches if a collision is found
        
        collision_count += int(batch_collision_exists)
    
    estimated_probability = collision_count / num_iterations
    return estimated_probability

if __name__ == "__main__":
    num_keys_list = [10**8]  # Large number of keys
    key_length = 64  # 64-bit keys
    num_iterations = 5  # Reduce iterations due to high computational cost
    batch_size = 10**6  # Process keys in batches of 1 million
    num_gpus = 1  # Set to the number of available GPUs
    
    results = []
    print(f"Starting large-scale simulation with {key_length}-bit keys...")
    
    for num_keys in num_keys_list:
        start_time = time.time()
        estimated_probability = monte_carlo_collision_simulation(
            num_keys, key_length, num_iterations, batch_size=batch_size, num_gpus=num_gpus
        )
        end_time = time.time()
        results.append((num_keys, estimated_probability))
        print(f"Num Keys: {num_keys}, Estimated Probability: {estimated_probability:.8f}, Execution Time: {end_time - start_time:.2f} seconds")
