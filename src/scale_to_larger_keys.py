# src/scale_to_larger_keys.py

import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import time

def generate_keys_gpu(num_keys, key_length):
    return cp.random.randint(0, 2, size=(num_keys, key_length), dtype=np.int32)

def has_collision_gpu(keys):
    keys_bool = keys.astype(bool).reshape(-1)
    packed = cp.packbits(keys_bool)
    bits_per_key = keys.shape[1]
    bytes_per_key = (bits_per_key + 7) // 8
    dtype = getattr(cp, f"uint{bytes_per_key * 8}")
    key_hashes = packed.view(dtype).reshape(keys.shape[0], -1)[:, 0]
    collision_exists = cp.any(cp.diff(cp.sort(key_hashes)) == 0)
    return int(collision_exists)

def monte_carlo_collision_simulation(num_keys, key_length, num_iterations, batch_size=10**6):
    collision_count = 0
    
    for _ in range(num_iterations):
        # Process keys in batches to reduce memory usage
        total_batches = (num_keys + batch_size - 1) // batch_size
        batch_collision_exists = False
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, num_keys)
            batch_num_keys = end_idx - start_idx
            keys = generate_keys_gpu(batch_num_keys, key_length)
            batch_collision = has_collision_gpu(keys)
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
    
    results = []
    print(f"Starting large-scale simulation with {key_length}-bit keys...")
    
    for num_keys in num_keys_list:
        start_time = time.time()
        estimated_probability = monte_carlo_collision_simulation(num_keys, key_length, num_iterations, batch_size=batch_size)
        end_time = time.time()
        results.append((num_keys, estimated_probability))
        print(f"Num Keys: {num_keys}, Estimated Probability: {estimated_probability:.8f}, Execution Time: {end_time - start_time:.2f} seconds")
    
    # Plot results
    num_keys, probabilities = zip(*results)
    plt.plot(num_keys, probabilities, marker='o')
    plt.xlabel("Number of Keys")
    plt.ylabel("Estimated Collision Probability")
    plt.title(f"Collision Probability vs. Number of Keys ({key_length}-bit keys)")
    plt.xscale('log')
    plt.grid(True)
    plt.savefig(f"../results/collision_probability_vs_num_keys_{key_length}_bits.png")
    plt.show()
