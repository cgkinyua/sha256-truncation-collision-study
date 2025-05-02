# src/validate_small_keys.py

import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import time

def generate_keys_gpu(num_keys, key_length):
    return cp.random.randint(0, 2, size=(num_keys, key_length), dtype=np.int32)

def has_collision_gpu(keys):
    # Flatten keys and convert to boolean
    keys_bool = keys.astype(bool).reshape(-1)
    packed = cp.packbits(keys_bool)  # Pack all bits together
    bits_per_key = keys.shape[1]
    bytes_per_key = (bits_per_key + 7) // 8
    dtype = getattr(cp, f"uint{bytes_per_key * 8}")
    
    # Reshape packed bits into hashes for each key
    key_hashes = packed.view(dtype).reshape(keys.shape[0], -1)[:, 0]
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
    num_keys_list = [100, 200, 250, 300, 400, 500]  # Vary number of keys
    key_length = 16  # 16-bit keys
    num_iterations = 100
    
    results = []
    print(f"Starting validation with {key_length}-bit keys...")
    
    for num_keys in num_keys_list:
        start_time = time.time()
        estimated_probability = monte_carlo_collision_simulation(num_keys, key_length, num_iterations)
        end_time = time.time()
        results.append((num_keys, estimated_probability))
        print(f"Num Keys: {num_keys}, Estimated Probability: {estimated_probability:.8f}, Execution Time: {end_time - start_time:.2f} seconds")
    
    # Plot results
    num_keys, probabilities = zip(*results)
    plt.plot(num_keys, probabilities, marker='o')
    plt.xlabel("Number of Keys")
    plt.ylabel("Estimated Collision Probability")
    plt.title(f"Collision Probability vs. Number of Keys ({key_length}-bit keys)")
    plt.grid(True)
    plt.savefig(f"../results/collision_probability_vs_num_keys_{key_length}_bits.png")
    plt.show()
