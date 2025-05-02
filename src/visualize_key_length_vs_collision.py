# src/visualize_key_length_vs_collision.py

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

def monte_carlo_collision_simulation(num_keys, key_length, num_iterations):
    collision_count = 0
    for _ in range(num_iterations):
        keys = generate_keys_gpu(num_keys, key_length)
        collision_exists = has_collision_gpu(keys)
        collision_count += collision_exists
    estimated_probability = collision_count / num_iterations
    return estimated_probability

if __name__ == "__main__":
    key_lengths = [16, 32, 64]  # Vary key lengths
    num_keys = 10**5  # Fixed number of keys
    num_iterations = 100  # Adjust as needed
    
    results = []
    print(f"Starting visualization with varying key lengths...")
    
    for key_length in key_lengths:
        start_time = time.time()
        estimated_probability = monte_carlo_collision_simulation(num_keys, key_length, num_iterations)
        end_time = time.time()
        results.append((key_length, estimated_probability))
        print(f"Key Length: {key_length}, Estimated Probability: {estimated_probability:.8f}, Execution Time: {end_time - start_time:.2f} seconds")
    
    # Plot results
    key_lengths, probabilities = zip(*results)
    plt.plot(key_lengths, probabilities, marker='o', linestyle='-', color='blue')
    plt.xlabel("Key Length (bits)", fontsize=12)
    plt.ylabel("Estimated Collision Probability", fontsize=12)
    plt.title("Collision Probability vs. Key Length", fontsize=14)
    plt.grid(True)
    plt.xticks(key_lengths)
    plt.savefig("../results/collision_probability_vs_key_length.png", dpi=300)
    plt.show()
