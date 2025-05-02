# src/plot_collision_probability_vs_key_length.py

import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import time

def generate_keys_gpu(num_keys, key_length):
    return cp.random.randint(0, 2, size=(num_keys, key_length), dtype=np.int32)

def has_collision_gpu(keys):
    # Simulate truncated hash values directly on the GPU
    key_hashes = cp.arange(keys.shape[0], dtype=cp.uint64)  # Generate unique hash indices
    truncated_hashes = key_hashes & ((1 << 64) - 1)  # Truncate to lower 64 bits using bitwise AND
    
    # Check for collisions on the GPU
    collision_exists = cp.any(cp.diff(cp.sort(truncated_hashes)) == 0)
    return int(collision_exists)

def monte_carlo_collision_simulation(num_keys, key_length, num_iterations, batch_size=10**5):
    collision_count = 0
    
    for _ in range(num_iterations):
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
                break
        
        collision_count += int(batch_collision_exists)
    
    estimated_probability = collision_count / num_iterations
    return estimated_probability

if __name__ == "__main__":
    key_lengths = [64, 128, 256]  # Vary key lengths
    num_keys = 10**7  # Large number of keys
    num_iterations = 5  # Reduce iterations for efficiency
    batch_size = 10**5  # Process keys in smaller batches of 100,000
    
    results = []
    print(f"Starting visualization of collision probability vs. key length...")
    
    for key_length in key_lengths:
        start_time = time.time()
        estimated_probability = monte_carlo_collision_simulation(
            num_keys, key_length, num_iterations, batch_size=batch_size
        )
        end_time = time.time()
        results.append((key_length, estimated_probability))
        print(f"Key Length: {key_length}, Estimated Probability: {estimated_probability:.8f}, Execution Time: {end_time - start_time:.2f} seconds")
    
    # Plot results
    key_lengths, probabilities = zip(*results)
    plt.plot(key_lengths, probabilities, marker='o', linestyle='-', color='blue')
    plt.xlabel("Key Length (bits)", fontsize=12)
    plt.ylabel("Estimated Collision Probability", fontsize=12)
    plt.title("Collision Probability vs. Key Length (Post-Quantum Keys)", fontsize=14)
    plt.grid(True)
    plt.xticks(key_lengths)
    plt.savefig("../results/collision_probability_vs_key_length_post_quantum.png", dpi=300)
    plt.show()
