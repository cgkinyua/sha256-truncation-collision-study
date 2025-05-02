# src/plot_execution_time_vs_key_length.py

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
    execution_times = []
    
    for _ in range(num_iterations):
        total_batches = (num_keys + batch_size - 1) // batch_size
        batch_collision_exists = False
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, num_keys)
            batch_num_keys = end_idx - start_idx
            keys = generate_keys_gpu(batch_num_keys, key_length)
            batch_start_time = time.time()
            batch_collision = has_collision_gpu(keys)
            batch_end_time = time.time()
            execution_times.append(batch_end_time - batch_start_time)
            if batch_collision:
                batch_collision_exists = True
                break
        
        collision_count += int(batch_collision_exists)
    
    estimated_probability = collision_count / num_iterations
    avg_execution_time = np.mean(execution_times)
    return estimated_probability, avg_execution_time

if __name__ == "__main__":
    key_lengths = [64, 128, 256]  # Vary key lengths
    num_keys = 10**7  # Large number of keys
    num_iterations = 5  # Reduce iterations for efficiency
    batch_size = 10**5  # Process keys in smaller batches of 100,000
    
    results = []
    print(f"Starting visualization of execution time vs. key length...")
    
    for key_length in key_lengths:
        estimated_probability, avg_execution_time = monte_carlo_collision_simulation(
            num_keys, key_length, num_iterations, batch_size=batch_size
        )
        results.append((key_length, avg_execution_time))
        print(f"Key Length: {key_length}, Average Execution Time: {avg_execution_time:.2f} seconds")
    
    # Plot results
    key_lengths, execution_times = zip(*results)
    plt.bar(key_lengths, execution_times, color=['blue', 'green', 'orange'])
    plt.xlabel("Key Length (bits)", fontsize=12)
    plt.ylabel("Average Execution Time (seconds)", fontsize=12)
    plt.title("Execution Time vs. Key Length (Post-Quantum Keys)", fontsize=14)
    plt.grid(axis='y')
    plt.xticks(key_lengths)
    plt.savefig("../results/execution_time_vs_key_length_post_quantum.png", dpi=300)
    plt.show()
