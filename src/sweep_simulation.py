import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import time, os

def load_kernel(kernel_file):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    kernel_path = os.path.join(current_dir, kernel_file)
    module = cp.RawModule(path=kernel_path)
    kernel = module.get_function('sha256_hash')
    return kernel

def generate_keys_gpu(num_keys, key_length):
    return cp.random.randint(0, 2, size=(num_keys, key_length), dtype=cp.int32)

def has_collision_gpu(keys, key_length, kernel):
    keys_flat = keys.astype(bool).reshape(-1)
    packed = cp.packbits(keys_flat)
    bytes_per_key = (key_length + 7) // 8
    keys_gpu = packed.view(cp.uint8).reshape(keys.shape[0], bytes_per_key)
    hashes_gpu = cp.empty(keys.shape[0], dtype=cp.uint64)
    block_size = 256
    grid_size = (keys.shape[0] + block_size - 1) // block_size
    kernel((grid_size,), (block_size,), (keys_gpu.data.ptr, hashes_gpu.data.ptr, keys.shape[0], key_length))
    collision_exists = cp.any(cp.diff(cp.sort(hashes_gpu)) == 0)
    return int(collision_exists)

def monte_carlo_collision_simulation(num_keys, key_length, num_iterations, batch_size, kernel):
    collision_count = 0
    total_time = 0.0
    for _ in range(num_iterations):
        start = time.time()
        total_batches = (num_keys + batch_size - 1) // batch_size
        batch_collision_exists = False
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, num_keys)
            batch_num_keys = end_idx - start_idx
            keys = generate_keys_gpu(batch_num_keys, key_length)
            if has_collision_gpu(keys, key_length, kernel):
                batch_collision_exists = True
                break
        end = time.time()
        total_time += (end - start)
        collision_count += int(batch_collision_exists)
    avg_time = total_time / num_iterations
    estimated_probability = collision_count / num_iterations
    return estimated_probability, avg_time

def main():
    # Parameters
    truncation_bits = 32  # Using 32-bit truncated hash
    kernel_file = f"sha256_kernel_{truncation_bits}.cubin"  # Ensure this file exists (from previous steps)
    key_length = 256  # bits
    num_iterations = 5
    batch_size = 10**5
    # Sweep over number of keys (logarithmically spaced)
    num_keys_values = [10**4, 10**5, 10**6, 10**7]
    
    kernel = load_kernel(kernel_file)
    
    collision_probs = []
    exec_times = []
    
    print(f"Running sweep simulation for {truncation_bits}-bit truncated hash, key length {key_length} bits.")
    for num_keys in num_keys_values:
        print(f"Simulating for num_keys = {num_keys} ...")
        prob, exec_time = monte_carlo_collision_simulation(num_keys, key_length, num_iterations, batch_size, kernel)
        print(f"  Estimated Collision Probability: {prob:.4f}, Average Execution Time: {exec_time:.2f} sec")
        collision_probs.append(prob)
        exec_times.append(exec_time)
    
    # Save plots in the results directory (assumed to be at ../results)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(current_dir, '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Plot: Collision Probability vs Number of Keys
    plt.figure(figsize=(8,6))
    plt.plot(num_keys_values, collision_probs, marker='o', linestyle='-', color='blue')
    plt.xlabel('Number of Keys')
    plt.ylabel('Estimated Collision Probability')
    plt.title(f'Collision Probability vs Number of Keys ({truncation_bits}-bit truncated hash)')
    plt.grid(True)
    plt.xscale('log')
    collision_fig = os.path.join(results_dir, f'collision_probability_vs_num_keys_{truncation_bits}_bits.png')
    plt.savefig(collision_fig)
    plt.close()
    
    # Plot: Execution Time vs Number of Keys
    plt.figure(figsize=(8,6))
    plt.plot(num_keys_values, exec_times, marker='o', linestyle='-', color='red')
    plt.xlabel('Number of Keys')
    plt.ylabel('Average Execution Time (sec)')
    plt.title(f'Execution Time vs Number of Keys ({truncation_bits}-bit truncated hash)')
    plt.grid(True)
    plt.xscale('log')
    exec_time_fig = os.path.join(results_dir, f'execution_time_vs_num_keys_{truncation_bits}_bits.png')
    plt.savefig(exec_time_fig)
    plt.close()
    
    print("Sweep simulation completed.")
    print(f"Collision Probability Plot saved to: {collision_fig}")
    print(f"Execution Time Plot saved to: {exec_time_fig}")

if __name__ == "__main__":
    main()

