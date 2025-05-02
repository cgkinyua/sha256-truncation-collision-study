import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import time, os, math

def load_kernel(kernel_file):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    kernel_path = os.path.join(current_dir, kernel_file)
    module = cp.RawModule(path=kernel_path)
    kernel = module.get_function('sha256_hash')
    return kernel

def generate_keys_gpu(num_keys, key_length):
    # Generate a random binary matrix (0s and 1s) on the GPU.
    return cp.random.randint(0, 2, size=(num_keys, key_length), dtype=cp.int32)

def has_collision_gpu(keys, key_length, kernel):
    # Convert the binary keys to a flat array and pack bits into uint8.
    keys_flat = keys.astype(bool).reshape(-1)
    packed = cp.packbits(keys_flat)
    bytes_per_key = (key_length + 7) // 8
    keys_gpu = packed.view(cp.uint8).reshape(keys.shape[0], bytes_per_key)
    
    # Allocate GPU memory for the computed hashes.
    hashes_gpu = cp.empty(keys.shape[0], dtype=cp.uint64)
    
    block_size = 256
    grid_size = (keys.shape[0] + block_size - 1) // block_size
    
    # Launch the CUDA kernel.
    kernel((grid_size,), (block_size,), (keys_gpu.data.ptr, hashes_gpu.data.ptr, keys.shape[0], key_length))
    
    # Check for collision by sorting the hash array and testing for adjacent duplicates.
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

def theoretical_collision_probability(n, N):
    # Using the birthday paradox approximation:
    # p ≈ 1 - exp(-n(n-1)/(2N))
    return 1 - math.exp(-n*(n-1)/(2*N))

def main():
    # Set up parameters:
    kernel_file = "sha256_kernel_32.cubin"  # Ensure your 32-bit truncated kernel is compiled and named accordingly.
    kernel = load_kernel(kernel_file)
    key_length = 256  # in bits
    num_iterations = 5
    batch_size = 10**5
    
    # Define a refined sweep over the number of keys near the birthday bound.
    # Here we use values from 10,000 to 100,000 in steps of 10,000.
    num_keys_values = [10000 * i for i in range(1, 11)]
    
    sim_probs = []
    exec_times = []
    
    print("Running sweep simulation for 32-bit truncated hash, key length 256 bits.")
    for n in num_keys_values:
        print(f"Simulating for num_keys = {n} ...")
        prob, exec_time = monte_carlo_collision_simulation(n, key_length, num_iterations, batch_size, kernel)
        print(f"  Estimated Collision Probability: {prob:.4f}, Average Execution Time: {exec_time:.2f} sec")
        sim_probs.append(prob)
        exec_times.append(exec_time)
    
    # Compute theoretical collision probabilities for a 32-bit hash (N = 2^32)
    N = 2**32
    theo_probs = [theoretical_collision_probability(n, N) for n in num_keys_values]
    
    # Create results directory (assumed to be at ../results relative to src)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(current_dir, '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Plot 1: Collision Probability vs Number of Keys (Simulation vs Theoretical)
    plt.figure(figsize=(8, 6))
    plt.plot(num_keys_values, theo_probs, label="Theoretical", marker="o", linestyle="-", color="blue")
    plt.plot(num_keys_values, sim_probs, label="Simulation", marker="x", linestyle="--", color="red")
    plt.xlabel("Number of Keys")
    plt.ylabel("Collision Probability")
    plt.title("Collision Probability vs Number of Keys (32-bit truncated hash)")
    plt.legend()
    plt.grid(True)
    # Using linear scale here for clarity; adjust as needed.
    collision_plot_file = os.path.join(results_dir, "collision_probability_vs_num_keys_32_bits_theoretical.pdf")
    plt.savefig(collision_plot_file)
    plt.close()
    
    # Plot 2: Execution Time vs Number of Keys
    plt.figure(figsize=(8, 6))
    plt.plot(num_keys_values, exec_times, marker="o", linestyle="-", color="red")
    plt.xlabel("Number of Keys")
    plt.ylabel("Average Execution Time (sec)")
    plt.title("Execution Time vs Number of Keys (32-bit truncated hash)")
    plt.grid(True)
    exec_time_plot_file = os.path.join(results_dir, "execution_time_vs_num_keys_32_bits.pdf")
    plt.savefig(exec_time_plot_file)
    plt.close()
    
    print("Sweep simulation completed.")
    print(f"Collision Probability Plot saved to: {collision_plot_file}")
    print(f"Execution Time Plot saved to: {exec_time_plot_file}")

if __name__ == "__main__":
    main()

