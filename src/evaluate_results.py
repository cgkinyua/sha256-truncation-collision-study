import os
import re
import numpy as np
import matplotlib.pyplot as plt

# Define the name of the simulation log file.
log_file = 'simulation_log.txt'

# Data lists to store parsed values.
key_lengths = []
collision_probabilities = []
execution_times = []

# Regular expression pattern to extract values.
pattern = r"Key Length:\s*(\d+),\s*Estimated Probability:\s*([\d\.Ee+-]+),\s*Execution Time:\s*([\d\.Ee+-]+) seconds"

# Read and parse the log file.
with open(log_file, 'r') as f:
    for line in f:
        match = re.search(pattern, line)
        if match:
            key_length = int(match.group(1))
            collision_probability = float(match.group(2))
            exec_time = float(match.group(3))
            key_lengths.append(key_length)
            collision_probabilities.append(collision_probability)
            execution_times.append(exec_time)

# Convert lists to numpy arrays.
key_lengths = np.array(key_lengths)
collision_probabilities = np.array(collision_probabilities)
execution_times = np.array(execution_times)

# Aggregate results by unique key length (taking averages if there are multiple runs).
unique_key_lengths = np.unique(key_lengths)
avg_probabilities = []
avg_times = []

for k in unique_key_lengths:
    indices = np.where(key_lengths == k)[0]
    avg_probabilities.append(np.mean(collision_probabilities[indices]))
    avg_times.append(np.mean(execution_times[indices]))

avg_probabilities = np.array(avg_probabilities)
avg_times = np.array(avg_times)

# Determine the results directory (one level up from src/results).
current_dir = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(current_dir, '..', 'results')
os.makedirs(results_dir, exist_ok=True)

# Plot 1: Collision Probability vs Key Length.
plt.figure(figsize=(8, 6))
plt.plot(unique_key_lengths, avg_probabilities, marker='o', linestyle='-', color='blue')
plt.xlabel('Key Length')
plt.ylabel('Estimated Collision Probability')
plt.title('Collision Probability vs Key Length')
plt.grid(True)
collision_prob_fig = os.path.join(results_dir, 'collision_probability_vs_key_length.png')
plt.savefig(collision_prob_fig)
plt.close()

# Plot 2: Execution Time vs Key Length.
plt.figure(figsize=(8, 6))
plt.plot(unique_key_lengths, avg_times, marker='o', linestyle='-', color='red')
plt.xlabel('Key Length')
plt.ylabel('Average Execution Time (seconds)')
plt.title('Execution Time vs Key Length')
plt.grid(True)
exec_time_fig = os.path.join(results_dir, 'execution_time_vs_key_length.png')
plt.savefig(exec_time_fig)
plt.close()

print("Plots have been saved in the results directory:")
print(f"  {collision_prob_fig}")
print(f"  {exec_time_fig}")

