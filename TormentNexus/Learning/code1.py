import numpy as np
import spikeinterface.core as sc
import spikeinterface.sorters as ss
import spikeinterface.comparison as scmp
import spikeinterface.widgets as sw
import matplotlib.pyplot as plt

# Generate synthetic recording
num_channels = 32
duration = 10  # seconds
sampling_frequency = 30000  # Hz
recording = sc.generate_recording(num_channels=num_channels, sampling_frequency=sampling_frequency, durations=[duration])

# Manually create synthetic spike times for 5 units and make a NumpySorting object
spike_times_dict = {unit_id: np.sort(np.random.randint(0, duration * sampling_frequency, size=100)) for unit_id in range(5)}
unit_ids = list(spike_times_dict.keys())
spike_trains = [np.array(times) for times in spike_times_dict.values()]

# Create sorting object
sorting_true = sc.NumpySorting.from_times_labels(times_list=spike_trains, labels_list=unit_ids, sampling_frequency=sampling_frequency)

# Check the synthetic recording and sorting information
print("Synthetic recording details:", recording)
print("True sorting details:", sorting_true)

# Run a spike sorter (e.g., SpyKING CIRCUS) with the remove_existing_folder option
sorting = ss.run_sorter("spykingcircus", recording, remove_existing_folder=True)

# Compare the sorted results with the true spikes
cmp = scmp.compare_sorter_to_ground_truth(sorting_true, sorting)

# Print comparison metrics
print("Comparison metrics:")
print(cmp.get_performance())

# Plot waveforms to visualize
wfs = sw.plot_unit_waveforms(recording, sorting)
plt.show()
