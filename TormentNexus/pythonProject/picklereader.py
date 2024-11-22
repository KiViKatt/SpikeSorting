import numpy as np
import spikeinterface.full as si
import matplotlib.pyplot as plt
import pickle  # Import pickle to load the data

# Load the data from the .pickle file
with open('synthetic_recording.pkl', 'rb') as f:
    data = pickle.load(f)

# Extract the recording data, spike trains, and sampling frequency
recording = data['recording']
spike_trains = data['spike_trains']
sampling_frequency = data['sampling_frequency']

# Reconstruct the RecordingExtractor
recording_extractor = si.NumpyRecording([recording], sampling_frequency)

# Reconstruct the SortingExtractor
sorting_extractor = si.NumpySorting.from_unit_dict(spike_trains, sampling_frequency)

# Optionally, proceed to use the recording_extractor and sorting_extractor
# For example, visualize a segment of the loaded recording

# Plot a segment of the recording
start_time = 0
end_time = 0.1  # First 100 ms
start_sample = int(start_time * sampling_frequency)
end_sample = int(end_time * sampling_frequency)
times = np.arange(recording.shape[0]) / sampling_frequency

plt.figure(figsize=(15, 20))

# Calculate the maximum absolute amplitude across all channels in the segment
max_amplitude = np.max(np.abs(recording[start_sample:end_sample, :]))

# Define the offset increment based on the max amplitude and a scaling factor
scaling_factor = 2  # Adjust this factor as needed
offset_increment = max_amplitude * scaling_factor

offset = 0
num_channels = recording.shape[1]
colors = plt.cm.get_cmap('tab20', num_channels)

for ch in range(num_channels):
    plt.plot(times[start_sample:end_sample],
             recording[start_sample:end_sample, ch] + offset,
             label=f'Channel {ch}', color=colors(ch))
    offset += offset_increment  # Increase offset for next channel

plt.xlabel('Time (s)')
plt.ylabel('Amplitude + Offset per Channel')
plt.title('Loaded Synthetic Recording')
plt.legend(loc='upper right', ncol=4, bbox_to_anchor=(1.0, 1.0))
plt.tight_layout()
plt.show()

# Output an array of data to the console
# For example, output the recording data of the first 5 samples and first 5 channels
print("Sample data from the recording (first 5 samples and first 5 channels):")
print(recording[:5, :5])

# Output spike times for a specific unit
unit_id = 0  # Change to the desired unit ID
print(f"\nSpike times for unit {unit_id} (in samples):")
print(sorting_extractor.get_unit_spike_train(unit_id))

# Optionally, you can proceed to use the recording_extractor and sorting_extractor
# with SpikeInterface tools for spike sorting or further analysis.

# Example: Printing basic information
print(f"\nNumber of channels: {recording_extractor.get_num_channels()}")
print(f"Sampling frequency: {recording_extractor.get_sampling_frequency()} Hz")
print(f"Duration: {recording_extractor.get_total_duration()} seconds")

print(f"\nNumber of units: {len(sorting_extractor.unit_ids)}")
for unit_id in sorting_extractor.unit_ids:
    num_spikes = len(sorting_extractor.get_unit_spike_train(unit_id))
    print(f"Unit {unit_id} has {num_spikes} spikes")
