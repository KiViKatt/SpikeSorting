import numpy as np
import spikeinterface.full as si
import matplotlib.pyplot as plt
import pickle

# Recording parameters
num_channels = 32
duration = 10.0  # in seconds
sampling_frequency = 30000  # in Hz

# Generate time vector
num_samples = int(duration * sampling_frequency)
times = np.arange(num_samples) / sampling_frequency

# Define spike times and amplitudes for each unit
np.random.seed(42)  # for reproducibility
num_units = 3
spike_times_list = []
spike_amplitudes_list = []

for unit_id in range(num_units):
    # Random spike times for each unit
    spike_times = np.sort(np.random.uniform(0, duration, size=50))
    spike_times_list.append(spike_times)

    # Variable amplitudes for each spike
    amplitudes = np.random.uniform(50 * (unit_id + 1), 100 * (unit_id + 1), size=spike_times.size)
    spike_amplitudes_list.append(amplitudes)

# Create a simple spike waveform template
waveform_duration = int(0.002 * sampling_frequency)  # 2 ms waveform
t_waveform = np.linspace(0, 2 * np.pi, waveform_duration)
waveform_template = np.sin(t_waveform)

# Normalize the waveform
waveform_template /= np.max(np.abs(waveform_template))

# Initialize recording array
recording = np.zeros((num_samples, num_channels))

# Assemble the recording by adding spikes with variable amplitudes
for unit_id in range(num_units):
    spike_times = spike_times_list[unit_id]
    amplitudes = spike_amplitudes_list[unit_id]

    for spike_time, amplitude in zip(spike_times, amplitudes):
        spike_index = int(spike_time * sampling_frequency)
        if spike_index + waveform_duration < num_samples:
            # Add the waveform to each channel (customize as needed)
            for ch in range(num_channels):
                recording[spike_index:spike_index + waveform_duration, ch] += amplitude * waveform_template

# **Corrected Line**
# Create RecordingExtractor
recording_extractor = si.NumpyRecording([recording], sampling_frequency)

# Create SortingExtractor
spike_trains = {}
for unit_id in range(num_units):
    spike_times = spike_times_list[unit_id] * sampling_frequency  # convert to sample indices
    spike_trains[unit_id] = spike_times.astype(int)

sorting_extractor = si.NumpySorting.from_unit_dict(spike_trains, sampling_frequency)

# Plot a segment of the recording
start_time = 0
end_time = 0.1  # First 100 ms
start_sample = int(start_time * sampling_frequency)
end_sample = int(end_time * sampling_frequency)

plt.figure(figsize=(15, 20))
for ch in range(num_channels):
    plt.plot(times[start_sample:end_sample],
             recording[start_sample:end_sample, ch] + ch * 200,
             label=f'Channel {ch}')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Synthetic Recording with Variable Spike Amplitudes')
plt.legend()
plt.show()
