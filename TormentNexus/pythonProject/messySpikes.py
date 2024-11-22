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
    # Random spike times for each unit with added temporal jitter
    base_spike_times = np.sort(np.random.uniform(0, duration, size=50))
    jitter = np.random.normal(0, 0.001, size=base_spike_times.size)  # Jitter of 1 ms
    spike_times = base_spike_times + jitter
    spike_times = np.clip(spike_times, 0, duration)
    spike_times_list.append(spike_times)

    # Variable amplitudes for each spike with more inconsistency
    amplitudes = np.random.uniform(50 * (unit_id + 1), 100 * (unit_id + 1), size=spike_times.size)
    amplitude_noise = np.random.normal(0, 10 * (unit_id + 1), size=spike_times.size)
    amplitudes += amplitude_noise
    spike_amplitudes_list.append(amplitudes)

# Create a base spike waveform template with variability
waveform_duration = int(0.002 * sampling_frequency)  # 2 ms waveform
t_waveform = np.linspace(0, 2 * np.pi, waveform_duration)

# Initialize recording array with Gaussian noise (background activity)
recording = np.random.normal(0, 20, size=(num_samples, num_channels))

# Assemble the recording by adding spikes with variable amplitudes and shapes
for unit_id in range(num_units):
    spike_times = spike_times_list[unit_id]
    amplitudes = spike_amplitudes_list[unit_id]

    for spike_time, amplitude in zip(spike_times, amplitudes):
        spike_index = int(spike_time * sampling_frequency)
        if spike_index + waveform_duration < num_samples:
            # Create a variable waveform for each spike
            waveform_variation = np.random.normal(0, 0.1, size=waveform_duration)
            waveform_template = np.sin(t_waveform + np.random.uniform(-0.2, 0.2))
            waveform_template += waveform_variation
            waveform_template /= np.max(np.abs(waveform_template))  # Normalize

            # Add overlapping spikes occasionally
            overlap_chance = np.random.rand()
            if overlap_chance < 0.1:  # 10% chance of overlap
                overlap_unit = np.random.choice([uid for uid in range(num_units) if uid != unit_id])
                overlap_spike_time = spike_time + np.random.uniform(-0.001, 0.001)
                overlap_spike_index = int(overlap_spike_time * sampling_frequency)
                if overlap_spike_index + waveform_duration < num_samples:
                    overlap_amplitude = np.random.uniform(50 * (overlap_unit + 1), 100 * (overlap_unit + 1))
                    # Add overlapping spike
                    for ch in range(num_channels):
                        recording[overlap_spike_index:overlap_spike_index + waveform_duration, ch] += \
                            overlap_amplitude * waveform_template

            # Simulate missing data (dropouts) by skipping some spikes
            dropout_chance = np.random.rand()
            if dropout_chance < 0.05:  # 5% chance to skip the spike
                continue

            # Add the waveform to each channel with per-channel variability
            for ch in range(num_channels):
                channel_variation = np.random.normal(0, 0.05)
                recording[spike_index:spike_index + waveform_duration, ch] += \
                    amplitude * (waveform_template + channel_variation)

# Simulate occasional artifacts (e.g., large amplitude noise)
num_artifacts = 20
artifact_times = np.random.uniform(0, duration, size=num_artifacts)
for artifact_time in artifact_times:
    artifact_index = int(artifact_time * sampling_frequency)
    if artifact_index + 100 < num_samples:
        artifact = np.random.normal(0, 500, size=100)  # Large amplitude artifact
        ch = np.random.randint(0, num_channels)
        recording[artifact_index:artifact_index + 100, ch] += artifact

# Create RecordingExtractor
recording_extractor = si.NumpyRecording([recording], sampling_frequency)

# Create SortingExtractor
spike_trains = {}
for unit_id in range(num_units):
    spike_times = spike_times_list[unit_id] * sampling_frequency  # Convert to sample indices
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
             recording[start_sample:end_sample, ch] + ch * 500,
             label=f'Channel {ch}')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Messy and Inconsistent Synthetic Recording')
plt.legend()
plt.show()
