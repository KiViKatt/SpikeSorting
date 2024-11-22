import numpy as np
import spikeinterface.full as si
import matplotlib.pyplot as plt

# Recording parameters
num_channels = 32
duration = 10.0  # in seconds
sampling_frequency = 30000  # in Hz

# Generate time vector
num_samples = int(duration * sampling_frequency)
times = np.arange(num_samples) / sampling_frequency

# Define spike times and amplitudes for each unit
np.random.seed(42)  # for reproducibility
num_units = 10  # Increase number of units for more complexity
spike_times_list = []
spike_amplitudes_list = []

for unit_id in range(num_units):
    # Random spike times for each unit with larger temporal jitter
    base_spike_times = np.sort(np.random.uniform(0, duration, size=500))
    jitter = np.random.normal(0, 0.005, size=base_spike_times.size)  # Jitter of 5 ms
    spike_times = base_spike_times + jitter
    spike_times = np.clip(spike_times, 0, duration)
    spike_times_list.append(spike_times)

    # Variable amplitudes for each spike with more inconsistency
    amplitudes = np.random.uniform(20, 200, size=spike_times.size)
    amplitude_noise = np.random.normal(0, 50, size=spike_times.size)
    amplitudes += amplitude_noise
    spike_amplitudes_list.append(amplitudes)

# Initialize recording array with Gaussian noise (background activity)
# Start with higher noise level
noise_level = 50
recording = np.random.normal(0, noise_level, size=(num_samples, num_channels))

# Introduce periods with increased noise
num_noise_bursts = 10  # Number of high-noise periods
burst_duration = 0.5  # Duration of each noise burst in seconds
for _ in range(num_noise_bursts):
    burst_start_time = np.random.uniform(0, duration - burst_duration)
    burst_start_idx = int(burst_start_time * sampling_frequency)
    burst_end_idx = int((burst_start_time + burst_duration) * sampling_frequency)
    high_noise_level = 200  # Increased noise level
    recording[burst_start_idx:burst_end_idx, :] += np.random.normal(
        0, high_noise_level, size=(burst_end_idx - burst_start_idx, num_channels))

# Create per-channel spike waveform templates for each unit
waveform_duration = int(0.002 * sampling_frequency)  # 2 ms waveform
# Generate random waveforms for each unit and channel
unit_waveforms = {}
for unit_id in range(num_units):
    channel_waveforms = {}
    for ch in range(num_channels):
        # Create a random waveform for each channel and unit
        waveform_template = np.random.normal(0, 1, size=waveform_duration)
        waveform_template /= np.max(np.abs(waveform_template) + 1e-9)  # Normalize
        channel_waveforms[ch] = waveform_template
    unit_waveforms[unit_id] = channel_waveforms

# Assemble the recording by adding spikes with variable amplitudes and shapes
for unit_id in range(num_units):
    spike_times = spike_times_list[unit_id]
    amplitudes = spike_amplitudes_list[unit_id]

    for spike_time, amplitude in zip(spike_times, amplitudes):
        spike_index = int(spike_time * sampling_frequency)
        if 0 <= spike_index and spike_index + waveform_duration <= num_samples:
            # Simulate missing data (dropouts) by skipping some spikes
            dropout_chance = np.random.rand()
            if dropout_chance < 0.1:  # 10% chance to skip the spike
                continue

            # Randomly decide if spike occurs on this channel
            channels_with_spike = np.random.choice(
                [True, False], size=num_channels, p=[0.8, 0.2])  # 80% chance spike is on the channel

            # Add the waveform to each channel with random scaling
            for ch in range(num_channels):
                if channels_with_spike[ch]:
                    # Retrieve the unique waveform for this unit and channel
                    waveform_template = unit_waveforms[unit_id][ch]

                    # Add per-sample noise to the waveform
                    sample_noise = np.random.normal(0, 0.2, size=waveform_duration)
                    waveform = amplitude * (waveform_template + sample_noise)

                    # Apply random amplitude scaling
                    scaling_factor = np.random.uniform(0.5, 1.5)
                    waveform *= scaling_factor

                    # Add the waveform to the recording
                    recording[spike_index:spike_index + waveform_duration, ch] += waveform

            # Increase overlapping spikes probability
            overlap_chance = np.random.rand()
            if overlap_chance < 0.5:  # 50% chance of overlap
                num_overlaps = np.random.randint(1, num_units)
                overlapping_units = np.random.choice(
                    [uid for uid in range(num_units) if uid != unit_id],
                    size=num_overlaps, replace=False)
                for overlap_unit in overlapping_units:
                    overlap_spike_time = spike_time + np.random.uniform(-0.001, 0.001)  # Slight time shift
                    # Ensure overlap_spike_time is within valid bounds
                    max_overlap_time = (num_samples - waveform_duration) / sampling_frequency
                    overlap_spike_time = np.clip(overlap_spike_time, 0, max_overlap_time)
                    overlap_spike_index = int(overlap_spike_time * sampling_frequency)
                    if 0 <= overlap_spike_index and overlap_spike_index + waveform_duration <= num_samples:
                        overlap_amplitude = np.random.uniform(20, 200)
                        # Randomly decide if spike occurs on this channel
                        overlap_channels_with_spike = np.random.choice(
                            [True, False], size=num_channels, p=[0.8, 0.2])
                        for ch in range(num_channels):
                            if overlap_channels_with_spike[ch]:
                                # Retrieve the unique waveform for the overlapping unit and channel
                                overlap_waveform_template = unit_waveforms[overlap_unit][ch]
                                sample_noise = np.random.normal(0, 0.2, size=waveform_duration)
                                overlap_waveform = overlap_amplitude * (overlap_waveform_template + sample_noise)
                                # Apply random amplitude scaling
                                scaling_factor = np.random.uniform(0.5, 1.5)
                                overlap_waveform *= scaling_factor
                                recording[overlap_spike_index:overlap_spike_index + waveform_duration, ch] += overlap_waveform

# Simulate channel cross-talk by adding a fraction of neighboring channels' signals
cross_talk_level = 0.5  # 50% of neighboring channel signal
for ch in range(num_channels):
    if ch > 0:
        recording[:, ch] += cross_talk_level * recording[:, ch - 1]
    if ch < num_channels - 1:
        recording[:, ch] += cross_talk_level * recording[:, ch + 1]

# Simulate occasional artifacts (e.g., large amplitude noise)
num_artifacts = 50
artifact_times = np.random.uniform(0, duration, size=num_artifacts)
for artifact_time in artifact_times:
    artifact_index = int(artifact_time * sampling_frequency)
    artifact_duration = np.random.randint(50, 500)  # Random artifact duration between 50 and 500 samples
    if artifact_index + artifact_duration < num_samples:
        artifact = np.random.normal(0, 1000, size=artifact_duration)  # Large amplitude artifact
        ch = np.random.randint(0, num_channels)
        recording[artifact_index:artifact_index + artifact_duration, ch] += artifact

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

# Calculate the maximum absolute amplitude across all channels in the segment
max_amplitude = np.max(np.abs(recording[start_sample:end_sample, :]))

# Define the offset increment based on the max amplitude and a scaling factor
scaling_factor = 2  # Adjust this factor as needed
offset_increment = max_amplitude * scaling_factor

offset = 0
colors = plt.cm.get_cmap('tab20', num_channels)

for ch in range(num_channels):
    plt.plot(times[start_sample:end_sample],
             recording[start_sample:end_sample, ch] + offset,
             label=f'Channel {ch}', color=colors(ch))
    offset += offset_increment  # Increase offset for next channel

plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Synthetic Recording with Multiple Channels')
plt.legend(loc='upper right', ncol=4, bbox_to_anchor=(1.0, 1.0))
plt.tight_layout()
plt.show()
