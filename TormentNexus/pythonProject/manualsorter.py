import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

import spikeinterface.full as si
import spikeinterface.preprocessing as spre
from spikeinterface import NumpySorting
from probeinterface import Probe, ProbeGroup

# Step 1: Load the synthetic data from the .pickle file
with open('synthetic_recording_with_probe.pkl', 'rb') as f:
    data = pickle.load(f)

# Extract the recording data, sampling frequency, and probe
recording_data = data['recording_data']
sampling_frequency = data['sampling_frequency']
probe_dict = data['probe']  # Load the saved probe as a dictionary

# Step 2: Reconstruct the Recording object
recording = si.NumpyRecording([recording_data], sampling_frequency)

# Step 3: Reconstruct the Probe and attach it to the Recording
try:
    # Reconstruct the Probe from the saved dictionary
    probe = Probe.from_dict(probe_dict)
    print("Probe successfully reconstructed.")

    # Ensure device_channel_indices match recording channels
    if probe.device_channel_indices is None or len(probe.device_channel_indices) != recording.get_num_channels():
        probe.set_device_channel_indices(np.arange(recording.get_num_channels()))
        print("Device channel indices set.")

    # Create a ProbeGroup and add the Probe
    probegroup = ProbeGroup()
    probegroup.add_probe(probe)
    print("ProbeGroup created with the Probe.")

    # Attach the ProbeGroup to the Recording
    recording.set_probegroup(probegroup)
    print("ProbeGroup successfully attached to the recording.")

    # Directly confirm ProbeGroup contents
    attached_probegroup = recording.get_probegroup()
    if attached_probegroup is not None:
        print(f"ProbeGroup attached with {len(attached_probegroup.probes)} Probe(s).")
    else:
        raise ValueError("Probe attachment verification failed.")
except Exception as e:
    raise ValueError(f"Error with Probe reconstruction or attachment: {e}")

# Step 4: Apply a bandpass filter to remove slow drifts and high-frequency noise
recording_filtered = spre.bandpass_filter(recording, freq_min=300, freq_max=6000)

# Step 5: Compute noise levels manually using Median Absolute Deviation (MAD)
def compute_noise_levels_manual(recording):
    channel_ids = recording.channel_ids
    noise_levels = []
    for ch in channel_ids:
        # Get the trace for the channel
        trace = recording.get_traces(channel_ids=[ch]).flatten()
        # Compute the median absolute deviation (MAD)
        mad = np.median(np.abs(trace - np.median(trace))) / 0.6745
        noise_levels.append(mad)
    noise_levels = np.array(noise_levels)
    return noise_levels

noise_levels = compute_noise_levels_manual(recording_filtered)

# Step 6: Set detection thresholds at 5 times the noise levels
thresholds = 5 * noise_levels

# Step 7: Detect spikes that exceed the thresholds manually
def detect_peaks_manual(recording, thresholds, peak_sign='both', detect_interval=1):
    traces = recording.get_traces()
    num_samples, num_channels = traces.shape
    peak_list = []
    for ch in range(num_channels):
        trace = traces[:, ch]
        threshold = thresholds[ch]
        if peak_sign == 'both':
            idx_spikes = np.where(np.abs(trace) > threshold)[0]
        elif peak_sign == 'positive':
            idx_spikes = np.where(trace > threshold)[0]
        elif peak_sign == 'negative':
            idx_spikes = np.where(trace < -threshold)[0]
        else:
            raise ValueError("Invalid peak_sign")

        # Enforce minimum interval between peaks to avoid duplicates
        if len(idx_spikes) > 0:
            # Sort indices to ensure proper diff calculation
            idx_spikes = np.sort(idx_spikes)
            # Enforce minimum interval between peaks
            idx_diff = np.diff(idx_spikes)
            keep_inds = np.insert(idx_diff > detect_interval, 0, True)
            idx_spikes = idx_spikes[keep_inds]
            for idx in idx_spikes:
                peak_list.append((idx, ch, trace[idx]))
    peaks = np.array(peak_list, dtype=[('sample_index', 'int64'), ('channel_index', 'int64'), ('amplitude', 'float64')])
    return peaks

# Detect peaks
peaks = detect_peaks_manual(
    recording_filtered,
    thresholds,
    peak_sign='both',
    detect_interval=int(sampling_frequency * 0.001)  # 1 ms interval
)

# Step 8: Create a Sorting object from the detected peaks
sorting = NumpySorting.from_times_labels(
    times_list=[peaks['sample_index']],
    labels_list=[np.zeros(len(peaks), dtype='int64')],  # Single unit
    sampling_frequency=sampling_frequency
)

# Step 9: Create a SortingAnalyzer
sa_folder = 'sorting_analyzer_5std'

# Create the SortingAnalyzer object
sorting_analyzer = si.create_sorting_analyzer(
    sorting=sorting,
    recording=recording_filtered,
    analyzer_name=sa_folder,
    ms_before=1,
    ms_after=2,
)

# Step 10: Compute waveforms using the SortingAnalyzer
sorting_analyzer.compute_waveforms(max_spikes_per_unit=None)

# Step 11: Get the waveforms for the unit (unit_id=0)
unit_id = 0
waveforms = sorting_analyzer.get_waveforms(unit_id=unit_id)

# Step 12: Reshape waveforms for clustering
num_spikes, num_channels, num_samples = waveforms.shape
waveforms_reshaped = waveforms.reshape((num_spikes, num_channels * num_samples))

# Step 13: Perform PCA on the waveforms to reduce dimensionality
pca = PCA(n_components=5)  # Keep the first 5 principal components
waveforms_pca = pca.fit_transform(waveforms_reshaped)

# Step 14: Cluster the waveforms using KMeans clustering
num_clusters = 10  # Adjust based on expected units
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(waveforms_pca)

# Step 15: Visualize the mean waveform for each cluster
for cluster_id in np.unique(cluster_labels):
    cluster_waveforms = waveforms[cluster_labels == cluster_id]
    mean_waveform = np.mean(cluster_waveforms, axis=0)

    plt.figure(figsize=(12, 8))
    for ch in range(num_channels):
        plt.plot(mean_waveform[ch], label=f'Channel {ch}')
    plt.title(f'Cluster {cluster_id} Mean Waveform')
    plt.xlabel('Time (samples)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.show()

# Optional: Print the number of spikes in each cluster
for cluster_id in np.unique(cluster_labels):
    num_spikes_in_cluster = np.sum(cluster_labels == cluster_id)
    print(f"Cluster {cluster_id} has {num_spikes_in_cluster} spikes")

