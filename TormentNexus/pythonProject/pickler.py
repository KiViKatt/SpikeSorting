import numpy as np
import spikeinterface.full as si
import pickle
import pandas as pd
import probeinterface as pi

# Function to create a probe from electrode geometry
def to_probeinterface(electrodes_df, **kwargs):
    probe_df = electrodes_df.copy()
    probe_df.rename(
        columns={
            "electrode": "contact_ids",
            "shank": "shank_ids",
            "x_coord": "x",
            "y_coord": "y",
        },
        inplace=True,
    )
    # Get the contact shapes. By default, it's set to circle with a radius of 10.
    contact_shapes = kwargs.get("contact_shapes", "circle")
    assert (
        contact_shapes in pi.probe._possible_contact_shapes
    ), f"contacts shape should be in {pi.probe._possible_contact_shapes}"

    probe_df["contact_shapes"] = contact_shapes
    if contact_shapes == "circle":
        probe_df["radius"] = kwargs.get("radius", 10)
    elif contact_shapes == "square":
        probe_df["width"] = kwargs.get("width", 10)
    elif contact_shapes == "rect":
        probe_df["width"] = kwargs.get("width")
        probe_df["height"] = kwargs.get("height")

    return pi.Probe.from_dataframe(probe_df)

# Recording parameters
num_channels = 32
duration = 10.0  # in seconds
sampling_frequency = 30000  # in Hz

# Generate time vector
num_samples = int(duration * sampling_frequency)
times = np.arange(num_samples) / sampling_frequency

# Define spike times and amplitudes for each unit
np.random.seed(42)  # For reproducibility
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
noise_level = 50
recording = np.random.normal(0, noise_level, size=(num_samples, num_channels))

# Create the electrode geometry DataFrame
electrode_geometry = pd.DataFrame({
    "electrode": np.arange(32),
    "x_coord": np.zeros(32),
    "y_coord": np.arange(0, 3200, 100),  # y-coordinate from 0 to 3100 µm
    "shank": np.zeros(32),
    "channel_idx": np.arange(32)
})

# Create the probe using the electrode geometry
probe = to_probeinterface(electrode_geometry)

# Set the device_channel_indices to map the probe's contacts to recording channels
probe.set_device_channel_indices(np.arange(num_channels))

# Create the RecordingExtractor
recording_extractor = si.NumpyRecording([recording], sampling_frequency)

# Attach the Probe to the Recording
recording_extractor.set_probe(probe)

# **Create spike_trains dictionary**
spike_trains = {}
for unit_id in range(num_units):
    # Convert spike times to sample indices
    spike_times = spike_times_list[unit_id] * sampling_frequency
    spike_trains[unit_id] = spike_times.astype(int)

# Save the recording data, sampling frequency, spike trains, and probe information
with open('synthetic_recording_with_probe.pkl', 'wb') as f:
    pickle.dump({
        'recording_data': recording,
        'sampling_frequency': sampling_frequency,
        'spike_trains': spike_trains,
        'probe': probe.to_dict()  # Save the probe as a dictionary
    }, f)

print("Data has been saved to 'synthetic_recording_with_probe.pkl'")
