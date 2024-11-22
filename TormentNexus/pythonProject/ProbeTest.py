import numpy as np
import spikeinterface.full as si
from probeinterface import Probe, ProbeGroup
import json

# Step 1: Create a synthetic recording
num_channels = 32
sampling_frequency = 30000
duration = 10.0
recording = si.NumpyRecording(
    [np.random.randn(int(sampling_frequency * duration), num_channels)],
    sampling_frequency
)

# Step 2: Create a simple Probe
probe = Probe(ndim=2, si_units='um')
probe.set_contacts(
    positions=np.zeros((num_channels, 2)),
    shapes='circle',
    shape_params={'radius': 5},
    contact_ids=np.arange(num_channels)
)
probe.set_device_channel_indices(np.arange(num_channels))

# Print Probe details
print("Probe Details:")
print(f"- Summary: {probe}")
print(f"- Device channel indices: {probe.device_channel_indices}")
print(f"- Number of contacts: {len(probe.contact_ids)}")

# Step 3: Attach the Probe directly
try:
    print("\nAttaching Probe to the recording...")
    recording.set_probe(probe)
    print("Probe successfully attached using set_probe().")
except Exception as e:
    print(f"Error attaching Probe using set_probe(): {e}")

# Verify Probe attachment directly
try:
    attached_probe = recording.get_probe()
    print(f"\nAttached Probe: {attached_probe}")
except Exception as e:
    print(f"Error accessing attached Probe: {e}")

# Step 4: Forcefully create and attach a ProbeGroup
try:
    print("\nCreating a ProbeGroup and attaching...")
    probegroup = ProbeGroup()
    probegroup.add_probe(probe)

    recording.set_probegroup(probegroup)
    print("ProbeGroup successfully attached using set_probegroup().")
except Exception as e:
    print(f"Error attaching ProbeGroup: {e}")

# Step 5: Verify the ProbeGroup attachment
try:
    probegroup_attached = recording.get_probegroup()
    if probegroup_attached:
        print(f"Attached ProbeGroup successfully with {len(probegroup_attached.probes)} probes.")
        for i, attached_probe in enumerate(probegroup_attached.probes):
            print(f"Probe {i}: {attached_probe}")
    else:
        print("Failed to retrieve attached ProbeGroup.")
except Exception as e:
    print(f"Error inspecting attached ProbeGroup: {e}")

# Step 6: Save Probe details to JSON
try:
    print("\nSaving Probe details to 'probe_debug.json'...")
    probe_dict = probe.to_dict()

    # Convert numpy arrays to lists for JSON compatibility
    for key, value in probe_dict.items():
        if isinstance(value, np.ndarray):
            probe_dict[key] = value.tolist()

    with open("probe_debug.json", "w") as f:
        json.dump(probe_dict, f)
    print("Probe details saved to 'probe_debug.json'.")
except Exception as e:
    print(f"Error saving Probe details to JSON: {e}")

# Step 7: Print recording summary
print("\nRecording Summary:")
print(recording)

# Step 8: Diagnostics
print("\nDiagnostics:")
print(f"- Recording channel count: {recording.get_num_channels()}")
print(f"- Recording sampling frequency: {recording.get_sampling_frequency()}")

# Step 9: Inspect library versions
try:
    import subprocess
    print("\nLibrary Versions:")
    spikeinterface_version = subprocess.check_output(["pip", "show", "spikeinterface"]).decode()
    probeinterface_version = subprocess.check_output(["pip", "show", "probeinterface"]).decode()
    print(spikeinterface_version)
    print(probeinterface_version)
except Exception as e:
    print(f"Error checking library versions: {e}")
