import numpy as np
import spikeinterface.full as si
from probeinterface import Probe, ProbeGroup

# Create a synthetic recording
num_channels = 32
sampling_frequency = 20000
duration = 10.0
recording = si.NumpyRecording(
    [np.random.randn(int(sampling_frequency * duration), num_channels)],
    sampling_frequency
)

# Create a Probe
probe = Probe(ndim=2, si_units='um')
probe.set_contacts(
    positions=np.zeros((num_channels, 2)),
    shapes='circle',
    shape_params={'radius': 5},
    contact_ids=np.arange(num_channels)
)
probe.set_device_channel_indices(np.arange(num_channels))

# Attach the Probe
print("\nAttaching Probe...")
BFix1 = recording.set_probe(probe)
try:
    attached_probe = BFix1
    print(f"Attached Probe: {attached_probe}")
except Exception as e:
    print(f"Error accessing attached Probe: {e}")

# Attach a ProbeGroup
print("\nAttaching ProbeGroup...")
probegroup = ProbeGroup()
probegroup.add_probe(probe)
BFix2 = recording.set_probegroup(probegroup)
try:
    attached_probegroup = BFix2
    print(f"Attached ProbeGroup: {attached_probegroup}")
except Exception as e:
    print(f"Error accessing attached ProbeGroup: {e}")

# Print Recording Summary
print("\nRecording Summary:")
print(recording)
