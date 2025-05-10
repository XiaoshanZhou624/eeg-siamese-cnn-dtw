import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyvistaqt as pv
from mne import vertex_to_mni
from pyvistaqt import BackgroundPlotter
import time
from mne.time_frequency import tfr_multitaper
from mne.time_frequency import tfr_morlet
from mne_connectivity import spectral_connectivity_epochs, spectral_connectivity_time
from mne_connectivity.viz import plot_sensors_connectivity
from mne.viz import plot_compare_evokeds, plot_evoked_topo

# Define Emotiv Epoc+ EEG channel names
eeg_channels = ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"]

# Approximate 10-20 system locations for Emotiv Epoc+ (X, Y, Z coordinates)
channel_positions = {
    "AF3": [-0.03,  0.08, 0],
    "F7":  [-0.07,  0.06, 0],
    "F3":  [-0.04,  0.05, 0],
    "FC5": [-0.06,  0.03, 0],
    "T7":  [-0.08,  0.00, 0],
    "P7":  [-0.07, -0.04, 0],
    "O1":  [-0.03, -0.06, 0],
    "O2":  [ 0.03, -0.06, 0],
    "P8":  [ 0.07, -0.04, 0],
    "T8":  [ 0.08,  0.00, 0],
    "FC6": [ 0.06,  0.03, 0],
    "F4":  [ 0.04,  0.05, 0],
    "F8":  [ 0.07,  0.06, 0],
    "AF4": [ 0.03,  0.08, 0]
}

# Convert positions to MNE DigMontage format
montage = mne.channels.make_dig_montage(
    ch_pos=channel_positions,  # Channel positions
    coord_frame="head"  # Set coordinate frame to 'head'
)

# Load the EDF file
raw = mne.io.read_raw_edf("0322-2-2-1_EPOCX.edf", preload=True)

# Keep only EEG channels
raw.pick(eeg_channels)

# Apply the custom montage
raw.set_montage(montage, match_case=False)

# Plot sensor locations
raw.plot_sensors(kind='topomap', show_names=True)

fig = raw.plot(duration=10, n_channels=14, scalings="auto")

fig = raw.plot_psd(fmax=60)
fig.savefig("raw_psd.png", dpi=600)


# Load the event marker file (replace with your actual filename)
event_file = "0322-2-2-1_EPOCX_intervalMarker.csv"  
df_events = pd.read_csv(event_file, delimiter=",")  # Use "\t" if tab-separated

# Display first few rows
print("event head")
print(df_events.head())

# Get sampling frequency from EEG data
sfreq = raw.info["sfreq"]  # Ensure raw EEG data is loaded first

# Convert latencies (in seconds) to sample indices
sample_indices = (df_events["latency"] * sfreq).astype(int).values

# Extract event IDs from "marker_value" column
event_ids = df_events["marker_value"].values

# Construct MNE-compatible event array
events = np.column_stack((sample_indices, np.zeros(len(sample_indices), dtype=int), event_ids))

# Verify event array
print(events[:5])

# Create an event dictionary for MNE
event_dict = {f"Stimuli_{i}": i for i in np.unique(event_ids)}

# print("Event Dictionary:", event_dict)

# # Define time window for epochs (-0.2s before, 1.0s after event)
tmin, tmax = -0.2, 5

# # Create epochs using the detected events
epochs = mne.Epochs(raw, events, event_id=event_dict, tmin=tmin, tmax=tmax, baseline=(None, 0), preload=True) # baseline=(None, -2.0)

# calculate the power spectral density (PSD) of the EEG data and plot the topo map
fig = epochs["Stimuli_2"].compute_psd().plot(picks="eeg", exclude="bads", amplitude=True)
fig.savefig("psd_topo_map_1_2.png", dpi=600)
# Plot the topo map of psd
bands = {"Theta (4–8 Hz)": (4, 8), "Alpha (8–12 Hz)": (8, 12), "Low Beta (12–16 Hz)": (12, 16), "High Beta (16-25 Hz)": (16, 25), "Gamma (25–45 Hz)": (25, 45)}
spectrum = epochs["Stimuli_1"].compute_psd()
fig = spectrum.plot_topomap(bands=bands)
fig.savefig("psd_topo_2_1.png", dpi=600)

# plot the trf
# Define frequency range and number of cycles per frequency
freqs = np.arange(2, 40, 2)  # Frequencies from 2Hz to 40Hz in steps of 2Hz
n_cycles = freqs / 2  # Number of cycles per frequency (default: freq/2)

# Compute TFR using Morlet wavelets/Multitaper wavelets
tfr = tfr_morlet(epochs["Stimuli_1"], freqs=freqs, n_cycles=n_cycles, use_fft=True, return_itc=False, picks="eeg")
# tfr = tfr_multitaper(epochs["Stimuli_1"], freqs=freqs, n_cycles=n_cycles, time_bandwidth=4.0, return_itc=False, picks="eeg")
# Plot TFR for a specific electrode (e.g., 'Cz')
tfr.plot(picks="F4", baseline=(-0.2, 0), mode="logratio", title="TFR - F4 (Morlet Wavelet)")

tfr.plot_joint(tmin=-0.2, tmax=3, mode="mean", timefreqs=[(0.75, 10), (2, 22)])

# Connectivity Analysis
tmin = 0.0  # exclude the baseline period for connectivity estimation
Freq_Bands = {"High Beta": [16.0, 25.0]}  # frequency of interest
n_freq_bands = len(Freq_Bands)
min_freq = np.min(list(Freq_Bands.values()))
max_freq = np.max(list(Freq_Bands.values()))

# Prepare the freq points
freqs = np.linspace(min_freq, max_freq, int((max_freq - min_freq) * 4 + 1))

fmin = tuple([list(Freq_Bands.values())[f][0] for f in range(len(Freq_Bands))])
fmax = tuple([list(Freq_Bands.values())[f][1] for f in range(len(Freq_Bands))])

# We specify the connectivity measurements
connectivity_methods = ["wpli"]
n_con_methods = len(connectivity_methods)

# Compute connectivity over trials
con_epochs = spectral_connectivity_epochs(
    epochs["Stimuli_2"],
    method=connectivity_methods,
    sfreq=sfreq,
    mode="cwt_morlet",
    cwt_freqs=freqs,
    fmin=fmin,
    fmax=fmax,
    faverage=True,
    tmin=tmin,
    cwt_n_cycles=4,
)

# Plot the global connectivity over time
n_channels = epochs.info["nchan"]  # get number of channels
times = epochs.times[epochs.times >= tmin]  # get the timepoints
n_connections = (n_channels * n_channels - n_channels) / 2

# Get global avg connectivity over all connections
con_epochs_raveled_array = con_epochs.get_data(output="raveled")
global_con_epochs = np.sum(con_epochs_raveled_array, axis=0) / n_connections

# Since there is only one freq band, we choose the first dimension
global_con_epochs = global_con_epochs[0]

fig = plt.figure()
plt.plot(times, global_con_epochs)
plt.xlabel("Time (s)")
plt.ylabel("Global high beta wPLI over trials")
# fig.savefig("global_alpha_wPLI.png", dpi=600)

# Get the timepoint with highest global connectivity right after stimulus
t_con_max = np.argmax(global_con_epochs[times <= 5])
t_con_min = np.argmin(global_con_epochs[times <= 5])
print(f"Global high beta wPLI max peaks {times[t_con_max]:.3f}s after stimulus")
print(f"Global high beta wPLI min peaks {times[t_con_min]:.3f}s after stimulus")

# Plot the connectivity matrix at the timepoint with highest global wPLI
con_epochs_matrix = con_epochs.get_data(output="dense")[:, :, 0, t_con_max]
con_epochs_matrix = con_epochs.get_data(output="dense")[:, :, 0, 3]

fig = plt.figure()
im = plt.imshow(con_epochs_matrix)
fig.colorbar(im, label="Connectivity")
plt.ylabel("Channels")
plt.xlabel("Channels")
plt.show()
fig.savefig("connectivity_matrix_theta_0-812s.png", dpi=600)

# Visualize top 20 connections in 3D
connectivity = plot_sensors_connectivity(epochs.info, con_epochs_matrix)
connectivity.plotter.screenshot("connectivity_3D_highBeta_3s_2.png")


# Compute Fourier coefficients for the epochs (returns an EpochsSpectrum object)
# (storing Fourier coefficients in EpochsSpectrum objects requires MNE >= 1.8)
tmin = 0.0  # exclude the baseline period
spectrum = epochs.compute_psd(method="multitaper", tmin=tmin, output="complex")

# Compute connectivity for the frequency band containing the evoked response
# (passing EpochsSpectrum objects as data requires MNE-Connectivity >= 0.8)
fmin, fmax = 4.0, 9.0
con = spectral_connectivity_epochs(
    data=spectrum, method="pli", fmin=fmin, fmax=fmax, faverage=True, n_jobs=1
)

# Now, visualize the connectivity in 3D:
plot_sensors_connectivity(epochs.info, con.get_data(output="dense")[:, :, 0])

# Average across trials to create an Evoked object
evoked = epochs["Stimuli_1"].average()

# Plot topographic maps of EEG activity at different time points
times = np.linspace(-2.0, 5.0, 10)  # Select 5 time points between -0.2s to 1.0s
evoked.plot_topomap(times=times, ch_type="eeg", size=3)

# Joint plots combine butterfly plots with scalp topographies
fig = evoked.plot_joint(times=times, title="Joint EEG Visualization", picks="eeg")

# Plot Single-Trial EEG Traces:
epochs.plot(n_epochs=5, n_channels=14, scalings="auto")

# Compare Different Stimuli:
evoked_1 = epochs["Stimuli_1"].average()
evoked_2 = epochs["Stimuli_2"].average()
mne.viz.plot_compare_evokeds([evoked_1, evoked_2], picks="eeg")

plot_evoked_topo([evoked_1, evoked_2], title="Evoked Response Comparison", background_color="w")
evoked_1.plot_joint(times=times, title="Stimulus 1 Response", picks="eeg")

# Plot 3D Field Maps
# Ensure PyVistaQt backend is set for 3D rendering
mne.viz.set_3d_backend("pyvistaqt")

# Compute the field maps for EEG
field_map = mne.make_field_map(evoked, trans=trans, subject="fsaverage", subjects_dir=subjects_dir)

mne.viz.set_3d_backend("pyvistaqt")  # Ensure correct 3D rendering backend

# Plot the field maps on the head
brain = evoked.plot_field(field_map, time=0.6)

brain.plotter.screenshot("field_map_3D_600.png")

# Define multiple time points to visualize
time_points = [0.1, 0.2, 0.3, 0.4, 0.8, 0.9]

# Loop through the time points and plot multiple field maps
for t in time_points:
    brain = evoked.plot_field(field_map, time=t)
    brain.plotter.show()  # Keep the PyVista window open
    time.sleep(2)  # Pause for 2 seconds before drawing the next map
# Keep the PyVista visualization open
brain.plotter.show()

# # Source Localization
# Use a standard MRI template
subjects_dir = str(mne.datasets.sample.data_path() / "subjects")
mne.datasets.fetch_fsaverage(verbose=True)
trans = "fsaverage"  # Default transformation

# Generate BEM surfaces
mne.bem.make_watershed_bem(subject="fsaverage", subjects_dir=subjects_dir, overwrite=True)

src = mne.setup_source_space("fsaverage", subjects_dir=subjects_dir, add_dist=False)
bem = mne.make_bem_model("fsaverage", subjects_dir=subjects_dir)
# Convert BEM surfaces to a Conductor Model
bem = mne.make_bem_solution(bem)  # This fixes the issue
fwd = mne.make_forward_solution(evoked.info, trans=trans, src=src, bem=bem)

# Use Minimum Norm Estimate (MNE) to compute source activity
# Estimate noise covariance from the baseline period (-0.2s to 0s)
noise_cov = mne.compute_covariance(epochs, tmin=-0.2, tmax=0, method="auto")
# Create the inverse operator using the computed noise covariance
evoked.set_eeg_reference('average', projection=True)
inverse_operator = mne.minimum_norm.make_inverse_operator(evoked.info, fwd, noise_cov=noise_cov)
# Apply EEG average reference projection
# evoked.set_eeg_reference(ref_channels="average", projection=True)
stc = mne.minimum_norm.apply_inverse(evoked, inverse_operator, lambda2=1.0 / 9.0, method="MNE")

# # Plot the 3D brain activation
# Define custom clim values (lower, middle, upper limit)
clim = dict(kind='value', lims=[1e-11, 1e-10, 2e-10])

brain = stc.plot(subject="fsaverage", subjects_dir=subjects_dir, initial_time=0.8, hemi="both", clim=clim, time_viewer=True)

# Find the peak source location for both hemispheres separately
peak_vertex_lh, peak_time_lh = stc.get_peak(hemi="lh", tmin=0, tmax=1.0, mode="abs")
peak_vertex_rh, peak_time_rh = stc.get_peak(hemi="rh", tmin=0, tmax=1.0, mode="abs")

print(f"Peak vertex (LH): {peak_vertex_lh}, Peak time: {peak_time_lh}s")
print(f"Peak vertex (RH): {peak_vertex_rh}, Peak time: {peak_time_rh}s")

# Convert left hemisphere peak vertex to MNI coordinates
mni_coords_lh = vertex_to_mni(peak_vertex_lh, hemis=0, subject="fsaverage", subjects_dir=subjects_dir)
# Convert right hemisphere peak vertex to MNI coordinates
mni_coords_rh = vertex_to_mni(peak_vertex_rh, hemis=1, subject="fsaverage", subjects_dir=subjects_dir)

print(f"Left Hemisphere MNI Coordinates: {mni_coords_lh}")
print(f"Right Hemisphere MNI Coordinates: {mni_coords_rh}")

# Specifity the time point
# Define the time point you want to extract
fixed_time_point = 0.8  # Specify the time in seconds

# Find the index in stc.times that is closest to fixed_time_point
time_idx = np.argmin(np.abs(stc.times - fixed_time_point))

# Get the vertex with the maximum absolute activation at this time for both hemispheres
lh_vertex = np.argmax(np.abs(stc.lh_data[:, time_idx]))  # Left hemisphere
rh_vertex = np.argmax(np.abs(stc.rh_data[:, time_idx]))  # Right hemisphere

# Convert left hemisphere vertex to MNI coordinates
mni_coords_lh = vertex_to_mni(lh_vertex, hemis=0, subject="fsaverage", subjects_dir=subjects_dir)

# Convert right hemisphere vertex to MNI coordinates
mni_coords_rh = vertex_to_mni(rh_vertex, hemis=1, subject="fsaverage", subjects_dir=subjects_dir)

# Print the results
print(f"Selected Time: {fixed_time_point}s")
print(f"Vertex (LH): {lh_vertex}, MNI: {mni_coords_lh}")
print(f"Vertex (RH): {rh_vertex}, MNI: {mni_coords_rh}")

# Keep the PyVista window open
# brain.show_view()  # Ensures the visualization is shown properly

brain.save_image("brain_activity_0.8s_stimulus_1.png")

plt.show()