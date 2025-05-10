import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.metrics import dtw_path
from tslearn.barycenters import (
    euclidean_barycenter,
    dtw_barycenter_averaging,
    dtw_barycenter_averaging_subgradient,
    softdtw_barycenter
)

# Load EEG Data
df = pd.read_csv('被试2/0322-2-2-1_output_EEG_power_head_with_marker_3s.csv')

# Define parameters
n = 24  # Number of time steps per sequence
feature_column = 'POW.AF4.Theta'

# Group data into sequences (original trial sequences, not averaged)
grouped = [df[i:i+n] for i in range(0, len(df), n)]

# Convert to NumPy arrays
X = np.array([group[feature_column].values for group in grouped]).reshape(-1, n, 1)
y = np.array([group['marker_value'].iloc[0] for group in grouped])

# Scale time series data
scaler = TimeSeriesScalerMeanVariance(mu=0., std=1.)  
X_scaled = scaler.fit_transform(X)

# Select two original trial sequences for DTW visualization
idx1, idx2 = 0, 1  # Manually pick first two sequences (or modify for random selection)
seq1 = X_scaled[idx1].ravel()
seq2 = X_scaled[idx2].ravel()

# Compute DTW alignment path
dtw_alignment, sim_dtw = dtw_path(seq1, seq2)

# Plot DTW Matching
plt.figure(figsize=(10, 6))
plt.plot(seq1, "b-", label='First EEG Trial')
plt.plot(seq2, "g-", label='Second EEG Trial')

# Draw alignment paths
for pos1, pos2 in dtw_alignment:
    plt.plot([pos1, pos2], [seq1[pos1], seq2[pos2]], color='orange')

plt.legend()
plt.title("Time Series Matching with DTW (EEG Data)")
plt.xlabel("Time Steps")
plt.ylabel("Standardized EEG Power")
plt.grid(True)


# save the image
plt.savefig("dtw_alignment.png", dpi=600, bbox_inches='tight')
plt.show()

# Plot the cost matrix
# Select two original trial sequences for DTW visualization
idx1, idx2 = 2, 3  # Manually pick first two sequences (or modify for random selection)
seq1 = X_scaled[idx1].ravel().reshape(-1, 1)  # Ensure column vector format
seq2 = X_scaled[idx2].ravel().reshape(-1, 1)

# Compute DTW alignment path and cost matrix
dtw_alignment, sim_dtw = dtw_path(seq1, seq2)
cost_matrix = cdist(seq1, seq2)  # Compute pairwise distances between time steps

# Define plot structure
plt.figure(figsize=(8, 8))

# Define positions for subplots
left, bottom = 0.01, 0.1
w_ts = h_ts = 0.2
left_h = left + w_ts + 0.02
width = height = 0.65
bottom_h = bottom + height + 0.02

# Create subplots for visualization
ax_gram = plt.axes([left_h, bottom, width, height])  # DTW cost matrix
ax_s_x = plt.axes([left_h, bottom_h, width, h_ts])  # Second time series
ax_s_y = plt.axes([left, bottom, w_ts, height])  # First time series

# Plot DTW Cost Matrix with alignment path
ax_gram.imshow(cost_matrix, origin='lower', cmap='viridis')
ax_gram.axis("off")
ax_gram.autoscale(False)
ax_gram.plot([j for (i, j) in dtw_alignment], [i for (i, j) in dtw_alignment], "w-", linewidth=2.5)

# Plot Second EEG Trial Time Series (horizontal)
ax_s_x.plot(np.arange(n), seq2, "b-", linewidth=2)
ax_s_x.axis("off")
ax_s_x.set_xlim((0, n - 1))

# Plot First EEG Trial Time Series (vertical)
ax_s_y.plot(-seq1, np.arange(n), "b-", linewidth=2)
ax_s_y.axis("off")
ax_s_y.set_ylim((0, n - 1))

plt.tight_layout()
plt.savefig("cost_matrix.png", dpi=600, bbox_inches='tight')
plt.show()

unique_markers = [1,2]
# Define function to plot barycenters for each marker
def plot_helper(barycenter, title, sequences):
    """Helper function to plot barycenters with EEG sequences."""
    for series in sequences:
        plt.plot(series.ravel(), "k-", alpha=.2)  # Plot original sequences
    plt.plot(barycenter.ravel(), "r-", linewidth=2)  # Plot computed barycenter
    plt.title(title)

# Loop through each EEG marker
for marker in unique_markers:
    print(f"Processing EEG marker: {marker}")
    
    # Select EEG sequences belonging to this marker
    sequences = X_scaled[y == marker]

    if len(sequences) < 2:
        print(f"Skipping marker {marker} (not enough sequences for barycenter computation).")
        continue

    # Compute different barycenters
    euclidean_bc = euclidean_barycenter(sequences)
    dba_bc = dtw_barycenter_averaging(sequences, max_iter=50, tol=1e-3)
    dba_subgrad_bc = dtw_barycenter_averaging_subgradient(sequences, max_iter=50, tol=1e-3)
    soft_dtw_bc = softdtw_barycenter(sequences, gamma=1., max_iter=50, tol=1e-3)

    # Plot results for this marker
    plt.figure(figsize=(8, 10))
    
    ax1 = plt.subplot(4, 1, 1)
    plot_helper(euclidean_bc, f"Euclidean Barycenter (Marker {marker})", sequences)

    plt.subplot(4, 1, 2, sharex=ax1)
    plot_helper(dba_bc, f"DBA Barycenter (Marker {marker})", sequences)

    plt.subplot(4, 1, 3, sharex=ax1)
    plot_helper(dba_subgrad_bc, f"Subgradient DBA Barycenter (Marker {marker})", sequences)

    plt.subplot(4, 1, 4, sharex=ax1)
    plot_helper(soft_dtw_bc, f"Soft-DTW Barycenter (Marker {marker})", sequences)

    # Set limits for better readability
    ax1.set_xlim([0, n])

    # Show and save plot
    plt.tight_layout()
    plt.savefig(f"barycenter_marker_{marker}.png", dpi=300, bbox_inches='tight')
    plt.show()
