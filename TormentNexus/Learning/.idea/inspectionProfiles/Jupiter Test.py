import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Data input
data_spykingcircus2 = {
    'Recording': ['MB1', 'MB2', 'MB3', 'MB4', 'MB5', 'MB6', 'MB7', 'MB8', 'E1', 'E2', 'E3', 'E4'],
    'Peaks': [1828, 5034, 1190, 1419, 362, 500, 125, np.nan, 31446, 6405, 14732, 4103],
    'Spikes': [68670, 58293, 45545, 38291, 1499, 1645, 321, np.nan, 17497, 61972, 94871, 79168],
    'Units': [31, 30, 30, 32, 7, 18, 5, np.nan, 33, 39, 21, 32]
}

# Data for tridesclous2

data_tridesclous2 = {
    'Recording': ['MB1', 'MB2', 'MB3', 'MB4', 'MB5', 'MB6', 'MB7', 'MB8', 'E1', 'E2', 'E3', 'E4'],
    'Peaks': [29, 415, 47, 32, 9, 8, 2, np.nan, 1656, 1487, 1894, 316],
    'Clusters': [0, 3, 0, 0, 0, 0, 0, np.nan, 6, 8, 12, 3]
}

# Convert to pandas DataFrames
df_spykingcircus2 = pd.DataFrame(data_spykingcircus2)
df_tridesclous2 = pd.DataFrame(data_tridesclous2)

# Replace 'ERROR' values with NaN
df_spykingcircus2.replace('ERROR', np.nan, inplace=True)
df_tridesclous2.replace('ERROR', np.nan, inplace=True)

# Calculate averages
avg_spykingcircus2_peaks = df_spykingcircus2['Peaks'].mean()
avg_spykingcircus2_spikes = df_spykingcircus2['Spikes'].mean()
avg_spykingcircus2_units = df_spykingcircus2['Units'].mean()

avg_tridesclous2_peaks = df_tridesclous2['Peaks'].mean()
avg_tridesclous2_clusters = df_tridesclous2['Clusters'].mean()

# Calculate how much higher the averages are in Spykingcircus2 vs Tridesclous2
peaks_difference = avg_spykingcircus2_peaks / avg_tridesclous2_peaks
spikes_difference = avg_spykingcircus2_spikes / avg_tridesclous2_clusters
units_difference = avg_spykingcircus2_units / avg_tridesclous2_clusters

print(f"Spykingcircus2 Peaks are {peaks_difference:.2f} times higher on average compared to Tridesclous2 Peaks.")
print(f"Spykingcircus2 Spikes are {spikes_difference:.2f} times higher on average compared to Tridesclous2 Clusters.")
print(f"Spykingcircus2 Units are {units_difference:.2f} times higher on average compared to Tridesclous2 Clusters.")

# Plotting the summary of Spykingcircus2 results
fig, ax1 = plt.subplots(figsize=(10, 6))

ax1.bar(df_spykingcircus2['Recording'], df_spykingcircus2['Peaks'], label='Peaks', alpha=0.7, color='blue')
ax1.bar(df_spykingcircus2['Recording'], df_spykingcircus2['Spikes'], label='Spikes', alpha=0.7, color='green')
ax1.bar(df_spykingcircus2['Recording'], df_spykingcircus2['Units'], label='Units', alpha=0.7, color='red')

ax1.set_xlabel('Recording')
ax1.set_ylabel('Counts')
ax1.set_title('Summary Sorting Results: Spykingcircus2')
ax1.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Plotting the summary of Tridesclous2 results
fig, ax2 = plt.subplots(figsize=(10, 6))

ax2.bar(df_tridesclous2['Recording'], df_tridesclous2['Peaks'], label='Peaks', alpha=0.7, color='blue')
ax2.bar(df_tridesclous2['Recording'], df_tridesclous2['Clusters'], label='Clusters', alpha=0.7, color='orange')

ax2.set_xlabel('Recording')
ax2.set_ylabel('Counts')
ax2.set_title('Summary Sorting Results: Tridesclous2')
ax2.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
