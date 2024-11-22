import numpy as np
import matplotlib.pyplot as plt


# Load NumPy file
def load_numpy_data(file_path):
    try:
        data = np.load(file_path, allow_pickle=True)  # Load the numpy file
        print(f"NumPy file '{file_path}' loaded successfully.")
        return data
    except Exception as e:
        print(f"Error loading NumPy file '{file_path}': {e}")
        return None


# Plot Amplitude vs. Sample Index with handling for missing data
def plot_amplitude_vs_sample_index(data_list, mb_ids):
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    axs = axs.ravel()  # Flatten the 2D array of axes for easy iteration

    for i, data in enumerate(data_list):
        if data is not None:
            sample_index = data['sample_index']
            amplitude = data['amplitude']

            axs[i].plot(sample_index, amplitude, marker='o', linestyle='-', color='b')
            axs[i].set_title(f'Amplitude vs. Sample Index (MB{mb_ids[i]:02d})')
        else:
            axs[i].set_title(f"MB{mb_ids[i]:02d} - No Data Available")

        axs[i].set_xlabel('Sample Index')
        axs[i].set_ylabel('Amplitude')
        axs[i].grid(True)

    plt.tight_layout()
    plt.show()


# Plot Amplitude vs. Channel Index with handling for missing data
def plot_amplitude_vs_channel_index(data_list, mb_ids):
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    axs = axs.ravel()  # Flatten the 2D array of axes for easy iteration

    for i, data in enumerate(data_list):
        if data is not None:
            channel_index = data['channel_index']
            amplitude = data['amplitude']

            axs[i].scatter(channel_index, amplitude, color='r')
            axs[i].set_title(f'Amplitude vs. Channel Index (MB{mb_ids[i]:02d})')
        else:
            axs[i].set_title(f"MB{mb_ids[i]:02d} - No Data Available")

        axs[i].set_xlabel('Channel Index')
        axs[i].set_ylabel('Amplitude')
        axs[i].grid(True)

    plt.tight_layout()
    plt.show()


# Process multiple files by iterating over MBXX paths and handling missing data
def process_multiple_files(start_mb, end_mb):
    # Updated base path for the new set of files
    base_path = r"C:\Users\ivank\OneDrive\Documents\GitHub\HelloWorld\Learning\.idea\inspectionProfiles\Task 2\utah_organoids_output\MB{mb_id:02d}\Control\30\spykingcircus2\spike_sorting\sorter_output\sorting\spikes.npy"

    data_list = []
    mb_ids = []

    # Load all the data first
    for mb_id in range(start_mb, end_mb + 1):
        file_path = base_path.format(mb_id=mb_id)
        data = load_numpy_data(file_path)
        data_list.append(data)  # Add None if data is missing
        mb_ids.append(mb_id)

    # Plot all the data in 2x2 grid
    plot_amplitude_vs_sample_index(data_list, mb_ids)
    plot_amplitude_vs_channel_index(data_list, mb_ids)


# Example of how to use this function
if __name__ == "__main__":
    # Process files from MB01 to MB04 for the new set of documents
    process_multiple_files(start_mb=1, end_mb=4)
