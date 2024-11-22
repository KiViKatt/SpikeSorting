import numpy as np


# Load NumPy file and inspect its structure
def load_numpy_data(file_path):
    try:
        data = np.load(file_path, allow_pickle=True)  # Load the numpy file
        print(f"\nNumPy file '{file_path}' loaded successfully.")

        # Print the structure of the loaded data
        if isinstance(data, np.ndarray):
            print(f"Data shape: {data.shape}")
            print(f"Data dtype: {data.dtype}")
            print("Data sample (first 5 entries):")
            print(data[:5])  # Print the first 5 entries to inspect structure
        else:
            print("The data is not in an expected NumPy array format.")
        return data
    except Exception as e:
        print(f"Error loading NumPy file '{file_path}': {e}")
        return None


# Process multiple files to check the structure of each file
def check_files_structure(start_mb, end_mb):
    base_path = r"C:\Users\ivank\OneDrive\Documents\GitHub\HelloWorld\Learning\.idea\inspectionProfiles\Task 2\utah_organoids_output\MB{mb_id:02d}\Control\30\spykingcircus2\spike_sorting\sorter_output\sorting\spikes.npy"

    # Load and check all the data
    for mb_id in range(start_mb, end_mb + 1):
        file_path = base_path.format(mb_id=mb_id)
        data = load_numpy_data(file_path)


# Example of how to use this function
if __name__ == "__main__":
    # Check files from MB01 to MB04 to inspect their structure
    check_files_structure(start_mb=1, end_mb=4)
