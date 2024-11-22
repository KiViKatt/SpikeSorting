import pickle
import pandas as pd


# Load data from a pickle file
def load_pickle_data(file_path):
    try:
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
        print("Pickle file loaded successfully.")
        return data
    except Exception as e:
        print(f"Error loading pickle file: {e}")
        return None


# Helper function to recursively analyze and print all dictionary data
def analyze_nested_dict(d, level=0):
    indent = "  " * level
    for key, value in d.items():
        if isinstance(value, dict):
            print(f"{indent}Key: {key} -> Nested dictionary:")
            analyze_nested_dict(value, level + 1)
        elif isinstance(value, list):
            print(f"{indent}Key: {key} -> List with {len(value)} elements")
            for i, item in enumerate(value[:5]):  # Display first 5 elements of a list for brevity
                print(f"{indent}  List item {i}: {item}")
            if len(value) > 5:
                print(f"{indent}  ... (and {len(value) - 5} more items)")
        else:
            print(f"{indent}Key: {key} -> Value: {value}")


# Analyze and print all data based on its structure
def analyze_data(data):
    if isinstance(data, dict):
        print("Data is a dictionary. Here is all the data:")
        analyze_nested_dict(data)

    elif isinstance(data, list):
        print("Data is a list with length:", len(data))
        for i, element in enumerate(data[:5]):  # Display first 5 elements of a list for brevity
            print(f"Element {i}: {element}")
        if len(data) > 5:
            print(f"... (and {len(data) - 5} more elements)")

    else:
        print(f"Data type is {type(data)}. Here is the data:")
        print(data)


# Example of how to use this function
if __name__ == "__main__":
    # Replace with your pickle file path
    pickle_file_path = r"C:\Users\ivank\OneDrive\Documents\GitHub\HelloWorld\Learning\.idea\inspectionProfiles\Task 2\utah_organoids_output\MB01\Control\26\tridesclous2\recording\si_recording.pkl"

    # Load and analyze the pickle data
    data = load_pickle_data(pickle_file_path)
    if data:
        analyze_data(data)
