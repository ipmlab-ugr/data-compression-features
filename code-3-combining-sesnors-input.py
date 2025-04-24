import pandas as pd
import os

# Define the directory where the individual time series set CSV files are stored
input_dir = 'C:/Dr-Elisa-Project/diiferent number of sensors/downsampled_data-114-mainonly-1-only-std_csv'  # Update with the correct path if needed

# Define the list of set numbers you want to combine
set_numbers = [1,3, 13, 35,36,39,40,41,42,43,46,48,50,53,56,57,60,66,69,71,73,74]  
#set_numbers = [3]  ## Update this list to the specific sets you want to combine
#set_numbers = list(range(1, 115))  # Generates a list from 1 to 114

# Initialize an empty list to store the DataFrames
dfs = []

# Iterate through the set numbers, read the CSV files, and store them in the list
for set_number in set_numbers:
    file_path = os.path.join(input_dir, f'time_series_set_{set_number}_features.csv')
    
    if os.path.exists(file_path):
        # Read the CSV file for the current set number
        df = pd.read_csv(file_path)
        dfs.append(df)  # Append the DataFrame to the list
        print(f"Loaded file: {file_path}")  # Debugging output
    else:
        print(f"File not found: {file_path}")

# Check if we have any DataFrames to combine
if dfs:
    # Concatenate the DataFrames vertically (row-wise)
    combined_df = pd.concat(dfs, axis=1, ignore_index=True)
    
    # Define the output directory to save the combined file
    output_dir = 'combined_data-114_new-features-csv'
    os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist

    # Define the output file path in the new directory
    output_file = os.path.join(output_dir, 'combined_time_series_features-selected-approach-3-only-1-new-features-onlystd-114gg.csv')
    
    # Save the combined DataFrame to the new directory
    combined_df.to_csv(output_file, index=False)
    print(f"Combined features saved to {output_file}")
else:
    print("No files were combined. Please check the input directory and set numbers.")
