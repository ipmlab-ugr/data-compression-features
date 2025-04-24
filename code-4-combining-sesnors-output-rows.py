import os
import pandas as pd

# Define the folder containing your CSV files
folder_path = 'C:/Dr-Elisa-Project/mode-shapes'

# List all files in the folder
all_files = os.listdir(folder_path)

# Select files based on their numbering (e.g., model_shape_1, model_shape_2, model_shape_3)
file_numbers_to_select = list(range(1, 74))  # Adjust range if needed
selected_files = [f"modeshape_b{num}.csv" for num in file_numbers_to_select if f"modeshape_b{num}.csv" in all_files]

# Initialize an empty list to store selected rows (second row from each file)
selected_rows = []

# Read each selected file and append the second row to the list
for file in selected_files:
    file_path = os.path.join(folder_path, file)
    
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)
    
    # Select only the second row (row index 1) and append it
    selected_rows.append(df.iloc[9, :])  # Use iloc to select the second row

# Concatenate all the selected rows into a single DataFrame
combined_df = pd.DataFrame(selected_rows)

# Save the combined DataFrame to a new CSV file
output_file = 'C:/Dr-Elisa-Project/mode-shapes/combinedmodeshape/combinedmodeshape-row10.csv'
combined_df.to_csv(output_file, index=False)

# Optionally print the combined DataFrame to verify
print(combined_df)
