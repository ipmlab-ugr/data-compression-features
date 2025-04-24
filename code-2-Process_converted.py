import numpy as np
import pandas as pd

# Assuming 10 rows, 114 columns, and 73 iterations as per the original structure
modesave = np.zeros((10, 114, 73))

# Loop through the files and process the mode shapes
for i in range(1, 74):
    file_name = f'modeshape_{i}.csv'
    mode = pd.read_csv(file_name, header=None).values  # Load CSV file as NumPy array

    if i == 1:
        mode_ref = mode[1:, :]  # Exclude the first row for reference mode

    mode_1 = mode[1:, :]  # Exclude the first row for processing

    for j in range(10):
        i_moderef = mode_ref[j, :].reshape(-1, 1)  # Reshape for matrix multiplication
        i_mode1 = mode_1[j, :].reshape(-1, 1)

        A = np.linalg.inv(i_mode1.T @ i_moderef) @ (i_moderef.T @ i_moderef)  # Matrix operation
        # Uncomment below if needed to adjust mode_1 by A
        # mode_1[j, :] = mode_1[j, :] * A

    modesave[:, :, i - 1] = mode_1  # Save processed mode shape

# Save processed data into new CSV files with clean formatting
for i in range(1, 74):
    file_name = f'modeshape_b{i}.csv'
    
    # Prepare the DataFrame with clean headers
    df = pd.DataFrame(modesave[:, :, i - 1])
    
    # Generate column names like Mode1, Mode2, ..., Mode114
    df.columns = [f'Mode{col + 1}' for col in range(df.shape[1])]
    
    # Save to CSV, clean and organized, each mode in separate columns
    df.to_csv(file_name, index=False, float_format='%.6f')

# Uncomment the following lines to visualize the data (plotting)
# import matplotlib.pyplot as plt
# for i in range(10):
#     toplot = modesave[i, :, :].reshape(114, 73)
#     plt.subplot(2, 5, i + 1)
#     plt.plot(toplot)
# plt.show()
