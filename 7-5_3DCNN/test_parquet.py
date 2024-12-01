import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_parquet("0-1tv3Xf5u4.parquet")
print(df)
print(df.columns)

# Plot the histogram for the "clip_num" column
# plt.hist(df["clip_num"], bins=30, edgecolor="black")  # Adjust 'bins' as necessary
# plt.title("Histogram of clip_num")
# plt.xlabel("clip_num")
# plt.ylabel("Frequency")
# plt.show()
df = df.drop(columns=df.columns[:4])

# Iterate through the rows and reshape the optical flow data (assuming each row has 1024 elements)
for idx, row in df.iterrows():
    # Extract the 1024 elements from the row
    optical_flow_data = row.values  # These should be the remaining columns (1024 in total)

    # Reshape the 1024 data into a 32x32 2D array
    optical_flow_reshaped = optical_flow_data.reshape(32, 32)

    # Display the reshaped optical flow data as an image
    plt.imshow(optical_flow_reshaped, cmap='gray')  # Use grayscale for visualizing optical flow
    plt.title(f'Optical Flow - Row {idx}')
    plt.colorbar()  # Add a color bar for better interpretation of pixel values
    plt.show()
