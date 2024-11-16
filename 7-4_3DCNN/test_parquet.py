import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_parquet('0-0sOa1kk5c.parquet')
print(df)
print(df.columns)
# Plot the histogram for the "clip_num" column
plt.hist(df['clip_num'], bins=30, edgecolor='black')  # Adjust 'bins' as necessary
plt.title('Histogram of clip_num')
plt.xlabel('clip_num')
plt.ylabel('Frequency')
plt.show()
