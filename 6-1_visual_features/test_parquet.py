import pandas as pd
df = pd.read_parquet('test_1024.parquet')
print(df)
print(df.columns)
