import pandas as pd

try:
    df = pd.read_csv('2024-5K-supreme-court-decisions-analyzed.csv', nrows=5)
    print("Columns:", df.columns.tolist())
    print("First row:", df.iloc[0].to_dict())
except Exception as e:
    print(e)
