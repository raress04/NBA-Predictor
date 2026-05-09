import pandas as pd
df = pd.read_parquet('advanced_model/database/feature_matrix.parquet')
print(df['game_date'].min(), df['game_date'].max())
print(df['category'].value_counts())
print(df.columns.tolist())
