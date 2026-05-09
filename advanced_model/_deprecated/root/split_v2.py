import pandas as pd

df = pd.read_parquet('advanced_model/database/feature_matrix_v2.parquet')
df['game_date'] = pd.to_datetime(df['game_date'])
df = df.dropna(subset=['model_bias'])

train = df[df['game_date'] <  pd.Timestamp('2026-02-01')]
val   = df[(df['game_date'] >= pd.Timestamp('2026-02-01')) & (df['game_date'] < pd.Timestamp('2026-03-16'))]
test  = df[df['game_date'] >= pd.Timestamp('2026-03-16')]

train.to_parquet('advanced_model/database/train_v2.parquet', index=False)
val.to_parquet('advanced_model/database/val_v2.parquet', index=False)
test.to_parquet('advanced_model/database/test_v2.parquet', index=False)

for name, s in [('train', train), ('val', val), ('test', test)]:
    print(f"{name}: {len(s):,} rows  [{s['game_date'].min().date()} -> {s['game_date'].max().date()}]")
    print(s['category'].value_counts().to_string())
    print()
