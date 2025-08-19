import pandas as pd
def clean_data(df):
    df = df.dropna(subset=['id', 'name'])
    df = df[df['age'].apply(lambda x: pd.notnull(x) and x >= 0 if isinstance(x, (int, float)) else False)]
    df = df[df['email'].apply(lambda x: isinstance(x, str) and '@' in x and '.' in x.split('@')[-1])]
    if 'age' in df.columns:
        mean_age = df['age'].dropna().mean()
        df['age'] = df['age'].fillna(mean_age)
    return df
# Usage: df = clean_data(df)
