import pandas as pd
from datetime import datetime

def clean_data():
    df = pd.read_csv('input.csv')

    # Clean date column
    df['date'] = df['date'].astype(str).str.replace(r'[^0-9]', '', regex=True)
    df['date'] = pd.to_datetime(df['date'], format='%Y%m%d', errors='coerce').dt.strftime('%Y%m%d')
    df['date'] = df['date'].fillna('00000000')

    # Clean site column
    df['site'] = df['site'].astype(str).str.replace(r'[^0-9]', '', regex=True)
    df['site'] = df['site'].str.zfill(4)

    # Clean article column
    df['article'] = df['article'].astype(str).str.replace(r'[^0-9]', '', regex=True)
    df['article'] = df['article'].str.zfill(18)

    # Clean stock_units column
    df['stock_units'] = pd.to_numeric(df['stock_units'], errors='coerce').fillna(0).astype(int)

    # Clean retail_value and cost_value columns
    df['retail_value'] = pd.to_numeric(df['retail_value'], errors='coerce').fillna(0.0).round(2)
    df['cost_value'] = pd.to_numeric(df['cost_value'], errors='coerce').fillna(0.0).round(2)

    # Clean currency column
    df['currency'] = df['currency'].astype(str).str.upper().str.replace(r'[^A-Z]', '', regex=True)
    df['currency'] = df['currency'].apply(lambda x: 'GBP' if x == 'GBP' else 'GBP')

    # Drop duplicates based on primary key
    df = df.drop_duplicates(subset=['date', 'site', 'article'])

    # Save cleaned data
    df.to_csv('clean_data.csv', index=False)

clean_data()