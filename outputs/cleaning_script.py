import pandas as pd
import numpy as np
from datetime import datetime
import json

def clean_domd_data(input_file='input.csv', clean_output='clean_data.csv', unclean_output='unclean_data.json', anomalies_output='anomalies.json'):
    # Load data
    df = pd.read_csv(input_file, dtype=str, keep_default_na=False, na_values=[''])

    # Initialize outputs
    cleaned_rows = []
    unclean_rows = []
    anomalies = []

    # Helper functions
    def validate_date(date_str):
        if not date_str or pd.isna(date_str):
            return np.nan
        date_str = str(date_str).strip()
        if len(date_str) != 8 or not date_str.isdigit():
            anomalies.append({
                'row': idx,
                'column': 'date',
                'issue': f'Invalid date format: {date_str}',
                'expected': 'YYYYMMDD (8 digits)'
            })
            return np.nan
        try:
            datetime.strptime(date_str, '%Y%m%d')
            return date_str
        except ValueError:
            anomalies.append({
                'row': idx,
                'column': 'date',
                'issue': f'Invalid date value: {date_str}',
                'expected': 'Valid calendar date in YYYYMMDD format'
            })
            return np.nan

    def validate_site(site_str):
        if not site_str or pd.isna(site_str):
            return np.nan
        site_str = str(site_str).strip()
        if not site_str.isdigit():
            anomalies.append({
                'row': idx,
                'column': 'site',
                'issue': f'Non-numeric site: {site_str}',
                'expected': '4-digit numeric string'
            })
            return np.nan
        return site_str.zfill(4)

    def validate_article(article_str):
        if not article_str or pd.isna(article_str):
            return np.nan
        article_str = str(article_str).strip()
        if not article_str.isdigit():
            anomalies.append({
                'row': idx,
                'column': 'article',
                'issue': f'Non-numeric article: {article_str}',
                'expected': '18-digit numeric string starting with zeros'
            })
            return np.nan
        return article_str.zfill(18)[-18:]  # Pad or truncate to 18 digits

    def validate_currency(currency_str):
        if not currency_str or pd.isna(currency_str):
            return np.nan
        currency_str = str(currency_str).strip().upper()
        if currency_str != 'GBP':
            anomalies.append({
                'row': idx,
                'column': 'currency',
                'issue': f'Invalid currency: {currency_str}',
                'expected': 'GBP'
            })
            return np.nan
        return currency_str

    def validate_numeric(value, col_name, is_integer=False):
        if not value or pd.isna(value):
            return 0.0 if not is_integer else 0
        value = str(value).strip()
        try:
            num = float(value)
            if num < 0:
                anomalies.append({
                    'row': idx,
                    'column': col_name,
                    'issue': f'Negative value: {value}',
                    'expected': 'Non-negative number'
                })
                return 0.0 if not is_integer else 0
            return int(num) if is_integer else round(num, 2)
        except ValueError:
            anomalies.append({
                'row': idx,
                'column': col_name,
                'issue': f'Non-numeric value: {value}',
                'expected': 'Numeric value'
            })
            return 0.0 if not is_integer else 0

    # Process each row
    for idx, row in df.iterrows():
        original_row = row.to_dict()
        cleaned_row = {}

        # Validate required fields
        date = validate_date(row.get('date'))
        site = validate_site(row.get('site'))
        article = validate_article(row.get('article'))

        # Check if required fields are missing after validation
        if pd.isna(date) or pd.isna(site) or pd.isna(article):
            unclean_rows.append(original_row)
            continue

        cleaned_row['date'] = date
        cleaned_row['site'] = site
        cleaned_row['article'] = article

        # Validate other fields
        cleaned_row['currency'] = validate_currency(row.get('currency'))
        cleaned_row['stock_units'] = validate_numeric(row.get('stock_units'), 'stock_units', is_integer=True)
        cleaned_row['retail_value'] = validate_numeric(row.get('retail_value'), 'retail_value')
        cleaned_row['cost_value'] = validate_numeric(row.get('cost_value'), 'cost_value')

        # Check for missing required fields in cleaned row
        if any(pd.isna(cleaned_row[col]) for col in ['date', 'site', 'article', 'currency']):
            unclean_rows.append(original_row)
        else:
            cleaned_rows.append(cleaned_row)

    # Convert cleaned rows to DataFrame and save
    if cleaned_rows:
        cleaned_df = pd.DataFrame(cleaned_rows)
        # Check for duplicates in primary key
        duplicates = cleaned_df.duplicated(subset=['date', 'site', 'article'], keep=False)
        if duplicates.any():
            dup_rows = cleaned_df[duplicates]
            for _, row in dup_rows.iterrows():
                anomalies.append({
                    'row': row.name,
                    'column': 'primary_key',
                    'issue': f"Duplicate primary key: date={row['date']}, site={row['site']}, article={row['article']}",
                    'expected': 'Unique combination of date, site, and article'
                })
            # Remove duplicates (keep first occurrence)
            cleaned_df = cleaned_df[~duplicates] if not duplicates.all() else cleaned_df.iloc[0:0]

        # Save cleaned data
        cleaned_df.to_csv(clean_output, index=False)

    # Save unclean rows and anomalies
    with open(unclean_output, 'w') as f:
        json.dump(unclean_rows, f, indent=2)

    with open(anomalies_output, 'w') as f:
        json.dump(anomalies, f, indent=2)

    return cleaned_rows, unclean_rows, anomalies

if __name__ == '__main__':
    clean_domd_data()