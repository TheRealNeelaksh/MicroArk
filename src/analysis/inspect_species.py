import pandas as pd

try:
    df = pd.read_csv('data/metalog_raw.csv')
    print("Non-null counts:")
    print(df[['host_scientific_name', 'host_common_name', 'host', 'host_tax_id']].count())
    
    print("\nUnique 'host' values (first 10):")
    print(df['host'].dropna().unique()[:10])

    print("\nUnique 'host_scientific_name' values (first 10):")
    print(df['host_scientific_name'].dropna().unique()[:10])

except Exception as e:
    print(f"Error: {e}")
