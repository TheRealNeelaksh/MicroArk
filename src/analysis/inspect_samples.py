import pandas as pd

try:
    df = pd.read_csv('data/metalog_raw.csv')
    cols = ['host_scientific_name', 'host_common_name', 'geographic_location', 'location', 'collection_date']
    print(df[cols].head(10))
except Exception as e:
    print(f"Error: {e}")
