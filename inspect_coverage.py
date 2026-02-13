import pandas as pd

try:
    df = pd.read_csv('data/metalog_raw.csv')
    print("Non-null counts:")
    print(df[['location', 'geographic_location', 'collection_date', 'latitude', 'longitude']].count())
    
    print("\nUnique 'location' values (first 10):")
    print(df['location'].dropna().unique()[:10])

except Exception as e:
    print(f"Error: {e}")
