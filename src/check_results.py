
import pandas as pd

OUTPUT_FILE = 'data/cdhw_country_annual_summary.csv'

df = pd.read_csv(OUTPUT_FILE)

unique_countries = df['country'].unique()

print(f"Found {len(unique_countries)} unique countries in the output file:")
print(unique_countries)
