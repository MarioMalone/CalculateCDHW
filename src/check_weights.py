import rioxarray
import geopandas as gpd
import regionmask
import numpy as np

MAIZE_AREA_FILE = "data/spam2000v3r7_harvested-area_MAIZ.tif"
COUNTRIES_SHP_FILE = "data/ne_110m_admin_0_countries/ne_110m_admin_0_countries.shp"

def check_weights():
    print("--- Analyzing Maize Harvested Area Data ---")
    
    # Load data
    da_area = rioxarray.open_rasterio(MAIZE_AREA_FILE, masked=True).squeeze()
    countries = gpd.read_file(COUNTRIES_SHP_FILE)
    
    # Create a mask to assign grid cells to countries
    # 使用二维整数掩膜，索引值对应 countries DataFrame 的行号
    country_mask = regionmask.mask_geopandas(
        countries,
        da_area.x,
        da_area.y
    )
    
    # Fill NaNs with 0 to ensure all areas are included in the sum
    area_filled = da_area.fillna(0)
    
    # Group the area data by country and sum the harvested area for each
    total_area_per_country = area_filled.groupby(country_mask).sum()
    
    # Load the results into memory
    total_area_per_country.load()
    
    # Filter for countries with a total harvested area greater than zero
    countries_with_maize = total_area_per_country.where(total_area_per_country > 0, drop=True)
    
    country_name_map = {i: countries.iloc[i]['ADMIN'] for i in range(len(countries))}
    
    # Depending on xarray version, the new dimension after groupby may be called 'group'
    dim_name = None
    for possible in ['region', 'group']:
        if possible in countries_with_maize.dims or possible in countries_with_maize.coords:
            dim_name = possible
            break
    if dim_name is None:
        dim_name = list(countries_with_maize.dims)[0]  # fallback

    num_countries = countries_with_maize.sizes[dim_name]
    print(f"Found {num_countries} countries with non-zero maize harvested area.")
    print("----------------------------------------------------")
    print("List of countries with maize data:")
    for region_idx in countries_with_maize[dim_name].values:
        print(f" - {country_name_map.get(int(region_idx), 'Unknown')}")

if __name__ == "__main__":
    check_weights()
