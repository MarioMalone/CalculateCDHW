import pandas as pd
import xarray as xr
import numpy as np
import rioxarray
import glob
import os
import geopandas as gpd
import regionmask

# --- Configuration ---
DATA_DIR = "data"
TMAX_FILES_PATTERN = os.path.join(DATA_DIR, "MaxTemp_Merged_0.5deg", "ERA5_MaxTemp_*_0.5deg.nc")
SPEI_FILE = os.path.join(DATA_DIR, "spei03.nc")
MAIZE_AREA_FILE = os.path.join(DATA_DIR, "spam2000v3r7_harvested-area_MAIZ.tif")
GROWING_SEASON_FILE = os.path.join(DATA_DIR, "global.maize.growing.season.csv")
COUNTRIES_SHP_FILE = os.path.join(DATA_DIR, "ne_110m_admin_0_countries", "ne_110m_admin_0_countries.shp")
OUTPUT_FILE = os.path.join(DATA_DIR, "cdhw_country_annual_summary_AgERA5.csv")

# Thresholds
T_THRESH_C_29 = 29.0
T_THRESH_C_30 = 30.0
SPEI_THRESH = -1.0

# Convert Celsius to Kelvin
T_THRESH_K_29 = T_THRESH_C_29 + 273.15
T_THRESH_K_30 = T_THRESH_C_30 + 273.15

def create_growing_season_mask_vectorized(times, da_start, da_end):
    months = times.dt.month
    mask1 = (months >= da_start) & (months <= da_end)
    mask2 = (months >= da_start) | (months <= da_end)
    return xr.where(da_start <= da_end, mask1, mask2)

def process_chunk(tmax_file, ds_spei, da_area, df_gs, countries):
    """Processes a single Tmax file using a robust, explicit loop for aggregation."""
    print(f"--- Processing file: {tmax_file} ---")

    # 1. Data Loading and Alignment (same as before)
    ds_tmax = xr.open_dataset(tmax_file)
    # obtain the variable name list from nc dataset 
    variables = list(ds_tmax.data_vars) 
    # rename the first variable to 'tmax'
    ds_tmax = ds_tmax.rename({variables[0]: 'tmax'})
    ds_tmax.rio.write_crs("EPSG:4326", inplace=True)

    ds_spei_aligned = ds_spei.interp_like(ds_tmax, method='nearest')
    da_area_aligned = da_area.rio.reproject_match(ds_tmax.tmax)
    da_area_aligned = da_area_aligned.rename({'y': 'lat', 'x': 'lon'})
    
    spei_daily = ds_spei_aligned['spei'].resample(time='1D').ffill()

    min_time = ds_tmax.time.min().values
    max_time = ds_tmax.time.max().values
    ds_tmax_aligned = ds_tmax.sel(time=slice(min_time, max_time))
    spei_daily_aligned = spei_daily.sel(time=slice(min_time, max_time))

    # 2. Mask Creation (same as before)
    ds_gs = df_gs.set_index(['Latitude', 'Longitude']).to_xarray()
    ds_gs = ds_gs.rename({'Latitude': 'lat', 'Longitude': 'lon'})
    ds_gs_aligned = ds_gs.reindex_like(ds_tmax_aligned.isel(time=0, drop=True), method='nearest')
    
    def get_month_from_day_of_year(da):
        original_shape = da.shape
        flat_values = da.values.flatten()
        with np.errstate(invalid='ignore'):
            flat_months = pd.to_datetime(flat_values, format='%j', errors='coerce').month
        return np.reshape(flat_months, original_shape)

    start_month_2d = get_month_from_day_of_year(ds_gs_aligned['plant.start.day'])
    end_month_2d = get_month_from_day_of_year(ds_gs_aligned['harvest.end.day'])

    start_month_da = xr.DataArray(start_month_2d, coords=ds_gs_aligned.coords, dims=ds_gs_aligned.dims)
    end_month_da = xr.DataArray(end_month_2d, coords=ds_gs_aligned.coords, dims=ds_gs_aligned.dims)

    gs_mask = create_growing_season_mask_vectorized(ds_tmax_aligned.time, start_month_da, end_month_da)
    country_mask = regionmask.mask_geopandas(countries, ds_tmax_aligned.lon, ds_tmax_aligned.lat)

    # 3. CDHW Day Calculation (same as before)
    drought_mask = (spei_daily_aligned < SPEI_THRESH)
    heatwave_mask_29 = (ds_tmax_aligned['tmax'] > T_THRESH_K_29)
    cdhw_days_29 = heatwave_mask_29 & drought_mask & gs_mask

    heatwave_mask_30 = (ds_tmax_aligned['tmax'] > T_THRESH_K_30)
    cdhw_days_30 = heatwave_mask_30 & drought_mask & gs_mask

    annual_cdhw_29 = cdhw_days_29.groupby('time.year').sum(dim='time', dtype='int16')
    annual_cdhw_30 = cdhw_days_30.groupby('time.year').sum(dim='time', dtype='int16')

    area_weights = da_area_aligned.fillna(0).where(da_area_aligned > 0, 0)

    # 4. ROBUST AGGREGATION WITH EXPLICIT LOOPS
    print("Loading chunk data into memory for robust aggregation...")
    annual_cdhw_29.load()
    annual_cdhw_30.load()
    area_weights.load()
    country_mask.load()

    results_list = []
    regions = np.unique(country_mask.values)
    regions = regions[~np.isnan(regions)]
    country_name_map = {i: countries.iloc[i]['ADMIN'] for i in range(len(countries))}
    # 尝试获取 ISO3 代码字段，常见列名依次为 'ADM0_A3', 'ISO_A3', 'ISO_A3_EH'
    possible_iso_cols = [col for col in ['ADM0_A3', 'ISO_A3', 'ISO_A3_EH'] if col in countries.columns]
    iso_col = possible_iso_cols[0] if possible_iso_cols else None
    if iso_col is None:
        raise ValueError("No ISO3 column found in countries shapefile. Available columns: %s" % list(countries.columns))
    country_iso_map = {i: countries.iloc[i][iso_col] for i in range(len(countries))}

    for year_val in annual_cdhw_29.year.values:
        print(f"  Aggregating for year {year_val}...")
        for region_idx in regions:
            region_bool_mask = (country_mask == region_idx)
            
            cdhw29_year = annual_cdhw_29.sel(year=year_val)
            cdhw30_year = annual_cdhw_30.sel(year=year_val)

            country_weights = area_weights.where(region_bool_mask)
            total_weight = country_weights.sum().item()
            
            if total_weight > 0:
                country_cdhw29 = cdhw29_year.where(region_bool_mask)
                weighted_sum_29 = (country_cdhw29 * country_weights).sum().item()
                mean_29 = weighted_sum_29 / total_weight

                country_cdhw30 = cdhw30_year.where(region_bool_mask)
                weighted_sum_30 = (country_cdhw30 * country_weights).sum().item()
                mean_30 = weighted_sum_30 / total_weight
            else:
                mean_29 = np.nan
                mean_30 = np.nan

            results_list.append({
                'year': year_val,
                'country_code': int(region_idx),
                'CDHW29_days': mean_29,
                'CDHW30_days': mean_30,
                'country_iso': country_iso_map[int(region_idx)]
            })

            print(f"{region_idx}: {country_name_map[region_idx]} ({country_iso_map[region_idx]})")

    df = pd.DataFrame(results_list)
    df['country'] = df['country_code'].map(country_name_map)
    df['country_iso'] = df['country_code'].map(country_iso_map)
    return df.drop(columns='country_code')

def main():
    """Main function to calculate CDHW by processing files one by one."""
    print("--- Starting CDHW Calculation (Robust Aggregation) ---")

    print("Step 1: Loading non-timeseries data...")
    ds_spei = xr.open_dataset(SPEI_FILE).rename({'spei': 'spei'})
    da_area = rioxarray.open_rasterio(MAIZE_AREA_FILE, masked=True).squeeze()
    df_gs = pd.read_csv(GROWING_SEASON_FILE)
    countries = gpd.read_file(COUNTRIES_SHP_FILE)

    tmax_files = sorted(glob.glob(TMAX_FILES_PATTERN))
    if not tmax_files:
        raise FileNotFoundError(f"No Tmax files found: {TMAX_FILES_PATTERN}")

    all_results_dfs = []
    for tmax_file in tmax_files:
        chunk_df = process_chunk(tmax_file, ds_spei, da_area, df_gs, countries)
        all_results_dfs.append(chunk_df)

    print("--- Finalizing Results ---")
    final_df = pd.concat(all_results_dfs).dropna()
    # 为方便合并，保持 iso3 代码列，同时按 year、country 排序
    final_df.sort_values(['year', 'country'], inplace=True)

    final_df.to_csv(OUTPUT_FILE, index=False)

    print("--- Calculation Complete ---")
    print(f"Results saved to {OUTPUT_FILE}")
    print("Sample of the final results:")
    print(final_df.head())

if __name__ == "__main__":
    main()
