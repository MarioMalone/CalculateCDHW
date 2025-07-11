import pandas as pd
import xarray as xr
import numpy as np
import rioxarray
import glob
import os
import geopandas as gpd
import regionmask

import config

# Thresholds and configuration
TEMP_FILES_PATTERN = os.path.join(config.MeanTempPath, "*.nc")
MAIZE_AREA_FILE = config.MaizeAreaPath
GROWING_SEASON_FILE = config.GrowingSeasonPath
COUNTRIES_SHP_FILE = config.COUNTRIES_SHP_FILE
OUTPUT_FILE = os.path.join(config.results_path, "country_mean_temperature.csv")

def create_growing_season_mask_vectorized(times, da_start, da_end):
    """创建生长季节掩码的向量化函数"""
    months = times.dt.month
    mask1 = (months >= da_start) & (months <= da_end)
    mask2 = (months >= da_start) | (months <= da_end)
    return xr.where(da_start <= da_end, mask1, mask2)

def get_month_from_day_of_year(da):
    """将年积日转换为月份"""
    original_shape = da.shape
    flat_values = da.values.flatten()
    with np.errstate(invalid='ignore'):
        flat_months = pd.to_datetime(flat_values, format='%j', errors='coerce').month
    return np.reshape(flat_months, original_shape)

def process_temp_chunk(temp_file, da_area, df_gs, countries):
    """处理单个温度文件，计算每个国家的面积加权平均温度"""
    print(f"--- 正在处理文件: {temp_file} ---")

    # 1. 数据加载和对齐
    ds_temp = xr.open_dataset(temp_file)
    # 获取变量名列表
    variables = list(ds_temp.data_vars)
    # 重命名第一个变量为 'temp'
    ds_temp = ds_temp.rename({variables[0]: 'temp'})
    ds_temp.rio.write_crs("EPSG:4326", inplace=True)

    # 对齐面积数据
    da_area_aligned = da_area.rio.reproject_match(ds_temp.temp)
    da_area_aligned = da_area_aligned.rename({'y': 'lat', 'x': 'lon'})

    # 2. 生长季节掩码创建
    ds_gs = df_gs.set_index(['Latitude', 'Longitude']).to_xarray()
    ds_gs = ds_gs.rename({'Latitude': 'lat', 'Longitude': 'lon'})
    ds_gs_aligned = ds_gs.reindex_like(ds_temp.isel(time=0, drop=True), method='nearest')

    start_month_2d = get_month_from_day_of_year(ds_gs_aligned['plant.start.day'])
    end_month_2d = get_month_from_day_of_year(ds_gs_aligned['harvest.end.day'])

    start_month_da = xr.DataArray(start_month_2d, coords=ds_gs_aligned.coords, dims=ds_gs_aligned.dims)
    end_month_da = xr.DataArray(end_month_2d, coords=ds_gs_aligned.coords, dims=ds_gs_aligned.dims)

    gs_mask = create_growing_season_mask_vectorized(ds_temp.time, start_month_da, end_month_da)
    
    # 创建国家掩码，使用 regionmask.from_geopandas 来确保索引一致性
    country_regions = regionmask.from_geopandas(countries, names="ADMIN", abbrevs="ISO_A3")
    country_mask = country_regions.mask(ds_temp.lon, ds_temp.lat)

    # 3. 应用生长季节掩码到温度数据并转换为摄氏度
    temp_gs = ds_temp['temp'].where(gs_mask)
    # 将开尔文温度转换为摄氏度
    temp_gs_celsius = temp_gs - 273.15

    # 4. 按年份聚合温度数据
    annual_temp_mean = temp_gs_celsius.groupby('time.year').mean(dim='time')

    # 5. 面积权重
    area_weights = da_area_aligned.fillna(0).where(da_area_aligned > 0, 0)

    # 6. 稳健聚合：使用显式循环
    print("将数据加载到内存中进行稳健聚合...")
    annual_temp_mean.load()
    area_weights.load()
    country_mask.load()

    results_list = []
    regions = np.unique(country_mask.values)
    regions = regions[~np.isnan(regions)]
    
    # 使用 regionmask 的映射来创建名称和ISO代码映射
    country_name_map = dict(zip(country_regions.numbers, country_regions.names))
    country_iso_map = dict(zip(country_regions.numbers, country_regions.abbrevs))

    for year_val in annual_temp_mean.year.values:
        print(f"  正在聚合 {year_val} 年数据...")
        for region_idx in regions:
            region_bool_mask = (country_mask == region_idx)
            
            temp_year = annual_temp_mean.sel(year=year_val)
            country_weights = area_weights.where(region_bool_mask)
            total_weight = country_weights.sum().item()
            
            if total_weight > 0:
                country_temp = temp_year.where(region_bool_mask)
                weighted_sum = (country_temp * country_weights).sum().item()
                mean_temp = weighted_sum / total_weight
            else:
                mean_temp = np.nan

            results_list.append({
                'year': year_val,
                'country_code': int(region_idx),
                'mean_temp': mean_temp,
                'country_iso': country_iso_map[int(region_idx)]
            })

    df = pd.DataFrame(results_list)
    df['country'] = df['country_code'].map(country_name_map)
    df['country_iso'] = df['country_code'].map(country_iso_map)
    return df.drop(columns='country_code')

def calculate_country_mean_temp():
    """
    计算每个国家每年生长季节内的面积加权平均温度
    """
    print("--- 开始计算国家面积加权平均温度 ---")

    print("步骤1: 加载非时间序列数据...")
    da_area = rioxarray.open_rasterio(MAIZE_AREA_FILE, masked=True).squeeze()
    df_gs = pd.read_csv(GROWING_SEASON_FILE)
    countries = gpd.read_file(COUNTRIES_SHP_FILE)
    countries = countries[countries.ISO_A3 != "-99"]  # 过滤无效国家

    temp_files = sorted(glob.glob(TEMP_FILES_PATTERN))
    if not temp_files:
        raise FileNotFoundError(f"未找到温度文件: {TEMP_FILES_PATTERN}")

    print(f"找到 {len(temp_files)} 个温度文件")

    all_results_dfs = []
    for temp_file in temp_files:
        chunk_df = process_temp_chunk(temp_file, da_area, df_gs, countries)
        all_results_dfs.append(chunk_df)

    print("--- 整理最终结果 ---")
    final_df = pd.concat(all_results_dfs).dropna()
    final_df.sort_values(['year', 'country'], inplace=True)

    # 创建输出目录
    if not os.path.exists(config.results_path):
        os.makedirs(config.results_path)

    final_df.to_csv(OUTPUT_FILE, index=False)

    print("--- 计算完成 ---")
    print(f"结果已保存到 {OUTPUT_FILE}")
    print("最终结果样本:")
    print(final_df.head(10))
    print(f"总共计算了 {len(final_df)} 条记录")

if __name__ == "__main__":
    calculate_country_mean_temp()
