# CDHW (Concurrent Drought and Heatwave Days) Analysis

本项目用于分析全球范围内，在玉米种植区域内发生的干旱热浪复合事件（CDHW）情况，评估其年际变化及其国家尺度的分布特征。

## 🌍 项目目标
基于气象数据、干旱指数、玉米种植面积、生长季信息和国家边界，计算在不同国家、不同年份中，出现 **CDHW 事件的年平均天数**。

CDHW 被定义为满足以下三个条件的日子：
1. 日最高气温（Tmax）超过 29℃ 或 30℃
2. SPEI 干旱指数 < -1（表示干旱）
3. 处于作物生长季内（播种到收获）

## 📂 输入数据说明
- `tasmax`：每日最高气温（单位 Kelvin）NetCDF 文件，命名格式统一
- `spei`：月度 SPEI 干旱指数 NetCDF 文件
- `maize_area`：玉米种植面积 GeoTIFF，作为加权因子
- `growing season`：CSV 文件，记录各网格点播种日/收获日（儒略日）
- `country shapefile`：国家边界 shapefile，用于生成空间掩膜

## 🧪 CDHW 计算方法概述

### 1. 生长季掩膜生成
```python
create_growing_season_mask_vectorized(times, da_start, da_end)
```
- 将播种/收获日期转换为月份，并生成掩膜（考虑跨年情况）

### 2. 计算满足 CDHW 条件的天数
```python
cdhw_days = (tmax > threshold) & (spei < -1) & growing_season_mask
```
- 按日筛选出满足所有条件的网格点

### 3. 年度累计（按网格点）
```python
annual_cdhw = cdhw_days.groupby('time.year').sum(dim='time')
```

### 4. 国家级空间加权平均
```python
weighted_sum = (cdhw_days * area_weights).sum()
mean = weighted_sum / total_area
```
- 使用玉米种植面积作为权重，对每个国家区域进行平均

## 📤 输出结果
- 一个 CSV 文件，包含每个国家、每年下的平均 CDHW 天数
- 同时包含国家名称、ISO3 代码、不同温度阈值下的 CDHW（29℃ 和 30℃）

## 📎 示例输出格式
| year | country | country_iso | CDHW29_days | CDHW30_days |
|------|---------|-------------|-------------|-------------|
| 2011 | Brazil  | BRA         | 12.3        | 5.1         |

## 🚀 快速开始
```bash
python calculate_cdhw.py
```

确保 `data/` 文件夹下包含必要数据文件，输出结果将保存在 `cdhw_country_annual_summary.csv`。

## 📌 附：CDHW 术语定义
> CDHW（Concurrent Drought and Heatwave Days）是指：在玉米种植区的生长季内，
> 同时经历干旱（SPEI < -1）与高温（Tmax > 29℃ 或 30℃）的日子。

---