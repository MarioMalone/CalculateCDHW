#!/usr/bin/env bash
# nc_resample.sh
# 用 cdo 将 ERA5_MaxTemp_*.nc 从 0.1° 重采样到 0.5°

set -euo pipefail        # 出错立即退出，未定义变量报错
shopt -s nullglob        # 没有匹配文件时，通配符展开为空串而不是保持原样

# ----------------- 参数 -----------------
# 若调用时给参数，则以实参为准；否则用默认值
# 例：./nc_resample.sh /cygdrive/e/CalculateCDHW/data/MaxTemp_Merged
indir="${1:-/cygdrive/e/CalculateCDHW/data/MaxTemp_Merged}"
outdir="${2:-${indir}_0.5deg}"

# ----------------- 检查目录 -----------------
if [[ ! -d "$indir" ]]; then
  echo "输入目录不存在：$indir"
  exit 1
fi
mkdir -p "$outdir"

# ----------------- 生成 0.5° 网格文件 -----------------
gridfile="$(mktemp)"         # 临时文件
cat > "$gridfile" <<EOF
gridtype = lonlat
xsize    = 720
ysize    = 360
xfirst   = -179.75
xinc     = 0.5
yfirst   = -89.75
yinc     = 0.5
EOF

# ----------------- 批量处理 -----------------
echo "开始重采样，输入目录：$indir"
file_list=("$indir"/ERA5_MaxTemp_*.nc)

if [[ ${#file_list[@]} -eq 0 ]]; then
  echo "目录中没有匹配到 ERA5_MaxTemp_*.nc 文件"
  rm -f "$gridfile"
  exit 1
fi

for f in "${file_list[@]}"; do
  fname=$(basename "$f")
  echo "  -> 处理 $fname ..."
  cdo -O remapbil,"$gridfile" "$f" "$outdir/${fname%.nc}_0.5deg.nc"
done

rm -f "$gridfile"
echo "全部完成！重采样结果已保存到：$outdir"