import os
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from mpl_toolkits.basemap import Basemap
# 'TERRA_MODIS.20190101T000001.L2.SST.nc'
datanamelist = os.listdir('D:\codeeeee\practise_1\data2')
for i in range(0,287):
    # 文件路径
    dataname=datanamelist[i]
    file_path = f"D:\codeeeee\practise_1\data2\{dataname}"
    output_folder = 'D:\codeeeee\practise_1/result2'
    a=Dataset('D:\codeeeee\practise_1\data2/'+dataname, 'r')
    latandlon = [a.getncattr('easternmost_longitude'),
                 a.getncattr('westernmost_longitude'),
                 a.getncattr('southernmost_latitude'),
                 a.getncattr('northernmost_latitude')]
    os.makedirs(output_folder, exist_ok=True)
    # 经纬度范围和步长，目标矩阵创建
    lat_min, lat_max = latandlon[2], latandlon[3]
    lon_min, lon_max = latandlon[1], latandlon[0]
    lat_step, lon_step = 0.01, 0.01
    lat_new = np.arange(lat_min, lat_max, lat_step)
    lon_new = np.arange(lon_min, lon_max, lon_step)
    sst_new = np.full((len(lat_new), len(lon_new)), np.nan)
    # 打开 NetCDF 文件
    with Dataset(file_path, 'r') as nc:
        # 读取 geophysical_data 子组中的 SST 数据和质量等级数据并处理
        geophysical_data = nc.groups['geophysical_data']
        sst = geophysical_data.variables['sst'][:]
        qual_sst = geophysical_data.variables['qual_sst'][:]  # 读取 SST 质量等级数据
        sst[sst == -32767] = np.nan# 处理无效值 (标记云位置和无效 SST 值为 NaN)

        # 获取原始纬度和经度
        navigation_data = nc.groups['navigation_data']
        latitudes = navigation_data.variables['latitude'][:].flatten()
        longitudes = navigation_data.variables['longitude'][:].flatten()
        sst = sst.flatten()
        qual_sst = qual_sst.flatten()

        # 筛选
        valid_quality_mask = np.isin(qual_sst, [0, 1])  # 0 = best, 1 = good
        sst = np.where(valid_quality_mask, sst, np.nan)  # 将非 best 和 good 的数据设置为 NaN

        # 遍历原始经纬度，计算在新网格中的索引并填充数据
        for lat, lon, sst_value in zip(latitudes, longitudes, sst):
            if np.isscalar(sst_value) and not np.isnan(sst_value):  # 确保 sst_value 是标量并且不是 NaN
                if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
                    # 计算索引
                    lat_idx = int((lat - lat_min) / lat_step)
                    lon_idx = int((lon - lon_min) / lon_step)
                    if 0 <= lat_idx < len(lat_new) and 0 <= lon_idx < len(lon_new):
                        sst_new[lat_idx, lon_idx] = sst_value
    # plot
    fig = plt.figure(figsize=(12, 6))
    m = Basemap(projection='cyl', llcrnrlat=lat_min, urcrnrlat=lat_max, llcrnrlon=lon_min, urcrnrlon=lon_max, resolution='l')
    m.drawcoastlines() # 绘制海岸线和国家边界
    m.drawcountries()
    m.fillcontinents(color='lightgrey')
    m.drawparallels(np.arange(lat_min, lat_max + 1, 5), labels=[1, 0, 0, 0], color='gray')# 绘制经纬度网格
    m.drawmeridians(np.arange(lon_min, lon_max + 1, 5), labels=[0, 0, 0, 1], color='gray')
    lon_grid, lat_grid = np.meshgrid(lon_new, lat_new)# 绘制 SST 数据
    # 添加错误处理的 try-except 块
    try:
        cs = m.pcolormesh(lon_grid, lat_grid, sst_new, cmap='jet', shading='auto', latlon=True)
    except ValueError as e:
        print(f"Error in file {dataname}: {e}")
        continue  # 跳过当前文件，继续下一个循环
    #画图输出
    cbar = m.colorbar(cs, location='right', pad='10%')
    cbar.set_label('SST (°C)')
    plt.title('SST - MODIS.20190101T011001 (Best & Good Quality)')
    output_file = os.path.join(output_folder, f'TERRA_MODIS_SST{i}_region_basemap_quality_filtered.png')
    plt.savefig(output_file, dpi=600)
    print("Created")