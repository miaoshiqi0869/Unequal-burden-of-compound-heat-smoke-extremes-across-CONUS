# coding = 'utf-8'
import numpy as np
import xarray as xr

def heat_identify(temp, threshold, cons):
    '''
    input:
        temp: DataArray (time, id)
        threshold: 95th threshold (array:(id,))
        consecutive days: 1 day
    return:
        HD_flag (time, id): -1:non HD | 1:consecutive HDs | 0:occasional HD or nan
    '''
    
    # 初始化heat_day_flag
    HD_flag = xr.zeros_like(temp, dtype=np.int8)
    # 标记非热浪日为-1
    HD_flag = xr.where(temp <= threshold, -1, HD_flag)  # xarray自动按dim='id'对齐广播
    # 标记连续热浪日为1
    for i in range(len(temp.time)):
        window_left = i - cons//2 + 1 - cons%2  # cons为奇数时两边对称, cons为偶数时左边少1个
        window_right = i + cons//2
        if window_left >= 0 and window_right <= len(temp.time) - 1: # 防止索引超出
            window_slice = slice(window_left, window_right + 1)     # 按位置索引slice不含右边界
            HD_flag[{'time': window_slice}] = xr.where(
                (temp.isel(time=window_slice) >= threshold).all(dim='time'),    # all(dim='time')逐id判断, 生成(id,)形状bool数组
                1,
                HD_flag.isel(time=window_slice) # xarray会自动将cond沿time维度复制广播到(time, id)维度以匹配y
            )

    return HD_flag


def smoke_identify(pm25, threshold):
    '''
    input:
        pm25: DataArray
        threshold: float 绝对阈值
    return:
        SD_flag (time, id): True/False
    '''

    # 标记wildfire_smoke_day_flag
    SD_flag = (pm25 >= threshold)
    '''
    SD_flag = xr.zeros_like(pm25, dtype=np.int8)    # safe (≤5)
    SD_flag = xr.where(pm25>5, 1, SD_flag)  # low (5-10]
    SD_flag = xr.where(pm25>10, 2, SD_flag) # moderate (10-15]
    SD_flag = xr.where(pm25>15, 3, SD_flag) # high (15-25]
    SD_flag = xr.where(pm25>25, 4, SD_flag) # very high (25-35]
    SD_flag = xr.where(pm25>35, 5, SD_flag) # hazardous (>35)
    '''

    return SD_flag


def events_identify(HD_flag, SD_flag, lag_window):
    '''
    input:
        HD_flag (time, id): heat_day_flag
        SD_flag (time, id): smoke_day_flag
        lat_window: int, 滑动窗口宽度(eg. 3表示0-1 lag day), center=True
    return:
        final_compound_flag (time, id): True (0-1 lag day内的HD和SD及其连续段) | False
        individual_HW_flag (time, id): True | False
        individual_WFS_flag (time, id): True | False
    '''

    # ==========================================
    # 1. 初步判断符合0-1 lag day的compound day
    # ==========================================
    def any_in_window(arr, windowsize=lag_window):
        '''输入bool数组, 统计窗口期内heat/smoke day的数量'''
        return arr.rolling(time=windowsize, center=True).sum() > 0  # 滑动计算以每个时刻为中心的3天窗口期内True值的数量
    
    HD_in_window = any_in_window(HD_flag == 1)
    SD_in_window = any_in_window(SD_flag)
    preliminary_compound = (HD_in_window & SD_flag) | (SD_in_window & (HD_flag == 1))   # (与HD/SD同一天或滞后一天发生)

    # ==========================================
    # 2. 将连续事件段加入compound day
    # ==========================================
    def mark_continuous_events(flag, preliminary):
        '''输入flag (bool数组)和preliminary_compound, 返回补充标记连续段后的HD_segments/WFS_segments'''
        
        # 计算差分找到连续段边界
        diff = flag.astype(np.int8).diff(dim='time', label='upper') # 差分结果对齐到后一个时间点, 生成形状为(time-1, id)
        diff_aligned = xr.concat(
            [xr.DataArray([0], dims='time', coords={'time': [flag.time[0].values]}),    # 自动广播标量值到所有id维度
             diff],
            dim='time'
        )   # 在第一个时间点插入0使diff长度与flag一致

        # 判断是否连续段开始
        starts = (diff_aligned == 1)    # (time, id)
        starts[{'time': 0}] = flag.isel(time=0) # 单独处理第一个时间点(含仅第一个时间点flag=True和前两个时间点均=True的情况)
        # 判断是否连续段结束
        ends = (diff_aligned == -1)     # （time, id)
        ends = ends.shift(time=-1, fill_value=False)    # 向前移动一天修正结束位置(连续段的最后一天, 而非连续段的后一天)
        ends[{'time': -1}] = flag.isel(time=-1) # 单独处理最后一个时间点(含仅最后一个时间点flag=True和最后两个时间点均=True的情况)

        # 标记所有包含preliminary_compound的连续段
        full_segments = xr.zeros_like(flag)
        for id_idx in range(len(flag['id'])):
            # 获取逐id的连续段开始/结束位置
            id_starts = np.where(starts.isel(id=id_idx))[0]  # np.where返回符合条件的位置索引(分维度, 一维则返回(array([...]),))
            id_ends = np.where(ends.isel(id=id_idx))[0]

            if len(id_starts) == 0:
                continue
            if len(id_starts) != len(id_ends):
                print(f"id: {flag['id'][id_idx].values}起止位置数量不等")
                print(f'len(id_starts)={len(id_starts)}, {id_starts}')
                print(f'len(id_ends)={len(id_ends)}, {id_ends}')
                continue

            for start_idx, end_idx in zip(id_starts, id_ends):
                continuous_slice = slice(start_idx, end_idx + 1)
                if preliminary.isel(time=continuous_slice, id=id_idx).any():    # 如果连续段中含compound day
                    full_segments[{'time': continuous_slice, 'id': id_idx}] = True  # 整个连续段均标记为compound day
                    
        return full_segments
    
    # ==========================================
    # 3. 分离compound/individual events
    # ==========================================
    HW_continuous = mark_continuous_events(HD_flag == 1, preliminary_compound)  # 标记HW连续段
    WFS_continuous = mark_continuous_events(SD_flag, preliminary_compound)  # 标记WFS连续段
    final_compound_flag = preliminary_compound | HW_continuous | WFS_continuous # 终极compound day标记
    individual_HW_flag = (HD_flag == 1) & ~final_compound_flag  # HW_only
    individual_WFS_flag = SD_flag & ~final_compound_flag    # WFS_only

    return final_compound_flag, individual_HW_flag, individual_WFS_flag

def process_yr(yr, temp, metric, pm25, temp_threshold, temp_cons, pm_threshold, lag_window):
    '''封装循环内容, 局部变量自动释放, 以节省内存或避免变量污染'''
    # heat_day_identify
    HD_flag = heat_identify(temp, threshold=temp_threshold, cons=temp_cons)
        
    # smoke_day_identify
    SD_flag = smoke_identify(pm25, threshold=pm_threshold)
        
    # events_indentify
    compound_flag, HW_flag, WFS_flag = events_identify(HD_flag, SD_flag, lag_window=lag_window)

    # 计算intensity
    HW_intensity = (temp - temp_threshold).where(HW_flag).mean(dim='time', skipna=True).astype('float32')
    WFS_intensity = (pm25 - pm_threshold).where(WFS_flag).mean(dim='time', skipna=True).astype('float32')
    compound_HW_intensity = (temp - temp_threshold).where(compound_flag & (HD_flag==1)).mean(dim='time', skipna=True).astype('float32')
    compound_WFS_intensity = (pm25 - pm_threshold).where(compound_flag & SD_flag).mean(dim='time', skipna=True).astype('float32')
    if metric == 'HI':  # °F需转换为°C: °C = (°F - 32) * 5/9, intensity直接缩放倍数即可
        HW_intensity = HW_intensity * 5/9
        compound_HW_intensity = compound_HW_intensity * 5/9

    # 将结果保存为新的Dataset, _flag数据较大会使内存过载, 暂不存储
    result_ds = xr.Dataset({
        'HW_days': HW_flag.sum(dim='time').astype('int16'), # 当年individual_HW总天数
        'HW_intensity': HW_intensity,   # 当年individual_HW平均强度
        'WFS_days': WFS_flag.sum(dim='time').astype('int16'),
        'WFS_intensity': WFS_intensity,
        'compound_days': compound_flag.sum(dim='time').astype('int16'),
        'compound_HW_intensity': compound_HW_intensity,
        'compound_WFS_intensity': compound_WFS_intensity
    })
    # 添加坐标信息
    result_ds['longitude'] = temp['longitude']
    result_ds['latitude'] = temp['latitude']
    # 添加年份信息并设置为坐标变量
    result_ds = result_ds.assign_coords(year=yr).expand_dims('year')
    # 添加description
    result_ds = result_ds.assign_attrs({
        'temp_threshold': '95th for 1 day',
        'pm_threshold': '15 μg/m3'
    })

    # 统计月结果
    monthly_ds = xr.Dataset({
        'HW_days': HW_flag.groupby(['time.year', 'time.month']).sum().astype('int8'),
        'WFS_days': WFS_flag.groupby(['time.year', 'time.month']).sum().astype('int8'),
        'compound_days': compound_flag.groupby(['time.year', 'time.month']).sum().astype('int8')
    })

    return result_ds, monthly_ds


def process_all_yrs(metric, temp_dir, pm_dir, temp_threshold, pm_threshold, temp_cons, lag_window, combo_path):
    '''
    - temp_dir: dir for 10km temp data
    - pm_dir: dir for 10km pm data
    - combo_path: dir for final results (eg. '{metric}_{t0}_{p0}_{lag}')
    '''
    total_start_time = tm.time()

    results=[]
    months=[]   # 存储逐月统计结果
    for yr in range(2006, 2024):
        
        with xr.open_dataarray(rf'{temp_dir}\{metric}_{yr}.nc') as temp:
            with xr.open_dataarray(rf'{pm_dir}\smokePM_complete_{yr}.nc') as pm25:
                result_ds, monthly_ds = process_yr(yr, temp, metric=metric,
                                                   pm25=pm25,
                                                   temp_threshold=temp_threshold,
                                                   temp_cons=temp_cons,
                                                   pm_threshold=pm_threshold,
                                                   lag_window=lag_window)
                results.append(result_ds)
                result_ds.close()
                months.append(monthly_ds)
                monthly_ds.close()
        
    # 合并多年结果并存为identify_10km.nc
    identify_ds = xr.concat(results, dim='year')
    identify_ds.to_netcdf(rf'{combo_path}\identify_10km.nc', mode='w')
    identify_monthly_ds = xr.concat(months, dim='year')
    identify_monthly_ds.to_netcdf(rf'{combo_path}\identify_monthly_10km.nc', mode='w')

    identify_ds.close()
    identify_monthly_ds.close()

    

'''
print(identify_ds):
<xarray.Dataset> Size: 41MB
Dimensions:                 (year: 18, id: 100156, time: 6574)
Coordinates:
  * id                      (id) int32 401kB 1397 1398 1399 ... 221554 221555
  * time                    (time) datetime64[ns] 53kB 2006-01-01 ... 2023-12-31
  * year                    (year) int16 36B 2006 2007 2008 ... 2021 2022 2023
    longitude               (id) float32 401kB ...
    latitude                (id) float32 401kB ...
    quantile                float64 8B ...
Data variables:
    HW_days                 (year, id) int16 4MB ...
    HW_intensity            (year, id) float32 7MB ...
    WFS_days                (year, id) int16 4MB ...
    WFS_intensity           (year, id) float32 7MB ...
    compound_days           (year, id) int16 4MB ...
    compound_HW_intensity   (year, id) float32 7MB ...
    compound_WFS_intensity  (year, id) float32 7MB ...
Attributes:
    temp_threshold:  95th for 1 day
    pm_threshold:    15 μg/m3
