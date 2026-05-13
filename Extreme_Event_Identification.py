# coding = 'utf-8'
import numpy as np
import xarray as xr

def events_identify(HD_flag, SD_flag, lag_window):
    '''
    input:
        HD_flag (time, id): extreme hot day flag
        SD_flag (time, id): extreme smoke day flag
        lag_window: int, rolling window width (e.g., window=3 -> lag<=1 day)
    return:
        final_compound_flag (time, id): compound heat-smoke event flag
        individual_EH_flag (time, id): individual extreme heat event flag
        individual_WFS_flag (time, id): individual wildfire smoke event flag
    '''

    # ==========================================
    # 1. Preliminary identification of compound days
    # ==========================================
    def any_in_window(arr, windowsize=lag_window):
        '''input: bool array, count the number of True values within the rolling window, and convert to bool array'''
        return arr.rolling(time=windowsize, center=True).sum() > 0
    
    HD_in_window = any_in_window(HD_flag)
    SD_in_window = any_in_window(SD_flag)
    preliminary_compound = (HD_in_window & SD_flag) | (SD_in_window & HD_flag)   # preliminary compound day within the specified lag window

    # ==========================================
    # 2. Expand classification to include continuous extreme hot/smoke days as compound days
    # ==========================================
    def mark_continuous_events(flag, preliminary):
        '''input: flag (bool array) and preliminary_compound, return the extended segments of EH/WFS'''
        # calculate the difference to find the boundaries of continuous segments
        diff = flag.astype(np.int8).diff(dim='time', label='upper')
        diff_aligned = xr.concat(
            [xr.DataArray([0], dims='time', coords={'time': [flag.time[0].values]}),
             diff],
            dim='time'
        )   # insert 0 at the first time point to make the length of diff equal to flag

        # check if the continuous segment starts
        starts = (diff_aligned == 1)
        starts[{'time': 0}] = flag.isel(time=0)
        # check if the continuous segment ends
        ends = (diff_aligned == -1)
        ends = ends.shift(time=-1, fill_value=False)
        ends[{'time': -1}] = flag.isel(time=-1)

        # mark all continuous segments containing preliminary_compound
        full_segments = xr.zeros_like(flag)
        for id_idx in range(len(flag['id'])):
            # get the start/end positions of the continuous segments for each id (10km grid cell)
            id_starts = np.where(starts.isel(id=id_idx))[0]
            id_ends = np.where(ends.isel(id=id_idx))[0]

            if len(id_starts) == 0:
                continue
            if len(id_starts) != len(id_ends):
                print(f"id: {flag['id'][id_idx].values} has mismatched start/end counts")
                print(f'len(id_starts)={len(id_starts)}, {id_starts}')
                print(f'len(id_ends)={len(id_ends)}, {id_ends}')
                continue

            for start_idx, end_idx in zip(id_starts, id_ends):
                continuous_slice = slice(start_idx, end_idx + 1)
                if preliminary.isel(time=continuous_slice, id=id_idx).any():    # if the continuous segment contains preliminary compound day
                    full_segments[{'time': continuous_slice, 'id': id_idx}] = True  # mark the entire continuous segment as compound day

        return full_segments
    
    # ==========================================
    # 3. Separate compound/individual events
    # ==========================================
    EH_continuous = mark_continuous_events(HD_flag, preliminary_compound)
    WFS_continuous = mark_continuous_events(SD_flag, preliminary_compound)
    final_compound_flag = preliminary_compound | EH_continuous | WFS_continuous
    individual_EH_flag = HD_flag & ~final_compound_flag
    individual_WFS_flag = SD_flag & ~final_compound_flag

    return final_compound_flag, individual_EH_flag, individual_WFS_flag

def process_yr(yr, temp, metric, pm25, temp_threshold, pm_threshold, lag_window):
    HD_flag = (temp >= temp_threshold)   # extreme hot day flag (True/False), dimension: (time, id)
    SD_flag = (pm25 >= pm_threshold)   # extreme smoke day flag (True/False), dimension: (time, id)
    compound_flag, EH_flag, WFS_flag = events_identify(HD_flag, SD_flag, lag_window=lag_window)


    EH_intensity = (temp - temp_threshold).where(EH_flag).mean(dim='time', skipna=True).astype('float32')
    WFS_intensity = (pm25 - pm_threshold).where(WFS_flag).mean(dim='time', skipna=True).astype('float32')
    compound_heat_intensity = (temp - temp_threshold).where(compound_flag & HD_flag).mean(dim='time', skipna=True).astype('float32')
    compound_smoke_intensity = (pm25 - pm_threshold).where(compound_flag & SD_flag).mean(dim='time', skipna=True).astype('float32')
    if metric == 'HI':  # convert °F to °C: scale directly
        EH_intensity = EH_intensity * 5/9
        compound_heat_intensity = compound_heat_intensity * 5/9


    # save the results
    result_ds = xr.Dataset({
        'EH_frequency': EH_flag.sum(dim='time').astype('int16'),
        'EH_intensity': EH_intensity,
        'WFS_frequency': WFS_flag.sum(dim='time').astype('int16'),
        'WFS_intensity': WFS_intensity,
        'compound_frequency': compound_flag.sum(dim='time').astype('int16'),
        'compound_heat_intensity': compound_heat_intensity,
        'compound_smoke_intensity': compound_smoke_intensity
    })
    result_ds['longitude'] = temp['longitude']
    result_ds['latitude'] = temp['latitude']
    result_ds = result_ds.assign_coords(year=yr).expand_dims('year')
    result_ds = result_ds.assign_attrs({
        'temp_threshold': '95th',
        'pm_threshold': '15 μg/m3'
    })

    return result_ds


def process_all_yrs(metric, temp_dir, pm_dir, temp_threshold, pm_threshold, lag_window, combo_path):
    '''
    - metric: Ta, HI, WBGT, UTCI
    - temp_dir: directory for 10km temp data
    - pm_dir: directory for 10km pm data
    - temp_threshold: 95th percentile of 1979-2000
    - pm_threshold: 15 μg/m3
    - lag window: rolling window width (window=3 -> lag<=1 day)
    - combo_path: directory for final results (eg. '{metric}_{p0}_{lag}')
    '''

    results=[]
    for yr in range(2006, 2024):
        with xr.open_dataarray(rf'{temp_dir}\{metric}_{yr}.nc') as temp:
            with xr.open_dataarray(rf'{pm_dir}\smokePM_complete_{yr}.nc') as pm25:
                result_ds = process_yr(
                    yr, temp, metric=metric, pm25=pm25,
                    temp_threshold=temp_threshold,
                    pm_threshold=pm_threshold,
                    lag_window=lag_window
                )
                results.append(result_ds)
                result_ds.close()
    
    # concatenate the results
    identify_ds = xr.concat(results, dim='year')
    identify_ds.to_netcdf(rf'{combo_path}\identify_10km.nc', mode='w')

    identify_ds.close()
