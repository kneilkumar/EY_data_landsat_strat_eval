import datetime
import time

import numpy as np
import pandas as pd

import pystac_client
import planetary_computer
from odc.stac import stac_load
from pystac.extensions.eo import EOExtension as eo
import stackstac

from matplotlib import pyplot as plt

import xarray.ufuncs as ufuncs
import xarray as xr

from datetime import date
from tqdm import tqdm
import os

from dask.distributed import Client, LocalCluster, wait
from dask import delayed, compute, persist
import gc


def calculate_bb(station_coords, tier_size):
    buffer_deg = tier_size/110000
    aoi = [station_coords[1] - buffer_deg/2, station_coords[0] - buffer_deg/2,
           station_coords[1] + buffer_deg/2, station_coords[0] + buffer_deg/2]
    return aoi


def calculate_time_window(prototype_row, window_size):
    window_start = (prototype_row.loc['Sample Date dt'] - datetime.timedelta(days=window_size)).strftime('%Y-%m-%d')
    window_end = prototype_row.loc['Sample Date dt'].strftime('%Y-%m-%d')
    window_string = f"{window_start}/{window_end}"
    return window_string


def compute_indices(xarray,  qa_layer, window):
    eps = 1e-10
    xarray['MNDWI'] = (xarray['green'] - xarray['swir16'])/(xarray['green'] + xarray['swir16'] + eps)
    xarray['NDVI'] = (xarray['nir08'] - xarray['red'])/(xarray['nir08'] + xarray['red'] + eps)
    xarray['NDMI'] = (xarray['nir08'] - xarray['swir16'])/(xarray['nir08'] + xarray['swir16'] + eps)
    xarray['NDWI'] = (xarray['green'] - xarray['nir08'])/(xarray['green'] + xarray['nir08'] + eps)
    xarray['NDBI'] = (xarray['swir16'] - xarray['nir08'])/(xarray['swir16'] + xarray['nir08'] + eps)
    xarray['NDTI'] = (xarray['red'] - xarray['green'])/(xarray['red'] + xarray['green'] + eps)
    xarray['NDFI'] = (xarray['red'] - xarray['swir16']) / (xarray['red'] + xarray['swir16'] + eps)
    if window == 365:
        water_mask = (qa_layer & 128) != 0
        xarray[['NDVI', 'NDMI']] = xarray[['NDVI', 'NDMI']].where(~water_mask)
        water_fraction = (water_mask.sum(dim=['latitude', 'longitude'])/(water_mask.sizes['latitude'] * water_mask.sizes['longitude'])).mean(dim='time', skipna=True)
        xarray[['NDTI', 'NDFI']] = xarray[['NDTI', 'NDFI']].where(xarray['MNDWI'] > -0.1)
        return xarray, water_fraction
    return xarray


def valid_snap_mask(xarray, tier, sample_date):
    qa_layer = xarray['qa_pixel']
    spectral_bands = ["red", "green", "blue", "nir08", "swir16", "swir22"]
    invalid_bits = bit_flags['fill'] | bit_flags['dilated_cloud'] | bit_flags['cirrus'] | bit_flags['cloud'] | bit_flags['shadow'] | bit_flags['snow']
    mask = (qa_layer & invalid_bits) != 0
    valid_bits = 1 - mask
    mask_ratio_array = valid_bits.sum(dim=['latitude', 'longitude']) / (
                mask.sizes['latitude'] * mask.sizes['longitude'])
    valid_dates = xarray['time'].where(mask_ratio_array > 0.9).dropna(dim='time')
    if xarray.sizes['time'] == 0:
        lat = xarray.latitude
        lon = xarray.longitude

        nan_data = np.full((lat.size, lon.size), np.nan)

        nan_da = xr.DataArray(
            nan_data,
            dims=["latitude", "longitude"],
            coords={"latitude": lat, "longitude": lon},
        )
        scalar_nan_da = xr.DataArray(np.array([np.nan]))
        empty = xr.Dataset(data_vars={'blue': nan_da, 'red': nan_da, 'green': nan_da, 'nir08': nan_da, 'swir16': nan_da,
                                      'swir22': nan_da, 'MNDWI': nan_da, 'NDWI': nan_da, 'NDTI': nan_da, 'NDFI': nan_da,
                                      'NDVI': nan_da, 'NDBI': nan_da, 'NDMI': nan_da, 'ndti_ratio': scalar_nan_da,
                                      'ndfi_ratio': scalar_nan_da, 'is_composite': scalar_nan_da, 'mean_valid_pixel_ratio': scalar_nan_da,
                                      'days_from_sample': scalar_nan_da, 'older_than_45': scalar_nan_da})
        water_fraction = scalar_nan_da
        return empty, water_fraction
    elif (valid_dates.sizes['time'] == 0) or (tier != 90):
        print("composite activated")
        masked_xarr = xarray[spectral_bands].where(~mask)
        mean_valid_pixel_ratio = masked_xarr.count().to_array().sum()/(masked_xarr.sizes['latitude'] * masked_xarr.sizes['longitude'] * masked_xarr.sizes['time'] * len(spectral_bands))

        masked_xarr = compute_indices(masked_xarr, qa_layer, 90)
        water_mask = (qa_layer & 128) != 0
        water_fraction = (water_mask.sum(dim=['latitude', 'longitude'])/(water_mask.sizes['latitude'] * water_mask.sizes['longitude'])).mean(dim='time', skipna=True)
        masked_xarr[['NDVI', 'NDMI']] = masked_xarr[['NDVI', 'NDMI']].where(~water_mask)

        masked_xarr[['NDTI', 'NDFI']] = masked_xarr[['NDTI', 'NDFI']].where(masked_xarr['MNDWI'] > -0.1)
        composited_xarr = masked_xarr.median(dim='time')

        composited_xarr['ndti_ratio'] = (composited_xarr['NDTI'] >= 0.01).sum() / composited_xarr['MNDWI'].count()
        composited_xarr['ndfi_ratio'] = (composited_xarr['NDFI'] >= 0.5).sum() / composited_xarr['MNDWI'].count()

        composited_xarr['is_composite'] = 1
        composited_xarr['mean_valid_pixel_ratio'] = mean_valid_pixel_ratio
        composited_xarr['days_from_sample'] = np.nan
        composited_xarr['older_than_45'] = np.nan
        return composited_xarr, water_fraction
    else:
        time_deltas = np.abs(valid_dates.values - np.datetime64(sample_date))
        min_time_from_sample = np.argmin(time_deltas)
        days_from_sample = time_deltas[min_time_from_sample]/np.timedelta64(1, 'D')
        older_than_45 = np.where(days_from_sample > 45, 1, 0)
        closest_time = valid_dates.isel(time=min_time_from_sample)
        final_mask = mask.sel(time=closest_time)
        closest_slice = xarray.sel(time=closest_time)
        closest_slice = closest_slice[spectral_bands].where(~final_mask)

        water_mask = (qa_layer.sel(time=closest_time) & 128) != 0
        water_fraction = (water_mask.sum(dim=['latitude', 'longitude'])/(water_mask.sizes['latitude'] * water_mask.sizes['longitude']))

        closest_slice = compute_indices(closest_slice, qa_layer, 90)
        closest_slice[['NDVI', 'NDMI', 'NDBI']] = closest_slice[['NDVI', 'NDMI', 'NDBI']].where(~water_mask)
        closest_slice[['NDTI', 'NDFI']] = closest_slice[['NDTI', 'NDFI']].where(closest_slice['MNDWI'] > -0.1)

        closest_slice['ndti_ratio'] = (closest_slice['NDTI'] >= 0.01).sum() / closest_slice['MNDWI'].count()
        closest_slice['ndfi_ratio'] = (closest_slice['NDFI'] >= 0.5).sum() / closest_slice['MNDWI'].count()

        closest_slice['is_composite'] = 0
        closest_slice['mean_valid_pixel_ratio'] = np.nan
        closest_slice['days_from_sample'] = days_from_sample
        closest_slice['older_than_45'] = older_than_45
        return closest_slice, water_fraction


def valid_data_mask(xarray):
    invalid_bits = bit_flags['fill'] | bit_flags['dilated_cloud'] | bit_flags['cirrus'] | bit_flags['cloud'] | bit_flags['shadow'] | bit_flags['snow']
    qa_layer = xarray['qa_pixel']
    mask = (qa_layer & invalid_bits) != 0
    spectral_bands = ["red", "green", "blue", "nir08", "swir16", "swir22"]
    xarray = xarray[spectral_bands].where(~mask)
    xarray, water_frac365 = compute_indices(xarray, qa_layer, 365)
    xarray['ndti_ratio'] = (xarray['NDTI'] >= 0.01).sum() / xarray['MNDWI'].count()
    xarray['ndfi_ratio'] = (xarray['NDFI'] >= 0.5).sum() / xarray['MNDWI'].count()

    return xarray, water_frac365


def load_data(items, bounds):
    return stac_load(
        items,
        bands=["red", "green", "blue", "nir08", "swir16", "swir22", "qa_pixel"],
        crs="EPSG:4326",
        resolution=30/111320.,
        chunks={"time": 1, "x": 2048, "y": 2048},
        patch_url=planetary_computer.sign,
        bbox=bounds
)


def offset_scale(xarray):
    xarray['red'] = (xarray['red'] * 0.0000275) - 0.2
    xarray['green'] = (xarray['green'] * 0.0000275) - 0.2
    xarray['blue'] = (xarray['blue'] * 0.0000275) - 0.2
    xarray['nir08'] = (xarray['nir08'] * 0.0000275) - 0.2
    xarray['swir16'] = (xarray['swir16'] * 0.0000275) - 0.2
    xarray['swir22'] = (xarray['swir22'] * 0.0000275) - 0.2
    return xarray


def visualise_data(xarray):
    fig, ax = plt.subplots()
    xarray[['red', 'green', 'blue']].to_array().plot.imshow(robust=True, ax=ax, vmin=0, vmax=0.3, rgb='variable')
    plt.show()


def temporal_masking(xarray):
    valid_pixel_counts = xarray.count(dim='time')
    valid_temporal_mask = valid_pixel_counts >= 6
    return valid_temporal_mask


def spatial_aggregation(xarray, tier):
    if tier == 90:
        if 'time' not in xarray.dims:
            valid_pixels = xarray.count() / (xarray.sizes['latitude'] * xarray.sizes['longitude'])
            aggregated_median = xarray.mean(dim=['latitude', 'longitude'], skipna=True)
        else:
            valid_pixels = xarray.count()/(xarray.sizes['latitude'] * xarray.sizes['longitude'])
            aggregated_median = xarray.mean(dim=['latitude', 'longitude'], skipna=True).squeeze()
        return aggregated_median, valid_pixels
    else:
        if 'time' not in xarray.dims:
            valid_pixels = (xarray.count(dim=['latitude', 'longitude']) / (xarray.sizes['latitude'] * xarray.sizes['longitude'])).mean(skipna=True)
            aggregated_mean = xarray.mean(dim=['latitude', 'longitude'],skipna=True)
            aggregated_std = xarray.std(dim=['latitude', 'longitude'],skipna=True)
        else:
            valid_pixels = ((xarray.count(dim=['latitude', 'longitude']) / (xarray.sizes['latitude'] * xarray.sizes['longitude'])).mean(dim='time', skipna=True))
            aggregated_mean = xarray.mean(dim=['latitude', 'longitude'],skipna=True).mean(dim='time',skipna=True)
            aggregated_std = xarray.std(dim=['latitude', 'longitude'],skipna=True).mean(dim='time',skipna=True)
        return aggregated_mean, aggregated_std, valid_pixels


def compute_index_stats(xarray, tier):
    regular_indices = ["red", "green", "blue", "nir08", "swir16", "swir22", 'NDVI', 'NDWI', 'NDMI', 'MNDWI', 'NDBI', 'NDTI', 'NDFI']
    xarray_indices = xarray[regular_indices]
    if tier == 500:
        stats = ['NDVI_std_time', 'NDMI_std_time','NDVI_mean_time','NDMI_mean_time', 'NDTI_std_time', 'NDFI_std_time', 'NDTI_mean_time', 'NDFI_mean_time']
        stats_foundation = ['NDVI', 'NDMI', 'NDTI', 'NDFI']
        temporal_mask = temporal_masking(xarray[stats_foundation])
        xarray_stats = xarray[stats_foundation].where(temporal_mask)
        xarray_stats['NDVI_std_time'] = xarray_stats['NDVI'].std(dim='time', skipna=True)
        xarray_stats['NDMI_std_time'] = xarray_stats['NDMI'].std(dim='time', skipna=True)
        xarray_stats['NDVI_mean_time'] = xarray_stats['NDVI'].mean(dim='time', skipna=True)
        xarray_stats['NDMI_mean_time'] = xarray_stats['NDMI'].mean(dim='time', skipna=True)
        xarray_stats['NDTI_std_time'] = xarray_stats['NDTI'].std(dim='time', skipna=True)
        xarray_stats['NDFI_std_time'] = xarray_stats['NDFI'].std(dim='time', skipna=True)
        xarray_stats['NDTI_mean_time'] = xarray_stats['NDTI'].mean(dim='time', skipna=True)
        xarray_stats['NDFI_mean_time'] = xarray_stats['NDFI'].mean(dim='time', skipna=True)
    else:
        stats = ['NDMI_mean_time', 'SWIR_std_time', 'NDVI_amplitude', 'NDTI_std_time','NDFI_std_time','NDTI_mean_time', 'NDFI_mean_time']
        stats_foundation = ['NDMI', 'swir22', 'NDVI', 'NDTI', 'NDFI']
        temporal_mask = temporal_masking(xarray[stats_foundation])
        xarray_stats = xarray[stats_foundation].where(temporal_mask)
        xarray_stats['NDMI_mean_time'] = xarray_stats['NDMI'].mean(dim='time', skipna=True)
        xarray_stats['SWIR_std_time'] = xarray_stats['swir22'].std(dim='time', skipna=True)
        qs = xarray_stats['NDVI'].quantile([0.1, 0.9], dim='time')
        xarray_stats['NDVI_amplitude'] = qs.sel(quantile=0.9) - qs.sel(quantile=0.1)
        xarray_stats['NDTI_std_time'] = xarray_stats['NDTI'].std(dim='time', skipna=True)
        xarray_stats['NDFI_std_time'] = xarray_stats['NDFI'].std(dim='time', skipna=True)
        xarray_stats['NDTI_mean_time'] = xarray_stats['NDTI'].mean(dim='time', skipna=True)
        xarray_stats['NDFI_mean_time'] = xarray_stats['NDFI'].mean(dim='time', skipna=True)
    xarray_indices[stats] = xarray_stats[stats]
    return xarray_indices


def compute_lags_ints(xarray_short, xarray_long):
    ndvi_shift = xarray_short['NDVI'] - xarray_long['NDVI_mean_time']
    ndmi_shift = xarray_short['NDMI'] - xarray_long['NDMI_mean_time']
    growth_cycle = xarray_short['NDVI']/xarray_long['NDVI'].max(dim='time', skipna=True)
    drought_indicator = xarray_short['swir16']/xarray_short['nir08']
    gb_ratio = (xarray_short['NDVI'] - xarray_short['swir22'])/(xarray_short['NDVI'] + xarray_short['swir22'])


def pipeline_90day(data, station_coords, bb, sample_date):
    tier_1bb = calculate_bb(station_coords, 90)
    tier_2bb = calculate_bb(station_coords, 500)

    tier_1_data = data.sel(latitude=slice(tier_1bb[3], tier_1bb[1]), longitude=slice(tier_1bb[0], tier_1bb[2]))
    tier1_snapshot, water_frac_t1 = valid_snap_mask(tier_1_data, 90, sample_date)

    tier3_composite, water_frac_t3c = valid_snap_mask(data, 5000, sample_date)
    tier2_composite = tier3_composite.sel(latitude=slice(tier_2bb[3], tier_2bb[1]), longitude=slice(tier_2bb[0], tier_2bb[2]))
    t2_water_mask = (data['qa_pixel'].sel(latitude=slice(tier_2bb[3], tier_2bb[1]), longitude=slice(tier_2bb[0], tier_2bb[2])) & 128) != 0
    water_frac_t2c = (t2_water_mask.sum(dim=['latitude', 'longitude'])/(t2_water_mask.sizes['latitude'] * t2_water_mask.sizes['longitude'])).mean(dim='time', skipna=True)

    slice_90day = ['red', 'green', "blue", "nir08", "swir16", "swir22", 'NDVI', 'NDMI', "MNDWI", 'NDWI', 'NDBI', 'NDTI', "NDFI"]
    structural_metadata_90day = ['is_composite', 'older_than_45', 'mean_valid_pixel_ratio', 'days_from_sample', 'ndti_ratio', 'ndfi_ratio']
    t1_snap_temp = tier1_snapshot[structural_metadata_90day]
    t2_comp_temp = tier2_composite[structural_metadata_90day]
    t3_comp_temp = tier3_composite[structural_metadata_90day]

    t1_agg = spatial_aggregation(tier1_snapshot[slice_90day], 90)
    t2_agg = spatial_aggregation(tier2_composite[slice_90day], 500)
    t3_agg = spatial_aggregation(tier3_composite[slice_90day], 5000)

    tier1_sliced_90 = t1_agg[0]
    tier2_sliced_90 = t2_agg[0]
    tier3_sliced_90 = t3_agg[0]

    tier1_sliced_90 = tier1_sliced_90.assign(t1_snap_temp)
    tier2_sliced_90 = tier2_sliced_90.assign(t2_comp_temp)
    tier3_sliced_90 = tier3_sliced_90.assign(t3_comp_temp)

    water_frac_t3c.name = 'qa_pixel_t3c_waterfrac'
    water_frac_t2c.name = 'qa_pixel_t2c_waterfrac'
    water_frac_t1.name = 'qa_pixel_t1_waterfrac'

    water_frac_ds = xr.Dataset(
        {'t3c_waterfrac': water_frac_t3c, 't2c_waterfrac': water_frac_t2c, 't1_waterfrac': water_frac_t1})

    tier2_sliced_90_std = t2_agg[1][slice_90day]
    tier3_sliced_90_std = t3_agg[1][slice_90day]

    tier1_sliced_90['t1_90vp'] = t1_agg[1]['red']
    tier2_sliced_90['t2_90vp'] = t2_agg[2]['red']
    tier3_sliced_90['t3_90vp'] = t3_agg[2]['red']

    valid_pixels = xr.Dataset({'t1_90_vp': tier1_sliced_90['t1_90vp'],'t2_90_vp': tier2_sliced_90['t2_90vp'],'t3_90_vp': tier3_sliced_90['t3_90vp'],})

    tier1_sliced_90 = tier1_sliced_90.rename(
        {var: f"t1_90_{var}" for var in tier1_sliced_90.data_vars}
    )

    tier2_sliced_90 = tier2_sliced_90.rename(
        {var: f"t2_90_{var}" for var in tier2_sliced_90.data_vars}
    )

    tier3_sliced_90 = tier3_sliced_90.rename(
        {var: f"t3_90_{var}" for var in tier3_sliced_90.data_vars}
    )

    tier2_sliced_90_std = tier2_sliced_90_std.rename(
        {var: f"t2_90_{var}_std" for var in tier2_sliced_90_std.data_vars}
    )

    tier3_sliced_90_std = tier3_sliced_90_std.rename(
        {var: f"t3_90_{var}_std" for var in tier3_sliced_90_std.data_vars}
    )

    ninety_day_data = [tier1_sliced_90, tier2_sliced_90, tier3_sliced_90,water_frac_ds, tier2_sliced_90_std, tier3_sliced_90_std, valid_pixels]

    return ninety_day_data


def pipeline_365day(data, station_coords):
    tier_2bb = calculate_bb(station_coords, 500)

    masked_365_t3, water_frac_t3d = valid_data_mask(data)
    masked_365_t2 = masked_365_t3.sel(latitude=slice(tier_2bb[3], tier_2bb[1]), longitude=slice(tier_2bb[0], tier_2bb[2]))
    water_mask_t2d = (data['qa_pixel'].sel(latitude=slice(tier_2bb[3], tier_2bb[1]), longitude=slice(tier_2bb[0], tier_2bb[2])) & 128) != 0
    water_frac_t2d = (water_mask_t2d.sum(dim=['latitude', 'longitude'])/(water_mask_t2d.sizes['latitude'] * water_mask_t2d.sizes['longitude'])).mean(dim='time', skipna=True)

    masked_365_t2['qa_pixel'] = data['qa_pixel'].sel(latitude=slice(tier_2bb[3], tier_2bb[1]), longitude=slice(tier_2bb[0], tier_2bb[2]))
    masked_365_t3['qa_pixel'] = data['qa_pixel']

    masked_365_t2 = compute_index_stats(masked_365_t2, 500)
    masked_365_t3 = compute_index_stats(masked_365_t3, 5000)

    slice_365day_t3 = ['red', 'green', "blue", "nir08", "swir16", "swir22", 'NDVI', 'NDMI', "MNDWI", 'NDWI', 'NDBI', 'NDTI', 'NDFI',
                       'NDMI_mean_time', 'SWIR_std_time', 'NDVI_amplitude', 'NDFI_std_time', 'NDTI_std_time', 'NDFI_mean_time', 'NDTI_mean_time']
    slice_365day_t2 = ['red', 'green', "blue", "nir08", "swir16", "swir22", 'NDVI', 'NDMI', "MNDWI", 'NDWI', 'NDBI', 'NDTI', 'NDFI',
                       'NDMI_mean_time', 'NDMI_std_time', 'NDVI_mean_time', 'NDVI_std_time', 'NDFI_std_time', 'NDTI_std_time', 'NDFI_mean_time', 'NDTI_mean_time']

    t3_365_agg = spatial_aggregation(masked_365_t3[slice_365day_t3], 5000)
    t2_365_agg = spatial_aggregation(masked_365_t2[slice_365day_t2], 500)

    t3_365_sliced_means = t3_365_agg[0][slice_365day_t3]
    t2_365_sliced_means = t2_365_agg[0][slice_365day_t2]

    t3_365_sliced_std = t3_365_agg[1][slice_365day_t3]
    t2_365_sliced_std = t2_365_agg[1][slice_365day_t2]

    t3_365_sliced_means['t3_365vp'] = t3_365_agg[2]['red']
    t2_365_sliced_means['t2_365vp'] = t2_365_agg[2]['red']

    valid_pixels_365 = xr.Dataset({'t2_365_vp': t2_365_sliced_means['t2_365vp'], 't3_365_vp': t3_365_sliced_means['t3_365vp']})

    t2_365_sliced_means = t2_365_sliced_means.rename(
        {var: f"t2_365_{var}_mean" for var in t2_365_sliced_means.data_vars}
    )

    t3_365_sliced_means = t3_365_sliced_means.rename(
        {var: f"t3_365_{var}_mean" for var in t3_365_sliced_means.data_vars}
    )

    t3_365_sliced_std = t3_365_sliced_std.rename(
        {var: f"t3_365_{var}_std" for var in t3_365_sliced_std.data_vars}
    )

    t2_365_sliced_std = t2_365_sliced_std.rename(
        {var: f"t2_365_{var}_std" for var in t2_365_sliced_std.data_vars}
    )

    water_frac_t2d.name = 'qa_pixel_t2d_waterfrac'
    water_frac_t3d.name = 'qa_pixel_t3d_waterfrac'

    water_frac_ds_year = xr.Dataset({'t3d_waterfrac': water_frac_t3d, 't2d_waterfrac': water_frac_t2d})

    year_data = [t2_365_sliced_means, t3_365_sliced_means, t2_365_sliced_std, t3_365_sliced_std, water_frac_ds_year,valid_pixels_365]

    return year_data


def pc_query(sample):
    station_coords = (sample['Latitude'], sample['Longitude'])
    dates = sample['Sample Date dt']
    bb = calculate_bb(station_coords, 5000)
    min_date = (dates - datetime.timedelta(days=365)).strftime('%Y-%m-%d')
    sampling_period = f"{min_date}/{dates.strftime('%Y-%m-%d')}"
    search = catalog.search(collections=["landsat-c2-l2"], bbox=bb, datetime=sampling_period, query={"platform": {"in": ["landsat-8"]}, "eo:cloud_cover": {"lt": 60}})
    items = search.item_collection()
    if not items:
        return pd.Series({
            "nir08": np.nan, "green": np.nan, "swir16": np.nan, "swir22": np.nan, 'red': np.nan,
            'blue': np.nan, 'QA_PIXEL':np.nan, 'lwir11':np.nan
        })
    print("query completed")
    all_time_station_data = load_data(items, bb)
    all_time_station_data = offset_scale(all_time_station_data)

    window_365_start = dates - datetime.timedelta(days=365)
    window_90_start = dates - datetime.timedelta(days=90)
    data_365 = all_time_station_data.sel(time=slice(window_365_start, dates))
    data_90_5000m = all_time_station_data.sel(time=slice(window_90_start, dates))
    feats_90day = pipeline_90day(data_90_5000m, station_coords, bb, dates)
    feats_365day = pipeline_365day(data_365, station_coords)
    sample_level_data = xr.merge(feats_90day + feats_365day).to_array().to_series()
    print('sample_loaded')
    return sample_level_data


@delayed
def split_loading(xarray_row):
    return xarray_row


if __name__ == "__main__":
    # cluster = LocalCluster(n_workers=8, threads_per_worker=4)
    # client = Client(cluster)
    # print(client.dashboard_link)

    # catalog = pystac_client.Client.open(
    #     "https://planetarycomputer.microsoft.com/api/stac/v1",
    #     modifier=planetary_computer.sign_inplace,
    # )

    train_set = pd.read_csv(
        '/Users/neilkumar/Desktop/Python/EY_DATA_CHALLENGE/data_challenge_train_test_sets/water_quality_training_dataset.csv')
    train_set['Sample Date dt'] = pd.to_datetime(train_set['Sample Date'], dayfirst=True)
    train_set['Latitude'] = train_set['Latitude'].round(6)
    train_set['Longitude'] = train_set['Longitude'].round(6)

    bit_flags = {
        'fill': 1 << 0,
        'dilated_cloud': 1 << 1,
        'cirrus': 1 << 2,
        'cloud': 1 << 3,
        'shadow': 1 << 4,
        'snow': 1 << 5,
        'water': 1 << 7
    }

    station_list = []

    for i in range(4000, 4100):
        row = delayed(pc_query)(train_set.loc[i])
        station_list.append(row)

    print('end of lazy loading')
    print("starting parallelism")
    compute_time = time.perf_counter()
    station_batch = compute(*station_list)
    compute_time_end = time.perf_counter()
    print("end of parallelism")
    print(f"\n\n{compute_time_end - compute_time} for 100 rows\n\n")
    final_ds_df = pd.DataFrame(station_batch)
    final_ds_df.to_csv('landsat_fixed_5700-5750.csv')






