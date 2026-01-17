"""
Coverage and access window calculations for EO sensors.

Handles target access computation, look angle calculations, and swath geometry.
"""

import json
import warnings
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

import numpy as np
import pandas as pd
import geopandas as gpd
from pyproj import Geod
from shapely.geometry import Point, Polygon, LineString
from shapely.prepared import prep

from .orbits import TLEData, calculate_ground_track


def calculate_look_angle(
    sat_lats: np.ndarray,
    sat_lons: np.ndarray,
    sat_alts: np.ndarray,
    target_lat: float,
    target_lon: float,
) -> np.ndarray:
    """
    Calculate look angles (elevation from target) between satellite positions and a target.

    Accounts for Earth's curvature and horizon obstruction.

    Parameters
    ----------
    sat_lats : array-like
        Satellite latitudes in degrees.
    sat_lons : array-like
        Satellite longitudes in degrees.
    sat_alts : array-like
        Satellite altitudes in km.
    target_lat : float
        Target latitude in degrees.
    target_lon : float
        Target longitude in degrees.

    Returns
    -------
    np.ndarray
        Look angles in degrees (NaN if target beyond horizon).
    """
    sat_lats = np.atleast_1d(sat_lats)
    sat_lons = np.atleast_1d(sat_lons)
    sat_alts = np.atleast_1d(sat_alts)

    # Convert to radians
    sat_lats_rad = np.radians(sat_lats)
    sat_lons_rad = np.radians(sat_lons)
    target_lat_rad = np.radians(target_lat)
    target_lon_rad = np.radians(target_lon)

    # Earth radius in km
    R = 6378.137

    # Great circle distance using Haversine formula
    dlat = sat_lats_rad - target_lat_rad
    dlon = sat_lons_rad - target_lon_rad

    a = np.sin(dlat / 2)**2 + np.cos(sat_lats_rad) * np.cos(target_lat_rad) * np.sin(dlon / 2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    ground_distance = R * c

    # Horizon distance for each satellite position
    horizon_distance = R * np.arccos(R / (R + sat_alts))

    # Mask points beyond horizon
    beyond_horizon = ground_distance > horizon_distance

    # Calculate look angle (elevation from target perspective)
    look_angles = np.degrees(np.arctan2(sat_alts, ground_distance))
    look_angles = np.where(beyond_horizon, np.nan, look_angles)

    if len(look_angles) == 1:
        return float(look_angles[0])
    return look_angles


def calculate_off_nadir_angle(look_angle: float) -> float:
    """
    Convert look angle to off-nadir angle.

    Parameters
    ----------
    look_angle : float
        Look angle in degrees (elevation from target).

    Returns
    -------
    float
        Off-nadir angle in degrees.
    """
    return 90.0 - look_angle


def load_targets(
    geojson_path: Optional[str] = None,
    target_points: Optional[List[Tuple[float, float]]] = None,
    use_grid: bool = False,
    grid_params: Optional[Dict[str, float]] = None,
) -> gpd.GeoDataFrame:
    """
    Load target AOIs from various sources.

    Parameters
    ----------
    geojson_path : str, optional
        Path to GeoJSON file with target points/polygons.
    target_points : List[Tuple], optional
        List of (lat, lon) tuples for target points.
    use_grid : bool
        Whether to generate a grid of targets.
    grid_params : Dict, optional
        Grid parameters: lat_min, lat_max, lon_min, lon_max, spacing_deg.

    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame with target geometries.
    """
    if geojson_path and Path(geojson_path).exists():
        with open(geojson_path, 'r') as f:
            geojson_data = json.load(f)
        return gpd.GeoDataFrame.from_features(geojson_data['features'], crs='EPSG:4326')

    points = []

    if target_points:
        for lat, lon in target_points:
            points.append({'geometry': Point(lon, lat), 'lat': lat, 'lon': lon})

    if use_grid and grid_params:
        lat_min = grid_params.get('lat_min', -60)
        lat_max = grid_params.get('lat_max', 60)
        lon_min = grid_params.get('lon_min', -180)
        lon_max = grid_params.get('lon_max', 180)
        spacing = grid_params.get('spacing_deg', 5.0)

        lats = np.arange(lat_min, lat_max + spacing, spacing)
        lons = np.arange(lon_min, lon_max + spacing, spacing)

        for lat in lats:
            for lon in lons:
                points.append({'geometry': Point(lon, lat), 'lat': lat, 'lon': lon})

    if not points:
        # Default: single target at origin
        points.append({'geometry': Point(0, 0), 'lat': 0, 'lon': 0})

    return gpd.GeoDataFrame(points, crs='EPSG:4326')


def convert_wgs_to_utm(lon: float, lat: float) -> int:
    """
    Convert WGS84 coordinates to UTM EPSG code.

    Parameters
    ----------
    lon : float
        Longitude in degrees.
    lat : float
        Latitude in degrees.

    Returns
    -------
    int
        UTM EPSG code.
    """
    utm_zone = int((lon + 180) / 6) + 1
    if lat >= 0:
        return 32600 + utm_zone
    return 32700 + utm_zone


def _la_to_swath_distance(altitude_m: float, look_angle_deg: float) -> Tuple[float, float]:
    """
    Convert look angle to swath distance on ground.

    Parameters
    ----------
    altitude_m : float
        Satellite altitude in meters.
    look_angle_deg : float
        Look angle in degrees.

    Returns
    -------
    Tuple[float, float]
        (swath_distance_m, incidence_angle_deg)
    """
    from math import asin, sin, degrees, radians

    radius_earth = 6378145  # meters

    # Handle edge cases
    if look_angle_deg >= 90:
        # Nadir pointing - swath distance is 0
        return 0.0, 0.0
    if look_angle_deg <= 0:
        return 0.0, 0.0

    la_rad = radians(look_angle_deg)

    # Check domain for asin
    arg = (1 + altitude_m / radius_earth) * sin(la_rad)
    if arg > 1.0:
        arg = 1.0  # Clamp to valid range

    ia = asin(arg)
    theta = ia - la_rad
    swath_dist = theta * radius_earth

    return swath_dist, degrees(ia)


def calculate_swath_polygon(
    ground_track: List[Tuple[float, float]],
    altitude_km: float,
    off_nadir_min_deg: float,
    off_nadir_max_deg: float,
    target_lon: float,
    target_lat: float,
) -> Optional[Polygon]:
    """
    Calculate the access swath polygon for a satellite pass.

    Parameters
    ----------
    ground_track : List[Tuple]
        List of (lon, lat) points representing the ground track.
    altitude_km : float
        Satellite altitude in km.
    off_nadir_min_deg : float
        Minimum off-nadir angle in degrees.
    off_nadir_max_deg : float
        Maximum off-nadir angle in degrees.
    target_lon : float
        Target longitude (for UTM zone selection).
    target_lat : float
        Target latitude (for UTM zone selection).

    Returns
    -------
    Optional[Polygon]
        Swath polygon or None if calculation fails.
    """
    if len(ground_track) < 2:
        return None

    try:
        line = LineString(ground_track)
        epsg_code = convert_wgs_to_utm(target_lon, target_lat)

        altitude_m = altitude_km * 1000

        # Convert off-nadir to look angle (look = 90 - off_nadir)
        la_min = 90 - off_nadir_max_deg  # Look angle at max off-nadir
        la_max = 90 - off_nadir_min_deg  # Look angle at min off-nadir

        far_edge_dist, _ = _la_to_swath_distance(altitude_m, la_min)
        far_edge = gpd.GeoSeries(line, crs=4326).to_crs(epsg_code).buffer(far_edge_dist, cap_style=2).to_crs(4326)

        # If off_nadir_min is 0, we include nadir, so no inner buffer needed
        if off_nadir_min_deg <= 0.5:
            swath = far_edge
        else:
            near_edge_dist, _ = _la_to_swath_distance(altitude_m, la_max)
            near_edge = gpd.GeoSeries(line, crs=4326).to_crs(epsg_code).buffer(near_edge_dist).to_crs(4326)
            swath = far_edge.difference(near_edge, align=False)

        if swath.empty or swath.iloc[0].is_empty:
            return None

        return swath.iloc[0]

    except Exception as e:
        warnings.warn(f"Swath calculation error: {e}")
        return None


def calculate_access_windows(
    orbit_df: pd.DataFrame,
    targets_gdf: gpd.GeoDataFrame,
    off_nadir_min_deg: float = 0.0,
    off_nadir_max_deg: float = 45.0,
    min_access_duration_s: float = 45.0,
    pass_type: str = "both",
) -> pd.DataFrame:
    """
    Calculate target access windows from orbit data.

    Parameters
    ----------
    orbit_df : pd.DataFrame
        Propagated orbit data with columns: Satellite, Epoch, Latitude, Longitude, Altitude_km
    targets_gdf : gpd.GeoDataFrame
        Target AOIs as points.
    off_nadir_min_deg : float
        Minimum off-nadir angle in degrees.
    off_nadir_max_deg : float
        Maximum off-nadir angle in degrees.
    min_access_duration_s : float
        Minimum access duration to keep in seconds.
    pass_type : str
        Filter by pass type: "ascending" (northbound), "descending" (southbound), or "both".
        For LTDN-defined orbits, imaging typically occurs on descending passes.

    Returns
    -------
    pd.DataFrame
        DataFrame with access windows.
    """
    access_times = []

    for idx, aoi in targets_gdf.iterrows():
        target_lat = aoi.geometry.y
        target_lon = aoi.geometry.x

        # Calculate look angles for all orbit points
        look_angles = calculate_look_angle(
            orbit_df['Latitude'].values,
            orbit_df['Longitude'].values,
            orbit_df['Altitude_km'].values,
            target_lat,
            target_lon,
        )

        off_nadir_angles = 90 - look_angles

        # Filter by off-nadir constraints
        mask = (off_nadir_angles >= off_nadir_min_deg) & (off_nadir_angles <= off_nadir_max_deg)
        filtered_df = orbit_df[mask].copy()
        filtered_df['off_nadir_angle'] = off_nadir_angles[mask]

        if filtered_df.empty:
            continue

        # Group consecutive points into overpasses
        # Use time difference to detect gaps (more than 2x timestep = new pass)
        filtered_df = filtered_df.sort_values(['Satellite', 'Epoch'])
        filtered_df['time_diff'] = filtered_df.groupby('Satellite')['Epoch'].diff()

        # Detect overpass boundaries
        median_step = filtered_df['time_diff'].median()
        if pd.isna(median_step):
            median_step = timedelta(seconds=10)

        gap_threshold = median_step * 3
        filtered_df['new_pass'] = (
            (filtered_df['time_diff'] > gap_threshold) |
            (filtered_df['time_diff'].isna())
        )
        filtered_df['Overpass'] = filtered_df.groupby('Satellite')['new_pass'].cumsum()

        # Filter out single-point passes
        pass_sizes = filtered_df.groupby(['Satellite', 'Overpass']).size()
        valid_passes = pass_sizes[pass_sizes > 1].index
        filtered_df = filtered_df.set_index(['Satellite', 'Overpass'])
        filtered_df = filtered_df.loc[filtered_df.index.isin(valid_passes)].reset_index()

        if filtered_df.empty:
            continue

        # Aggregate to get start/end times
        overpass_agg = filtered_df.groupby(['Satellite', 'Overpass']).agg({
            'Epoch': ['min', 'max'],
            'off_nadir_angle': ['min', 'max'],
            'Latitude': ['first', 'last'],
            'Longitude': 'first',
            'Altitude_km': 'mean',
        }).reset_index()

        overpass_agg.columns = [
            'Satellite', 'Overpass', 'Start_Time', 'End_Time',
            'OffNadir_Min', 'OffNadir_Max', 'Sat_Lat_Start', 'Sat_Lat_End', 'Sat_Lon', 'Altitude_km'
        ]

        # Determine pass direction: descending = latitude decreasing (moving south)
        overpass_agg['Lat_Change'] = overpass_agg['Sat_Lat_End'] - overpass_agg['Sat_Lat_Start']
        overpass_agg['Pass_Direction'] = np.where(
            overpass_agg['Lat_Change'] < 0, 'descending', 'ascending'
        )
        overpass_agg['Sat_Lat'] = overpass_agg['Sat_Lat_Start']  # Use start lat for compatibility

        overpass_agg['AOI_Lat'] = target_lat
        overpass_agg['AOI_Lon'] = target_lon
        overpass_agg['Access_Duration'] = (
            overpass_agg['End_Time'] - overpass_agg['Start_Time']
        ).dt.total_seconds()

        # Filter by minimum duration
        overpass_agg = overpass_agg[overpass_agg['Access_Duration'] >= min_access_duration_s]

        # Filter by pass type if specified
        if pass_type == "descending":
            overpass_agg = overpass_agg[overpass_agg['Pass_Direction'] == 'descending']
        elif pass_type == "ascending":
            overpass_agg = overpass_agg[overpass_agg['Pass_Direction'] == 'ascending']
        # else "both" - keep all passes

        # Drop helper columns
        overpass_agg = overpass_agg.drop(columns=['Sat_Lat_Start', 'Sat_Lat_End', 'Lat_Change'])

        access_times.append(overpass_agg)

    if not access_times:
        return pd.DataFrame()

    access_df = pd.concat(access_times, ignore_index=True)
    access_df = access_df.sort_values('Start_Time').reset_index(drop=True)

    return access_df


def filter_access_by_swath(
    access_df: pd.DataFrame,
    tle_data: Dict[int, TLEData],
    off_nadir_min_deg: float = 0.0,
    off_nadir_max_deg: float = 45.0,
) -> Tuple[pd.DataFrame, gpd.GeoDataFrame]:
    """
    Filter access windows by checking if target falls within sensor swath.

    Parameters
    ----------
    access_df : pd.DataFrame
        Access windows from calculate_access_windows.
    tle_data : Dict[int, TLEData]
        TLE data dictionary.
    off_nadir_min_deg : float
        Minimum off-nadir angle.
    off_nadir_max_deg : float
        Maximum off-nadir angle.

    Returns
    -------
    Tuple[pd.DataFrame, gpd.GeoDataFrame]
        Filtered access DataFrame and swath geometries.
    """
    valid_indices = []
    swaths = []

    for idx, access in access_df.iterrows():
        try:
            sat_id = access['Satellite']
            start_time = access['Start_Time']
            duration_s = access['Access_Duration']

            # Calculate ground track
            ground_track = calculate_ground_track(
                tle_data,
                sat_id,
                start_time.to_pydatetime() if hasattr(start_time, 'to_pydatetime') else start_time,
                duration=timedelta(seconds=duration_s),
                step_s=10.0,
            )

            if len(ground_track) < 2:
                continue

            # Calculate swath polygon
            swath = calculate_swath_polygon(
                ground_track,
                access['Altitude_km'],
                off_nadir_min_deg,
                off_nadir_max_deg,
                access['AOI_Lon'],
                access['AOI_Lat'],
            )

            if swath is None:
                continue

            # Check if target is within swath
            target_point = Point(access['AOI_Lon'], access['AOI_Lat'])
            if swath.contains(target_point):
                valid_indices.append(idx)
                swaths.append({
                    'access_idx': idx,
                    'Satellite': sat_id,
                    'Start_Time': start_time,
                    'geometry': swath,
                })

        except Exception as e:
            warnings.warn(f"Failed to calculate swath for access {idx}: {e}")
            continue

    filtered_df = access_df.loc[valid_indices].copy()

    swaths_gdf = gpd.GeoDataFrame(swaths, crs='EPSG:4326') if swaths else gpd.GeoDataFrame()

    return filtered_df, swaths_gdf


def calculate_mid_point(
    access_df: pd.DataFrame,
    tle_data: Dict[int, TLEData],
) -> pd.DataFrame:
    """
    Add midpoint time and position to access DataFrame.

    Parameters
    ----------
    access_df : pd.DataFrame
        Access windows DataFrame.
    tle_data : Dict[int, TLEData]
        TLE data dictionary.

    Returns
    -------
    pd.DataFrame
        Access DataFrame with Mid_Time, Mid_Lat, Mid_Lon columns added.
    """
    from .orbits import get_satellite_position

    access_df = access_df.copy()
    access_df['Mid_Time'] = access_df['Start_Time'] + pd.to_timedelta(access_df['Access_Duration'] / 2, unit='s')

    mid_lats = []
    mid_lons = []

    for _, access in access_df.iterrows():
        try:
            mid_time = access['Mid_Time']
            if hasattr(mid_time, 'to_pydatetime'):
                mid_time = mid_time.to_pydatetime()

            lat, lon, _ = get_satellite_position(tle_data, access['Satellite'], mid_time)
            mid_lats.append(lat)
            mid_lons.append(lon)
        except Exception:
            mid_lats.append(np.nan)
            mid_lons.append(np.nan)

    access_df['Mid_Lat'] = mid_lats
    access_df['Mid_Lon'] = mid_lons

    return access_df


def validate_coverage(
    access_df: pd.DataFrame,
    config_duration_days: int,
) -> Dict[str, Any]:
    """
    Run sanity checks on coverage/access data.

    Parameters
    ----------
    access_df : pd.DataFrame
        Access windows DataFrame.
    config_duration_days : int
        Expected analysis duration in days.

    Returns
    -------
    Dict[str, Any]
        Validation results.
    """
    results = {
        'passed': True,
        'errors': [],
        'warnings': [],
        'stats': {},
    }

    if access_df.empty:
        results['warnings'].append("No access windows found")
        results['stats']['total_accesses'] = 0
        return results

    results['stats']['total_accesses'] = len(access_df)
    results['stats']['num_satellites'] = access_df['Satellite'].nunique()
    results['stats']['num_targets'] = access_df.groupby(['AOI_Lat', 'AOI_Lon']).ngroups

    # Check access counts are plausible
    accesses_per_day = len(access_df) / config_duration_days
    results['stats']['accesses_per_day'] = accesses_per_day

    if accesses_per_day < 0.1:
        results['warnings'].append(f"Very few accesses: {accesses_per_day:.2f}/day")
    elif accesses_per_day > 1000:
        results['warnings'].append(f"Unusually high access count: {accesses_per_day:.1f}/day")

    # Check duration statistics
    mean_duration = access_df['Access_Duration'].mean()
    max_duration = access_df['Access_Duration'].max()
    results['stats']['mean_duration_s'] = mean_duration
    results['stats']['max_duration_s'] = max_duration

    if max_duration > 600:  # 10 minutes
        results['warnings'].append(f"Long access duration detected: {max_duration:.0f}s")

    # Revisit statistics
    if results['stats']['num_targets'] > 0:
        revisits = access_df.groupby(['AOI_Lat', 'AOI_Lon']).size()
        results['stats']['mean_revisits'] = revisits.mean()
        results['stats']['min_revisits'] = revisits.min()
        results['stats']['max_revisits'] = revisits.max()

    print(f"Coverage validation: {results['stats']['total_accesses']} accesses")
    print(f"  {results['stats']['num_satellites']} satellites, {results['stats']['num_targets']} targets")
    print(f"  {accesses_per_day:.1f} accesses/day, mean duration {mean_duration:.0f}s")

    if results['errors']:
        print(f"  ERRORS: {results['errors']}")
    if results['warnings']:
        print(f"  Warnings: {results['warnings']}")

    return results
