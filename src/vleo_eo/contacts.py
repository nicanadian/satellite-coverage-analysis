"""
Ground station contact window calculations and TTNC (Time To Next Contact) metrics.

Handles communication cone geometry, contact finding, and downlink scheduling.
"""

import warnings
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import geopandas as gpd
from pyproj import Geod
from shapely.geometry import Point, Polygon
from shapely.prepared import prep
from skyfield.api import load, EarthSatellite, wgs84

from .orbits import TLEData
from .config import GroundStationConfig, ImagingModeConfig


def calculate_ground_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate great circle distance between two points using Haversine formula.

    Parameters
    ----------
    lat1, lon1 : float
        First point coordinates in degrees.
    lat2, lon2 : float
        Second point coordinates in degrees.

    Returns
    -------
    float
        Distance in kilometers.
    """
    R = 6371.0  # Earth radius in km

    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)

    a = np.sin(dlat / 2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return R * c


def calculate_comm_range_km(min_elevation_deg: float, sat_altitude_km: float) -> float:
    """
    Calculate maximum ground range for communication given elevation and altitude.

    Parameters
    ----------
    min_elevation_deg : float
        Minimum elevation angle in degrees.
    sat_altitude_km : float
        Satellite altitude in km.

    Returns
    -------
    float
        Maximum ground range in km.
    """
    earth_radius_km = 6378.137
    elevation_rad = np.radians(min_elevation_deg)

    # Central angle calculation
    central_angle = np.arccos(
        earth_radius_km * np.cos(elevation_rad) / (earth_radius_km + sat_altitude_km)
    ) - elevation_rad

    return earth_radius_km * central_angle


def create_communication_cone(
    lon: float,
    lat: float,
    min_elevation_deg: float,
    sat_altitude_m: float = 600000,
) -> Polygon:
    """
    Create a communication cone (coverage footprint) for a ground station.

    Parameters
    ----------
    lon : float
        Ground station longitude in degrees.
    lat : float
        Ground station latitude in degrees.
    min_elevation_deg : float
        Minimum elevation angle in degrees.
    sat_altitude_m : float
        Satellite altitude in meters (default 600 km).

    Returns
    -------
    Polygon
        Communication cone polygon in WGS84 coordinates.
    """
    geod = Geod(ellps='WGS84')
    earth_radius = 6378137  # meters

    # Calculate maximum ground range from geometry
    elevation_rad = np.radians(min_elevation_deg)

    # Central angle calculation
    central_angle = np.arccos(
        earth_radius * np.cos(elevation_rad) / (earth_radius + sat_altitude_m)
    ) - elevation_rad

    max_ground_range = earth_radius * central_angle

    # Create polygon by projecting points at various azimuths
    azimuths = np.linspace(0, 360, 360)
    lats = []
    lons_out = []

    for azimuth in azimuths:
        lon2, lat2, _ = geod.fwd(float(lon), float(lat), azimuth, max_ground_range)

        # Handle longitude wrapping
        if lon > 0:
            if lon2 < 0:
                lon2 += 360
        else:
            if lon2 > 180:
                lon2 -= 360

        lats.append(lat2)
        lons_out.append(lon2)

    # Close the polygon
    lats.append(lats[0])
    lons_out.append(lons_out[0])

    return Polygon(zip(lons_out, lats))


def find_next_contact(
    tle_data: Dict[int, TLEData],
    sat_id: int,
    start_time: datetime,
    comm_cone: Polygon,
    max_search_time: timedelta = timedelta(hours=3),
    time_step_s: float = 15.0,
    gs_lat: float = None,
    gs_lon: float = None,
    min_elevation_deg: float = 5.0,
) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    """
    Find the next ground station contact window for a satellite.

    Uses distance-based validation to avoid polygon containment errors.

    Parameters
    ----------
    tle_data : Dict[int, TLEData]
        TLE data dictionary.
    sat_id : int
        Satellite ID.
    start_time : datetime
        Start time to search from (must be timezone-aware).
    comm_cone : Polygon
        Communication cone geometry (used as fallback).
    max_search_time : timedelta
        Maximum time to search for contact.
    time_step_s : float
        Time step for searching in seconds.
    gs_lat : float, optional
        Ground station latitude (required for distance-based validation).
    gs_lon : float, optional
        Ground station longitude (required for distance-based validation).
    min_elevation_deg : float
        Minimum elevation angle for contact.

    Returns
    -------
    Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]
        (contact_start, contact_end) or (None, None) if no contact found.
    """
    if start_time.tzinfo is None:
        raise ValueError("start_time must be timezone-aware")

    if sat_id not in tle_data:
        return (None, None)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # Create time range
        start_ts = pd.Timestamp(start_time).floor('s')
        end_ts = (start_ts + max_search_time).floor('s')
        times = pd.date_range(start=start_ts, end=end_ts, freq=f'{int(time_step_s)}s')

        if times.tz is None:
            times = times.tz_localize('UTC')
        elif str(times.tz) != 'UTC':
            times = times.tz_convert('UTC')

        ts = load.timescale()
        t = ts.from_datetimes([t.to_pydatetime() for t in times])

    # Propagate satellite
    tle = tle_data[sat_id]
    sat = EarthSatellite(tle.line1, tle.line2, str(sat_id), ts)
    positions = sat.at(t)

    subpoints = wgs84.geographic_position_of(positions)
    lats = subpoints.latitude.degrees
    lons = subpoints.longitude.degrees

    # Use distance-based validation if ground station coordinates provided
    if gs_lat is not None and gs_lon is not None:
        # Calculate max comm range based on satellite altitude and min elevation
        sat_alt_km = tle.altitude_km
        max_range_km = calculate_comm_range_km(min_elevation_deg, sat_alt_km)

        # Calculate distance from each satellite position to ground station
        in_range = np.zeros(len(lats), dtype=bool)
        for i in range(len(lats)):
            dist_km = calculate_ground_distance_km(lats[i], lons[i], gs_lat, gs_lon)
            in_range[i] = dist_km <= max_range_km
    else:
        # Fallback to polygon containment (less reliable)
        prepared_cone = prep(comm_cone)
        coords = np.column_stack((lons, lats)).astype(np.float64)
        points = [Point(x, y) for x, y in coords]
        in_range = np.array([prepared_cone.contains(point) for point in points])

    # Find contact periods
    contacts = []
    in_contact = False
    contact_start_time = None

    for idx, val in enumerate(in_range):
        if val and not in_contact:
            contact_start_time = times[idx]
            in_contact = True
        elif not val and in_contact:
            contact_end_time = times[idx]
            contacts.append((contact_start_time, contact_end_time))
            in_contact = False

    if in_contact:
        contact_end_time = times[-1]
        contacts.append((contact_start_time, contact_end_time))

    if not contacts:
        return (None, None)

    return contacts[0]


def calculate_raw_contact_windows(
    tle_data: Dict[int, TLEData],
    ground_stations: List[GroundStationConfig],
    config: 'AnalysisConfig',
) -> pd.DataFrame:
    """
    Calculate all contact windows for satellites and ground stations.

    Simplified version for optimization that doesn't require access windows.

    Parameters
    ----------
    tle_data : Dict[int, TLEData]
        TLE data dictionary.
    ground_stations : List[GroundStationConfig]
        Ground station configurations.
    config : AnalysisConfig
        Analysis configuration.

    Returns
    -------
    pd.DataFrame
        Contact windows with columns: sat_id, gs_name, contact_start, contact_end.
    """
    from .config import AnalysisConfig

    ts = load.timescale()

    # Generate time samples
    start_dt = config.start_datetime
    end_dt = config.end_datetime

    # Use coarser time step for contact detection (saves computation)
    time_step_s = max(config.time_step_s, 30.0)

    times = pd.date_range(start=start_dt, end=end_dt, freq=f'{int(time_step_s)}s')
    if times.tz is None:
        times = times.tz_localize('UTC')

    t = ts.from_datetimes([t.to_pydatetime() for t in times])

    contacts = []

    for sat_id, tle in tle_data.items():
        sat = EarthSatellite(tle.line1, tle.line2, str(sat_id), ts)
        positions = sat.at(t)
        subpoints = wgs84.geographic_position_of(positions)
        lats = subpoints.latitude.degrees
        lons = subpoints.longitude.degrees

        for gs in ground_stations:
            # Use distance-based validation for accurate contact detection
            sat_alt_km = tle.altitude_km
            max_range_km = calculate_comm_range_km(gs.min_elevation_deg, sat_alt_km)

            # Calculate distance from each satellite position to ground station
            in_range = np.zeros(len(lats), dtype=bool)
            for i in range(len(lats)):
                dist_km = calculate_ground_distance_km(lats[i], lons[i], gs.lat, gs.lon)
                in_range[i] = dist_km <= max_range_km

            # Find contact periods
            in_contact = False
            contact_start_time = None

            for idx, val in enumerate(in_range):
                if val and not in_contact:
                    contact_start_time = times[idx]
                    in_contact = True
                elif not val and in_contact:
                    contact_end_time = times[idx]
                    contacts.append({
                        'sat_id': sat_id,
                        'gs_name': gs.name,
                        'contact_start': contact_start_time,
                        'contact_end': contact_end_time,
                    })
                    in_contact = False

            if in_contact:
                contact_end_time = times[-1]
                contacts.append({
                    'sat_id': sat_id,
                    'gs_name': gs.name,
                    'contact_start': contact_start_time,
                    'contact_end': contact_end_time,
                })

    return pd.DataFrame(contacts)


def calculate_slew_duration(
    tle_data: Dict[int, TLEData],
    sat_id: int,
    imaging_stop_time: datetime,
    target_lat: float,
    target_lon: float,
    gs_lat: float,
    gs_lon: float,
    slew_rate_deg_per_s: float,
    mode_switch_time_s: float,
) -> float:
    """
    Calculate slew duration from imaging target to ground station pointing.

    Parameters
    ----------
    tle_data : Dict[int, TLEData]
        TLE data dictionary.
    sat_id : int
        Satellite ID.
    imaging_stop_time : datetime
        Time when imaging stops.
    target_lat, target_lon : float
        Target coordinates.
    gs_lat, gs_lon : float
        Ground station coordinates.
    slew_rate_deg_per_s : float
        Antenna slew rate in degrees per second.
    mode_switch_time_s : float
        Minimum mode switch time in seconds.

    Returns
    -------
    float
        Slew duration in seconds.
    """
    from .orbits import get_satellite_position

    sat_lat, sat_lon, _ = get_satellite_position(tle_data, sat_id, imaging_stop_time)

    def to_unit_vector(lat: float, lon: float) -> np.ndarray:
        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)
        x = np.cos(lat_rad) * np.cos(lon_rad)
        y = np.cos(lat_rad) * np.sin(lon_rad)
        z = np.sin(lat_rad)
        return np.array([x, y, z])

    sat_to_target = to_unit_vector(target_lat, target_lon) - to_unit_vector(sat_lat, sat_lon)
    sat_to_gs = to_unit_vector(gs_lat, gs_lon) - to_unit_vector(sat_lat, sat_lon)

    sat_to_target = sat_to_target / np.linalg.norm(sat_to_target)
    sat_to_gs = sat_to_gs / np.linalg.norm(sat_to_gs)

    angle_rad = np.arccos(np.clip(np.dot(sat_to_target, sat_to_gs), -1.0, 1.0))
    angle_deg = np.degrees(angle_rad)

    slew_time = angle_deg / slew_rate_deg_per_s

    return max(slew_time, mode_switch_time_s)


def calculate_contact_windows(
    access_df: pd.DataFrame,
    tle_data: Dict[int, TLEData],
    ground_stations: List[GroundStationConfig],
    imaging_modes: List[ImagingModeConfig],
    slew_rate_deg_per_s: float = 0.6,
    mode_switch_time_s: float = 60.0,
    contact_buffer_s: float = 30.0,
    ttnc_max_search_hours: float = 3.0,
) -> Dict[str, pd.DataFrame]:
    """
    Calculate contact windows and delivery timelines for all imaging modes.

    Parameters
    ----------
    access_df : pd.DataFrame
        Access windows DataFrame.
    tle_data : Dict[int, TLEData]
        TLE data dictionary.
    ground_stations : List[GroundStationConfig]
        Ground station configurations.
    imaging_modes : List[ImagingModeConfig]
        Imaging mode configurations.
    slew_rate_deg_per_s : float
        Slew rate in degrees per second.
    mode_switch_time_s : float
        Mode switch time in seconds.
    contact_buffer_s : float
        Buffer time for contact validation.
    ttnc_max_search_hours : float
        Maximum search window for TTNC in hours.

    Returns
    -------
    Dict[str, pd.DataFrame]
        Dictionary mapping imaging mode names to processed DataFrames.
    """
    # Pre-compute communication cones for each ground station and satellite
    comm_cones = {}
    gs_gdf_data = []

    for gs in ground_stations:
        gs_gdf_data.append({
            'name': gs.name,
            'lat': gs.lat,
            'lon': gs.lon,
            'geometry': Point(gs.lon, gs.lat),
            'min_elevation_deg': gs.min_elevation_deg,
            'ka_capable': gs.ka_capable,
            'downlink_rate_mbps': gs.downlink_rate_mbps,
        })

        for sat_id in access_df['Satellite'].unique():
            if sat_id not in tle_data:
                continue
            sat_alt_m = tle_data[sat_id].altitude_km * 1000
            cone = create_communication_cone(gs.lon, gs.lat, gs.min_elevation_deg, sat_alt_m)
            comm_cones[(gs.name, sat_id)] = cone

    gs_gdf = gpd.GeoDataFrame(gs_gdf_data, crs='EPSG:4326')

    mode_dfs = {}

    for mode in imaging_modes:
        print(f"\nProcessing contacts for imaging mode: {mode.name}")

        df = access_df.copy()

        # Calculate imaging parameters
        collect_dwell_time = mode.collect_dwell_time_s
        raw_data_size_gb = mode.collection_rate_gbps * collect_dwell_time
        processing_time_min = mode.processing_time_min

        # Find a representative downlink rate
        downlink_rate_mbps = ground_stations[0].downlink_rate_mbps if ground_stations else 800.0
        downlink_duration = (raw_data_size_gb * 8000) / downlink_rate_mbps  # seconds

        # Add mode-specific columns
        df['Imaging_Mode'] = mode.name
        df['Collect_Dwell_Time'] = collect_dwell_time
        df['Raw_Data_Size_GB'] = raw_data_size_gb
        df['Downlink_Duration'] = downlink_duration
        df['Processing_Time_Min'] = processing_time_min

        # Calculate imaging timeline for each access
        # Use start time as collect start (simplified from squint angle calculation)
        df['Squint_Start_Time'] = df['Start_Time']
        df['Imaging_Stop'] = df['Squint_Start_Time'] + pd.to_timedelta(collect_dwell_time, unit='s')

        # Initialize contact columns
        for gs in ground_stations:
            df[f'Next_{gs.name}_Contact_Start'] = pd.NaT
            df[f'Next_{gs.name}_Contact_End'] = pd.NaT
            df[f'Next_{gs.name}_Contact_Duration'] = np.nan
            df[f'Valid_Contact_{gs.name}'] = False

        # Process each access
        for idx, access in df.iterrows():
            sat_id = access['Satellite']

            # Calculate slew duration (use first ground station for estimation)
            if ground_stations:
                gs0 = ground_stations[0]
                try:
                    slew_duration = calculate_slew_duration(
                        tle_data, sat_id,
                        access['Imaging_Stop'].to_pydatetime() if hasattr(access['Imaging_Stop'], 'to_pydatetime') else access['Imaging_Stop'],
                        access['AOI_Lat'], access['AOI_Lon'],
                        gs0.lat, gs0.lon,
                        slew_rate_deg_per_s, mode_switch_time_s
                    )
                except Exception:
                    slew_duration = mode_switch_time_s
            else:
                slew_duration = mode_switch_time_s

            df.loc[idx, 'Slew_Duration'] = slew_duration
            df.loc[idx, 'Slew_Stop'] = access['Imaging_Stop'] + pd.to_timedelta(slew_duration, unit='s')
            df.loc[idx, 'Downlink_Stop'] = df.loc[idx, 'Slew_Stop'] + pd.to_timedelta(downlink_duration, unit='s')
            df.loc[idx, 'Processing_Stop'] = df.loc[idx, 'Downlink_Stop'] + pd.to_timedelta(processing_time_min * 60, unit='s')

            slew_stop_time = df.loc[idx, 'Slew_Stop']
            if pd.isna(slew_stop_time):
                continue

            slew_stop_dt = slew_stop_time.to_pydatetime() if hasattr(slew_stop_time, 'to_pydatetime') else slew_stop_time
            if slew_stop_dt.tzinfo is None:
                slew_stop_dt = slew_stop_dt.replace(tzinfo=timezone.utc)

            # Find contacts for each ground station
            for gs in ground_stations:
                try:
                    cone_key = (gs.name, sat_id)
                    if cone_key not in comm_cones:
                        continue

                    contact_start, contact_end = find_next_contact(
                        tle_data, sat_id, slew_stop_dt,
                        comm_cones[cone_key],
                        max_search_time=timedelta(hours=ttnc_max_search_hours),
                        gs_lat=gs.lat,
                        gs_lon=gs.lon,
                        min_elevation_deg=gs.min_elevation_deg,
                    )

                    if contact_start is not None:
                        df.loc[idx, f'Next_{gs.name}_Contact_Start'] = contact_start
                        df.loc[idx, f'Next_{gs.name}_Contact_End'] = contact_end
                        if contact_end:
                            duration = (contact_end - slew_stop_time).total_seconds()
                            df.loc[idx, f'Next_{gs.name}_Contact_Duration'] = duration

                except Exception as e:
                    warnings.warn(f"Error finding contact for {gs.name} at access {idx}: {e}")

        # Validate contacts
        for gs in ground_stations:
            downlink_stop = pd.to_datetime(df['Downlink_Stop'], utc=True)
            contact_end = pd.to_datetime(df[f'Next_{gs.name}_Contact_End'], utc=True)

            valid_rows = downlink_stop.notna() & contact_end.notna()
            df[f'Valid_Contact_{gs.name}'] = False
            if valid_rows.any():
                df.loc[valid_rows, f'Valid_Contact_{gs.name}'] = (
                    downlink_stop[valid_rows] < contact_end[valid_rows]
                )

        # Calculate total valid contacts
        valid_cols = [col for col in df.columns if col.startswith('Valid_Contact_')]
        df['Total_Valid_Contact'] = df[valid_cols].any(axis=1).astype(int)

        # Calculate collect-to-delivery timeline
        df['Collect_to_Delivery_Timeline'] = (
            df['Processing_Stop'] - df['Start_Time']
        ).dt.total_seconds()

        mode_dfs[mode.name] = df

        # Print statistics
        total = len(df)
        valid = df['Total_Valid_Contact'].sum()
        print(f"  {mode.name}: {valid}/{total} valid contacts ({valid/total*100:.1f}%)")

    return mode_dfs


def calculate_ttnc_ka(
    mode_dfs: Dict[str, pd.DataFrame],
    ground_stations: List[GroundStationConfig],
) -> pd.DataFrame:
    """
    Calculate Time To Next Ka Contact (TTNC) for each collection event.

    Parameters
    ----------
    mode_dfs : Dict[str, pd.DataFrame]
        Dictionary of mode DataFrames from calculate_contact_windows.
    ground_stations : List[GroundStationConfig]
        Ground station configurations (checks ka_capable flag).

    Returns
    -------
    pd.DataFrame
        DataFrame with TTNC metrics for each collection.
    """
    ka_stations = [gs.name for gs in ground_stations if gs.ka_capable]

    if not ka_stations:
        warnings.warn("No Ka-capable ground stations configured")
        return pd.DataFrame()

    ttnc_records = []

    for mode_name, df in mode_dfs.items():
        for idx, row in df.iterrows():
            # Use imaging stop time as collection end
            collect_end = row['Imaging_Stop']
            if pd.isna(collect_end):
                collect_end = row['End_Time']

            # Find earliest Ka contact
            earliest_ka_contact = None
            earliest_ka_station = None

            for gs_name in ka_stations:
                contact_start = row.get(f'Next_{gs_name}_Contact_Start')
                if pd.notna(contact_start):
                    if earliest_ka_contact is None or contact_start < earliest_ka_contact:
                        earliest_ka_contact = contact_start
                        earliest_ka_station = gs_name

            # Calculate TTNC
            if earliest_ka_contact is not None and pd.notna(collect_end):
                ttnc_seconds = (earliest_ka_contact - collect_end).total_seconds()
            else:
                ttnc_seconds = np.nan

            ttnc_records.append({
                'Access_ID': idx,
                'Imaging_Mode': mode_name,
                'Satellite': row['Satellite'],
                'Collect_End_Time': collect_end,
                'AOI_Lat': row['AOI_Lat'],
                'AOI_Lon': row['AOI_Lon'],
                'Next_Ka_Contact_Start': earliest_ka_contact,
                'Next_Ka_Station': earliest_ka_station,
                'TTNC_Ka_seconds': ttnc_seconds,
                'TTNC_Ka_minutes': ttnc_seconds / 60 if pd.notna(ttnc_seconds) else np.nan,
            })

    ttnc_df = pd.DataFrame(ttnc_records)

    # Calculate summary statistics
    if not ttnc_df.empty and not ttnc_df['TTNC_Ka_minutes'].isna().all():
        print("\nTTNC Ka Statistics (minutes):")
        print(f"  Median: {ttnc_df['TTNC_Ka_minutes'].median():.1f}")
        print(f"  P90: {ttnc_df['TTNC_Ka_minutes'].quantile(0.90):.1f}")
        print(f"  P95: {ttnc_df['TTNC_Ka_minutes'].quantile(0.95):.1f}")
        print(f"  Missing: {ttnc_df['TTNC_Ka_minutes'].isna().sum()}/{len(ttnc_df)}")

    return ttnc_df


def validate_contacts(
    mode_dfs: Dict[str, pd.DataFrame],
    ground_stations: List[GroundStationConfig],
) -> Dict[str, Any]:
    """
    Run sanity checks on contact data.

    Parameters
    ----------
    mode_dfs : Dict[str, pd.DataFrame]
        Dictionary of mode DataFrames.
    ground_stations : List[GroundStationConfig]
        Ground station configurations.

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

    if not mode_dfs:
        results['warnings'].append("No mode DataFrames provided")
        return results

    total_accesses = 0
    total_valid = 0
    gs_contact_counts = {gs.name: 0 for gs in ground_stations}

    for mode_name, df in mode_dfs.items():
        total_accesses += len(df)
        total_valid += df['Total_Valid_Contact'].sum()

        for gs in ground_stations:
            contact_col = f'Valid_Contact_{gs.name}'
            if contact_col in df.columns:
                gs_contact_counts[gs.name] += df[contact_col].sum()

    results['stats']['total_accesses'] = total_accesses
    results['stats']['total_valid_contacts'] = total_valid
    results['stats']['valid_percentage'] = (total_valid / total_accesses * 100) if total_accesses > 0 else 0
    results['stats']['contacts_by_station'] = gs_contact_counts

    # Check for at least some valid contacts
    if total_valid == 0:
        results['warnings'].append("No valid contacts found for any ground station")

    # Check each ground station has some contacts
    for gs_name, count in gs_contact_counts.items():
        if count == 0:
            results['warnings'].append(f"No valid contacts for station {gs_name}")

    # Check TTNC values are non-negative
    for mode_name, df in mode_dfs.items():
        for gs in ground_stations:
            contact_start_col = f'Next_{gs.name}_Contact_Start'
            if contact_start_col in df.columns:
                # Check for any negative TTNC (contact before slew stop)
                slew_stop = pd.to_datetime(df['Slew_Stop'])
                contact_start = pd.to_datetime(df[contact_start_col])
                valid_mask = slew_stop.notna() & contact_start.notna()
                if valid_mask.any():
                    negative_ttnc = (contact_start[valid_mask] < slew_stop[valid_mask]).sum()
                    if negative_ttnc > 0:
                        results['errors'].append(
                            f"Found {negative_ttnc} negative TTNC values for {gs.name} in {mode_name}"
                        )
                        results['passed'] = False

    print(f"Contact validation: {total_valid}/{total_accesses} valid ({results['stats']['valid_percentage']:.1f}%)")
    for gs_name, count in gs_contact_counts.items():
        print(f"  {gs_name}: {count} contacts")

    if results['errors']:
        print(f"  ERRORS: {results['errors']}")
    if results['warnings']:
        print(f"  Warnings: {results['warnings']}")

    return results
