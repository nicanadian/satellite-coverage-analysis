"""
Data volume modeling and backlog simulation.

Handles EO data generation rates, downlink capacity, and backlog time series.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

from .config import ImagingModeConfig, GroundStationConfig


def calculate_data_volume(
    mode: ImagingModeConfig,
) -> Dict[str, float]:
    """
    Calculate data volume parameters for an imaging mode.

    Parameters
    ----------
    mode : ImagingModeConfig
        Imaging mode configuration.

    Returns
    -------
    Dict[str, float]
        Data volume parameters:
        - raw_data_size_gb: Raw data per collection in GB
        - downlink_duration_s: Time to downlink at given rate
    """
    # Raw data size = collection rate * dwell time
    raw_data_size_gb = mode.collection_rate_gbps * mode.collect_dwell_time_s

    return {
        'raw_data_size_gb': raw_data_size_gb,
        'dwell_time_s': mode.collect_dwell_time_s,
        'collection_rate_gbps': mode.collection_rate_gbps,
    }


def calculate_downlink_duration(
    data_size_gb: float,
    downlink_rate_mbps: float,
) -> float:
    """
    Calculate downlink duration for a given data size.

    Parameters
    ----------
    data_size_gb : float
        Data size in GB.
    downlink_rate_mbps : float
        Downlink rate in Mbps.

    Returns
    -------
    float
        Downlink duration in seconds.
    """
    # Convert GB to Gb (multiply by 8) then to Mb (multiply by 1000)
    data_size_mb = data_size_gb * 8 * 1000
    return data_size_mb / downlink_rate_mbps


def calculate_contact_capacity(
    contact_duration_s: float,
    downlink_rate_mbps: float,
) -> float:
    """
    Calculate data capacity for a contact window.

    Parameters
    ----------
    contact_duration_s : float
        Contact duration in seconds.
    downlink_rate_mbps : float
        Downlink rate in Mbps.

    Returns
    -------
    float
        Data capacity in GB.
    """
    # Mbps * seconds = Mb, then / 8000 to get GB
    return (downlink_rate_mbps * contact_duration_s) / 8000


def simulate_backlog(
    mode_dfs: Dict[str, pd.DataFrame],
    ground_stations: List[GroundStationConfig],
    start_time: datetime,
    end_time: datetime,
    time_step_minutes: float = 15.0,
) -> pd.DataFrame:
    """
    Simulate data backlog over time based on collections and downlinks.

    Parameters
    ----------
    mode_dfs : Dict[str, pd.DataFrame]
        Dictionary of mode DataFrames with collection and contact info.
    ground_stations : List[GroundStationConfig]
        Ground station configurations.
    start_time : datetime
        Simulation start time.
    end_time : datetime
        Simulation end time.
    time_step_minutes : float
        Time step for backlog simulation in minutes.

    Returns
    -------
    pd.DataFrame
        Backlog time series with columns:
        - Time: Timestamp
        - Backlog_GB: Current backlog in GB
        - Collections_GB: Data collected in this step
        - Downlinked_GB: Data downlinked in this step
    """
    # Create time grid
    times = pd.date_range(start=start_time, end=end_time, freq=f'{int(time_step_minutes)}min')

    # Aggregate all collections across modes
    all_collections = []
    for mode_name, df in mode_dfs.items():
        for _, row in df.iterrows():
            if pd.notna(row.get('Raw_Data_Size_GB')):
                collect_time = row.get('Imaging_Stop', row.get('Start_Time'))
                if pd.notna(collect_time):
                    all_collections.append({
                        'time': pd.Timestamp(collect_time),
                        'data_gb': row['Raw_Data_Size_GB'],
                        'satellite': row['Satellite'],
                        'mode': mode_name,
                    })

    collections_df = pd.DataFrame(all_collections)
    if collections_df.empty:
        collections_df = pd.DataFrame(columns=['time', 'data_gb', 'satellite', 'mode'])

    # Aggregate all valid contacts
    all_contacts = []
    for mode_name, df in mode_dfs.items():
        for _, row in df.iterrows():
            for gs in ground_stations:
                valid_col = f'Valid_Contact_{gs.name}'
                start_col = f'Next_{gs.name}_Contact_Start'
                end_col = f'Next_{gs.name}_Contact_End'

                if row.get(valid_col, False) and pd.notna(row.get(start_col)):
                    contact_start = pd.Timestamp(row[start_col])
                    contact_end = pd.Timestamp(row[end_col]) if pd.notna(row.get(end_col)) else contact_start + pd.Timedelta(minutes=5)
                    duration_s = (contact_end - contact_start).total_seconds()

                    # Calculate capacity
                    capacity_gb = calculate_contact_capacity(duration_s, gs.downlink_rate_mbps)

                    all_contacts.append({
                        'start_time': contact_start,
                        'end_time': contact_end,
                        'capacity_gb': capacity_gb,
                        'satellite': row['Satellite'],
                        'station': gs.name,
                        'rate_mbps': gs.downlink_rate_mbps,
                    })

    contacts_df = pd.DataFrame(all_contacts)
    if contacts_df.empty:
        contacts_df = pd.DataFrame(columns=['start_time', 'end_time', 'capacity_gb', 'satellite', 'station', 'rate_mbps'])

    # Simulate backlog
    backlog_records = []
    current_backlog = 0.0
    step_td = pd.Timedelta(minutes=time_step_minutes)

    for i, t in enumerate(times):
        t_start = t
        t_end = t + step_td

        # Collections in this window
        if not collections_df.empty:
            collections_mask = (collections_df['time'] >= t_start) & (collections_df['time'] < t_end)
            collections_in_window = collections_df.loc[collections_mask, 'data_gb'].sum()
        else:
            collections_in_window = 0.0

        # Contacts active in this window - calculate overlap and downlink
        downlinked_in_window = 0.0
        if not contacts_df.empty:
            for _, contact in contacts_df.iterrows():
                # Calculate overlap between contact window and current time step
                overlap_start = max(t_start, contact['start_time'])
                overlap_end = min(t_end, contact['end_time'])

                if overlap_start < overlap_end:
                    overlap_seconds = (overlap_end - overlap_start).total_seconds()
                    # Calculate data downlinked based on rate and overlap duration
                    contact_total_seconds = (contact['end_time'] - contact['start_time']).total_seconds()
                    if contact_total_seconds > 0:
                        fraction = overlap_seconds / contact_total_seconds
                        downlinked_in_window += contact['capacity_gb'] * fraction

        # Update backlog
        current_backlog += collections_in_window
        actual_downlink = min(downlinked_in_window, current_backlog)
        current_backlog -= actual_downlink
        current_backlog = max(0, current_backlog)  # Ensure non-negative

        backlog_records.append({
            'Time': t,
            'Backlog_GB': current_backlog,
            'Collections_GB': collections_in_window,
            'Downlinked_GB': actual_downlink,
            'Cumulative_Collections_GB': 0,  # Will calculate below
            'Cumulative_Downlinked_GB': 0,
        })

    backlog_df = pd.DataFrame(backlog_records)

    # Calculate cumulative values
    backlog_df['Cumulative_Collections_GB'] = backlog_df['Collections_GB'].cumsum()
    backlog_df['Cumulative_Downlinked_GB'] = backlog_df['Downlinked_GB'].cumsum()

    return backlog_df


def calculate_daily_statistics(
    backlog_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate daily statistics from backlog time series.

    Parameters
    ----------
    backlog_df : pd.DataFrame
        Backlog time series from simulate_backlog.

    Returns
    -------
    pd.DataFrame
        Daily statistics with columns:
        - Date
        - Max_Backlog_GB
        - Mean_Backlog_GB
        - Total_Collections_GB
        - Total_Downlinked_GB
    """
    backlog_df = backlog_df.copy()
    backlog_df['Date'] = pd.to_datetime(backlog_df['Time']).dt.date

    daily_stats = backlog_df.groupby('Date').agg({
        'Backlog_GB': ['max', 'mean'],
        'Collections_GB': 'sum',
        'Downlinked_GB': 'sum',
    }).reset_index()

    daily_stats.columns = ['Date', 'Max_Backlog_GB', 'Mean_Backlog_GB',
                           'Total_Collections_GB', 'Total_Downlinked_GB']

    return daily_stats


def calculate_downlink_kpis(
    mode_dfs: Dict[str, pd.DataFrame],
    ground_stations: List[GroundStationConfig],
    backlog_df: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Calculate downlink-related KPIs.

    Parameters
    ----------
    mode_dfs : Dict[str, pd.DataFrame]
        Dictionary of mode DataFrames.
    ground_stations : List[GroundStationConfig]
        Ground station configurations.
    backlog_df : pd.DataFrame
        Backlog time series.

    Returns
    -------
    Dict[str, Any]
        Dictionary of KPIs.
    """
    kpis = {}

    # Total data collected
    total_collected = backlog_df['Cumulative_Collections_GB'].iloc[-1] if not backlog_df.empty else 0
    kpis['total_data_collected_gb'] = total_collected

    # Total data downlinked
    total_downlinked = backlog_df['Cumulative_Downlinked_GB'].iloc[-1] if not backlog_df.empty else 0
    kpis['total_data_downlinked_gb'] = total_downlinked

    # Peak backlog
    kpis['peak_backlog_gb'] = backlog_df['Backlog_GB'].max() if not backlog_df.empty else 0

    # Mean backlog
    kpis['mean_backlog_gb'] = backlog_df['Backlog_GB'].mean() if not backlog_df.empty else 0

    # Final backlog
    kpis['final_backlog_gb'] = backlog_df['Backlog_GB'].iloc[-1] if not backlog_df.empty else 0

    # Downlink efficiency (percentage of collected data that was downlinked)
    if total_collected > 0:
        kpis['downlink_efficiency_pct'] = (total_downlinked / total_collected) * 100
    else:
        kpis['downlink_efficiency_pct'] = 0

    # Contact statistics by ground station
    gs_stats = {}
    for gs in ground_stations:
        total_contacts = 0
        total_valid = 0
        total_duration_s = 0

        for mode_name, df in mode_dfs.items():
            valid_col = f'Valid_Contact_{gs.name}'
            duration_col = f'Next_{gs.name}_Contact_Duration'

            if valid_col in df.columns:
                total_contacts += df[valid_col].notna().sum()
                total_valid += df[valid_col].sum()

            if duration_col in df.columns:
                total_duration_s += df[duration_col].sum()

        capacity_gb = calculate_contact_capacity(total_duration_s, gs.downlink_rate_mbps)

        gs_stats[gs.name] = {
            'total_contacts': total_contacts,
            'valid_contacts': total_valid,
            'total_contact_duration_hours': total_duration_s / 3600,
            'total_capacity_gb': capacity_gb,
        }

    kpis['ground_station_stats'] = gs_stats

    return kpis


def validate_backlog(
    backlog_df: pd.DataFrame,
    mode_dfs: Dict[str, pd.DataFrame],
) -> Dict[str, Any]:
    """
    Run sanity checks on backlog simulation.

    Parameters
    ----------
    backlog_df : pd.DataFrame
        Backlog time series.
    mode_dfs : Dict[str, pd.DataFrame]
        Mode DataFrames.

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

    if backlog_df.empty:
        results['warnings'].append("Backlog DataFrame is empty")
        return results

    # Check for negative backlogs
    if (backlog_df['Backlog_GB'] < 0).any():
        results['errors'].append("Negative backlog values found")
        results['passed'] = False

    # Check monotonic behavior makes sense
    # Backlog should increase with collection, decrease with downlink
    # But the net effect depends on relative rates

    total_collected = backlog_df['Cumulative_Collections_GB'].iloc[-1]
    total_downlinked = backlog_df['Cumulative_Downlinked_GB'].iloc[-1]
    final_backlog = backlog_df['Backlog_GB'].iloc[-1]

    results['stats']['total_collected_gb'] = total_collected
    results['stats']['total_downlinked_gb'] = total_downlinked
    results['stats']['final_backlog_gb'] = final_backlog
    results['stats']['peak_backlog_gb'] = backlog_df['Backlog_GB'].max()

    # Check expected relationship: final = total_collected - total_downlinked
    expected_final = total_collected - total_downlinked
    if abs(final_backlog - expected_final) > 0.01:
        results['warnings'].append(
            f"Final backlog ({final_backlog:.2f} GB) doesn't match expected "
            f"({expected_final:.2f} GB = {total_collected:.2f} - {total_downlinked:.2f})"
        )

    # If downlink >> collection, backlog should trend toward zero
    if total_downlinked > total_collected * 1.5:
        if final_backlog > 0.1:
            results['warnings'].append(
                f"Downlink capacity exceeds collection but backlog remains: {final_backlog:.2f} GB"
            )

    print(f"Backlog validation:")
    print(f"  Total collected: {total_collected:.2f} GB")
    print(f"  Total downlinked: {total_downlinked:.2f} GB")
    print(f"  Final backlog: {final_backlog:.2f} GB")
    print(f"  Peak backlog: {results['stats']['peak_backlog_gb']:.2f} GB")

    if results['errors']:
        print(f"  ERRORS: {results['errors']}")
    if results['warnings']:
        print(f"  Warnings: {results['warnings']}")

    return results
