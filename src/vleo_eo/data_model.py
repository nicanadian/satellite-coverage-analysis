"""
Data volume modeling for VLEO EO Coverage Analysis.

Handles EO data generation rates and downlink capacity calculations.
"""

from typing import Dict, List, Any

import pandas as pd

from .config import GroundStationConfig


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


def calculate_downlink_kpis(
    mode_dfs: Dict[str, pd.DataFrame],
    ground_stations: List[GroundStationConfig],
) -> Dict[str, Any]:
    """
    Calculate downlink-related KPIs.

    Parameters
    ----------
    mode_dfs : Dict[str, pd.DataFrame]
        Dictionary of mode DataFrames.
    ground_stations : List[GroundStationConfig]
        Ground station configurations.

    Returns
    -------
    Dict[str, Any]
        Dictionary of KPIs.
    """
    kpis = {}

    # Calculate total data collected from mode DataFrames
    total_collected = 0.0
    for mode_name, df in mode_dfs.items():
        if 'Raw_Data_Size_GB' in df.columns:
            total_collected += df['Raw_Data_Size_GB'].sum()

    kpis['total_data_collected_gb'] = total_collected

    # Calculate total valid contact duration and capacity
    total_valid_contact_duration_s = 0.0
    for gs in ground_stations:
        for mode_name, df in mode_dfs.items():
            valid_col = f'Valid_Contact_{gs.name}'
            duration_col = f'Next_{gs.name}_Contact_Duration'

            if valid_col in df.columns and duration_col in df.columns:
                valid_mask = df[valid_col] == True
                total_valid_contact_duration_s += df.loc[valid_mask, duration_col].sum()

    # Calculate total downlink capacity
    # Use average downlink rate across stations
    if ground_stations:
        avg_rate = sum(gs.downlink_rate_mbps for gs in ground_stations) / len(ground_stations)
    else:
        avg_rate = 800  # Default

    total_capacity = calculate_contact_capacity(total_valid_contact_duration_s, avg_rate)

    # Data downlinked is limited by collected data or capacity
    total_downlinked = min(total_collected, total_capacity)
    kpis['total_data_downlinked_gb'] = total_downlinked

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
