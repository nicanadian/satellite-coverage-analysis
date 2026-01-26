"""
Report generation utilities for Excel and PowerPoint outputs.

Generates clean summary reports with tables, KPIs, and embedded plots.
"""

from pathlib import Path
from typing import Dict, Optional, Any

import pandas as pd

from .config import AnalysisConfig


def _convert_datetimes_to_string(df: pd.DataFrame) -> pd.DataFrame:
    """Convert all datetime columns in a DataFrame to strings for Excel compatibility."""
    df = df.copy()
    for col in df.columns:
        # Handle datetime64 columns
        if 'datetime' in str(df[col].dtype):
            try:
                # Remove timezone if present, then format
                if df[col].dt.tz is not None:
                    df[col] = df[col].dt.tz_localize(None)
                df[col] = df[col].dt.strftime('%Y-%m-%d %H:%M:%S')
            except Exception:
                df[col] = df[col].astype(str)
        # Handle object columns that might contain Timestamps
        elif df[col].dtype == 'object' and len(df) > 0:
            try:
                first_valid = df[col].dropna().iloc[0] if not df[col].dropna().empty else None
                if first_valid is not None and hasattr(first_valid, 'strftime'):
                    df[col] = df[col].apply(
                        lambda x: x.strftime('%Y-%m-%d %H:%M:%S') if pd.notna(x) and hasattr(x, 'strftime') else x
                    )
            except (IndexError, TypeError):
                pass
    return df


def generate_excel_report(
    config: AnalysisConfig,
    access_df: pd.DataFrame,
    mode_dfs: Dict[str, pd.DataFrame],
    downlink_delay_df: pd.DataFrame,
    coverage_kpis: Dict[str, Any],
    contact_kpis: Dict[str, Any],
    downlink_kpis: Dict[str, Any],
    output_path: Optional[Path] = None,
    optimization_results: Optional[Dict[str, Any]] = None,
    ttc_analysis: Optional[Dict[str, Any]] = None,
    ka_analysis: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Generate Excel report with all analysis results.

    Parameters
    ----------
    config : AnalysisConfig
        Analysis configuration.
    access_df : pd.DataFrame
        Access windows DataFrame.
    mode_dfs : Dict[str, pd.DataFrame]
        Dictionary of mode DataFrames.
    downlink_delay_df : pd.DataFrame
        Payload downlink delay DataFrame.
    coverage_kpis : Dict
        Coverage KPIs from validation.
    contact_kpis : Dict
        Contact KPIs from validation.
    downlink_kpis : Dict
        Downlink KPIs.
    output_path : Path, optional
        Output path for the Excel file.

    Returns
    -------
    Path
        Path to the generated Excel file.
    """
    if output_path is None:
        output_path = config.output_path / config.excel_filename

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Sheet 1: Configuration
        config_data = {
            'Parameter': [
                'Start Date',
                'Duration (days)',
                'Time Step (s)',
                'Number of Satellites',
                'Number of Ground Stations',
                'Number of Imaging Modes',
                'Min Access Duration (s)',
                'Slew Rate (deg/s)',
                'Mode Switch Time (s)',
                'Contact Buffer (s)',
                'Downlink Search (hours)',
            ],
            'Value': [
                config.start_date,
                config.duration_days,
                config.time_step_s,
                len(config.satellites),
                len(config.ground_stations),
                len(config.imaging_modes),
                config.min_access_duration_s,
                config.slew_rate_deg_per_s,
                config.mode_switch_time_s,
                config.contact_buffer_s,
                config.downlink_search_hours,
            ]
        }
        pd.DataFrame(config_data).to_excel(writer, sheet_name='Config', index=False)

        # Satellite details
        sat_data = []
        for sat in config.satellites:
            sat_data.append({
                'Satellite ID': sat.sat_id,
                'Inclination (deg)': sat.inclination_deg,
                'Altitude (km)': sat.altitude_km,
                'RAAN (deg)': sat.raan_deg,
            })
        if sat_data:
            pd.DataFrame(sat_data).to_excel(writer, sheet_name='Satellites', index=False)

        # Ground station details
        gs_data = []
        for gs in config.ground_stations:
            gs_data.append({
                'Name': gs.name,
                'Latitude': gs.lat,
                'Longitude': gs.lon,
                'Elevation (m)': gs.elevation_m,
                'Min Elevation (deg)': gs.min_elevation_deg,
                'Ka Capable': gs.ka_capable,
                'Downlink Rate (Mbps)': gs.downlink_rate_mbps,
            })
        if gs_data:
            pd.DataFrame(gs_data).to_excel(writer, sheet_name='Ground_Stations', index=False)

        # Imaging mode details
        mode_data = []
        for mode in config.imaging_modes:
            mode_data.append({
                'Name': mode.name,
                'FOV Half Angle (deg)': mode.fov_half_angle_deg,
                'Dwell Time (s)': mode.collect_dwell_time_s,
                'Collection Rate (Gbps)': mode.collection_rate_gbps,
                'Processing Time (min)': mode.processing_time_min,
                'Off-Nadir Min (deg)': mode.off_nadir_min_deg,
                'Off-Nadir Max (deg)': mode.off_nadir_max_deg,
            })
        if mode_data:
            pd.DataFrame(mode_data).to_excel(writer, sheet_name='Imaging_Modes', index=False)

        # Sheet 2: Coverage KPIs
        kpi_data = []
        if coverage_kpis and 'stats' in coverage_kpis:
            stats = coverage_kpis['stats']
            kpi_data.extend([
                {'KPI': 'Total Accesses', 'Value': stats.get('total_accesses', 0), 'Unit': 'count'},
                {'KPI': 'Accesses per Day', 'Value': round(stats.get('accesses_per_day', 0), 2), 'Unit': 'count/day'},
                {'KPI': 'Mean Access Duration', 'Value': round(stats.get('mean_duration_s', 0), 1), 'Unit': 'seconds'},
                {'KPI': 'Max Access Duration', 'Value': round(stats.get('max_duration_s', 0), 1), 'Unit': 'seconds'},
                {'KPI': 'Mean Revisits per Target', 'Value': round(stats.get('mean_revisits', 0), 1), 'Unit': 'count'},
            ])

        if contact_kpis and 'stats' in contact_kpis:
            stats = contact_kpis['stats']
            kpi_data.extend([
                {'KPI': 'Total Valid Contacts', 'Value': stats.get('total_valid_contacts', 0), 'Unit': 'count'},
                {'KPI': 'Valid Contact Percentage', 'Value': round(stats.get('valid_percentage', 0), 1), 'Unit': '%'},
            ])

        if downlink_kpis:
            kpi_data.extend([
                {'KPI': 'Total Data Collected', 'Value': round(downlink_kpis.get('total_data_collected_gb', 0), 2), 'Unit': 'GB'},
                {'KPI': 'Total Data Downlinked', 'Value': round(downlink_kpis.get('total_data_downlinked_gb', 0), 2), 'Unit': 'GB'},
                {'KPI': 'Downlink Efficiency', 'Value': round(downlink_kpis.get('downlink_efficiency_pct', 0), 1), 'Unit': '%'},
            ])

        if kpi_data:
            pd.DataFrame(kpi_data).to_excel(writer, sheet_name='Coverage_KPIs', index=False)

        # Sheet 3: Access Windows
        if not access_df.empty:
            access_export = access_df.copy()
            # Convert datetime columns to string for Excel
            access_export = _convert_datetimes_to_string(access_export)
            access_export.to_excel(writer, sheet_name='Access_Windows', index=False)

        # Sheet 4: Contacts (consolidated from all modes)
        contacts_list = []
        for mode_name, df in mode_dfs.items():
            mode_contacts = df.copy()
            mode_contacts['Imaging_Mode'] = mode_name
            # Select key columns
            key_cols = ['Imaging_Mode', 'Satellite', 'Start_Time', 'End_Time', 'AOI_Lat', 'AOI_Lon',
                        'Access_Duration', 'Total_Valid_Contact', 'Collect_to_Delivery_Timeline']
            # Add ground station columns
            for gs in config.ground_stations:
                for col_suffix in ['Contact_Start', 'Contact_End', 'Contact_Duration']:
                    col = f'Next_{gs.name}_{col_suffix}'
                    if col in mode_contacts.columns:
                        key_cols.append(col)
                valid_col = f'Valid_Contact_{gs.name}'
                if valid_col in mode_contacts.columns:
                    key_cols.append(valid_col)

            available_cols = [c for c in key_cols if c in mode_contacts.columns]
            contacts_list.append(mode_contacts[available_cols])

        if contacts_list:
            contacts_df = pd.concat(contacts_list, ignore_index=True)
            contacts_df = _convert_datetimes_to_string(contacts_df)
            contacts_df.to_excel(writer, sheet_name='Contacts', index=False)

        # Sheet 5: Downlink KPIs by Ground Station
        if downlink_kpis and 'ground_station_stats' in downlink_kpis:
            gs_stats_list = []
            for gs_name, stats in downlink_kpis['ground_station_stats'].items():
                gs_stats_list.append({
                    'Ground Station': gs_name,
                    'Total Contacts': stats.get('total_contacts', 0),
                    'Valid Contacts': stats.get('valid_contacts', 0),
                    'Contact Duration (hours)': round(stats.get('total_contact_duration_hours', 0), 2),
                    'Capacity (GB)': round(stats.get('total_capacity_gb', 0), 2),
                })
            if gs_stats_list:
                pd.DataFrame(gs_stats_list).to_excel(writer, sheet_name='Downlink_KPIs', index=False)

        # Sheet 6: Payload Downlink Delay
        if not downlink_delay_df.empty:
            delay_export = _convert_datetimes_to_string(downlink_delay_df)
            delay_export.to_excel(writer, sheet_name='Downlink_Delay', index=False)

            # Downlink Delay Summary
            delay_summary = []
            for mode in downlink_delay_df['Imaging_Mode'].unique():
                mode_data = downlink_delay_df[downlink_delay_df['Imaging_Mode'] == mode]['Downlink_Delay_minutes'].dropna()
                if not mode_data.empty:
                    delay_summary.append({
                        'Imaging Mode': mode,
                        'Count': len(mode_data),
                        'Median (min)': round(mode_data.median(), 1),
                        'P90 (min)': round(mode_data.quantile(0.90), 1),
                        'P95 (min)': round(mode_data.quantile(0.95), 1),
                        'Max (min)': round(mode_data.max(), 1),
                    })
            if delay_summary:
                pd.DataFrame(delay_summary).to_excel(writer, sheet_name='Downlink_Delay_Summary', index=False)

        # Sheet 8: Optimization Results (if available)
        if optimization_results:
            from .optimization import optimization_results_to_dataframe
            opt_summary_df = optimization_results_to_dataframe(optimization_results)
            opt_summary_df.to_excel(writer, sheet_name='Optimization_Summary', index=False)

            # Detailed station lists for each provider
            opt_details = []
            for provider, result in optimization_results.items():
                for station in result.ttc_stations:
                    opt_details.append({
                        'Provider': provider,
                        'Type': 'TT&C',
                        'Station Name': station.name,
                        'Latitude': station.lat,
                        'Longitude': station.lon,
                        'S-Band': station.s_band,
                        'X-Band': station.x_band,
                        'Ka-Band': station.ka_band,
                    })
                for station in result.ka_stations:
                    opt_details.append({
                        'Provider': provider,
                        'Type': 'Ka Payload',
                        'Station Name': station.name,
                        'Latitude': station.lat,
                        'Longitude': station.lon,
                        'S-Band': station.s_band,
                        'X-Band': station.x_band,
                        'Ka-Band': station.ka_band,
                    })
            if opt_details:
                pd.DataFrame(opt_details).to_excel(writer, sheet_name='Optimization_Stations', index=False)

        # Sheet: TT&C Coverage Analysis (if available)
        if ttc_analysis:
            ttc_data = []

            # Summary KPIs
            ttc_data.append({'Category': 'Summary', 'Metric': 'Total TT&C Contacts', 'Value': ttc_analysis.get('total_contacts', 0), 'Unit': 'count'})
            ttc_data.append({'Category': 'Summary', 'Metric': 'Total Revolutions', 'Value': ttc_analysis.get('total_revs', 0), 'Unit': 'count'})
            ttc_data.append({'Category': 'Summary', 'Metric': 'Contacts per Day', 'Value': round(ttc_analysis.get('contacts_per_day', 0), 1), 'Unit': 'count/day'})
            ttc_data.append({'Category': 'Summary', 'Metric': 'Contacts per Revolution', 'Value': round(ttc_analysis.get('contacts_per_rev', 0), 2), 'Unit': 'count/rev'})

            # Gap Analysis
            ttc_data.append({'Category': 'Gap Analysis', 'Metric': 'Maximum Gap', 'Value': round(ttc_analysis.get('max_gap_min', 0), 1), 'Unit': 'minutes'})
            ttc_data.append({'Category': 'Gap Analysis', 'Metric': 'Average Gap', 'Value': round(ttc_analysis.get('avg_gap_min', 0), 1), 'Unit': 'minutes'})
            ttc_data.append({'Category': 'Gap Analysis', 'Metric': 'Median Gap', 'Value': round(ttc_analysis.get('median_gap_min', 0), 1), 'Unit': 'minutes'})
            ttc_data.append({'Category': 'Gap Analysis', 'Metric': 'P95 Gap', 'Value': round(ttc_analysis.get('p95_gap_min', 0), 1), 'Unit': 'minutes'})

            # Requirement Compliance
            ttc_data.append({'Category': 'Requirement', 'Metric': 'Revolutions Without Contact', 'Value': ttc_analysis.get('revs_without_contact', 0), 'Unit': 'count'})
            ttc_data.append({'Category': 'Requirement', 'Metric': 'Coverage Percentage', 'Value': round(ttc_analysis.get('coverage_pct', 0), 1), 'Unit': '%'})
            ttc_data.append({'Category': 'Requirement', 'Metric': '1 Pass/Rev Requirement Met', 'Value': 'YES' if ttc_analysis.get('requirement_met', False) else 'NO', 'Unit': ''})

            pd.DataFrame(ttc_data).to_excel(writer, sheet_name='TTC_Coverage', index=False)

            # Per-station breakdown
            if 'station_contacts' in ttc_analysis and ttc_analysis['station_contacts']:
                station_data = []
                for station_name, stats in ttc_analysis['station_contacts'].items():
                    station_data.append({
                        'Station': station_name,
                        'TT&C Contacts': stats.get('count', 0),
                        'Ka-Band Capable': 'Yes' if stats.get('is_ka', False) else 'No',
                    })
                if station_data:
                    pd.DataFrame(station_data).to_excel(writer, sheet_name='TTC_By_Station', index=False)

            # Raw gap data for periodicity analysis
            if 'gaps' in ttc_analysis and ttc_analysis['gaps']:
                gaps_data = []
                orbital_period = ttc_analysis.get('orbital_period_min', 90)
                for gap in ttc_analysis['gaps']:
                    gap_time = gap.get('time')
                    # Calculate day number from start
                    if gap_time and hasattr(config, 'start_datetime'):
                        day_num = (gap_time - config.start_datetime).total_seconds() / 86400
                    else:
                        day_num = None
                    gaps_data.append({
                        'Gap_Start_Time': gap_time.strftime('%Y-%m-%d %H:%M:%S') if gap_time and hasattr(gap_time, 'strftime') else str(gap_time),
                        'Day_Number': round(day_num, 2) if day_num is not None else None,
                        'Gap_Duration_min': round(gap.get('gap_min', 0), 1),
                        'Gap_Duration_revs': round(gap.get('gap_min', 0) / orbital_period, 2) if orbital_period > 0 else None,
                        'After_Station': gap.get('after_station', ''),
                        'Before_Station': gap.get('before_station', ''),
                        'Exceeds_1_Rev': 'Yes' if gap.get('gap_min', 0) > orbital_period else 'No',
                    })
                if gaps_data:
                    pd.DataFrame(gaps_data).to_excel(writer, sheet_name='TTC_Gaps', index=False)

        # Sheet: Ka-band Coverage Analysis (if available)
        if ka_analysis:
            ka_data = []

            # Summary KPIs
            ka_data.append({'Category': 'Summary', 'Metric': 'Total Collections', 'Value': ka_analysis.get('total_collects', 0), 'Unit': 'count'})
            ka_data.append({'Category': 'Summary', 'Metric': 'Viable First Ka Contacts', 'Value': ka_analysis.get('viable_collects', 0), 'Unit': 'count'})
            ka_data.append({'Category': 'Summary', 'Metric': 'Viable Percentage', 'Value': round(ka_analysis.get('viable_pct', 0), 1), 'Unit': '%'})

            # Timing
            ka_data.append({'Category': 'Timing', 'Metric': 'Min Contact Duration Needed', 'Value': round(ka_analysis.get('min_contact_duration_needed', 0), 1), 'Unit': 'minutes'})
            median_time = ka_analysis.get('median_time_to_viable')
            ka_data.append({'Category': 'Timing', 'Metric': 'Median Time to Viable Ka', 'Value': round(median_time, 1) if median_time else 'N/A', 'Unit': 'minutes'})
            p95_time = ka_analysis.get('p95_time_to_viable')
            ka_data.append({'Category': 'Timing', 'Metric': 'P95 Time to Viable Ka', 'Value': round(p95_time, 1) if p95_time else 'N/A', 'Unit': 'minutes'})

            # Configuration
            ka_data.append({'Category': 'Config', 'Metric': 'Image Size', 'Value': ka_analysis.get('image_size_gb', 0), 'Unit': 'GB'})
            ka_data.append({'Category': 'Config', 'Metric': 'Downlink Rate', 'Value': ka_analysis.get('downlink_rate_mbps', 0), 'Unit': 'Mbps'})
            ka_data.append({'Category': 'Config', 'Metric': 'Processing Buffer', 'Value': ka_analysis.get('processing_buffer_min', 5), 'Unit': 'minutes'})

            pd.DataFrame(ka_data).to_excel(writer, sheet_name='Ka_Coverage', index=False)

            # Per-station breakdown
            if 'station_contacts' in ka_analysis and ka_analysis['station_contacts']:
                station_data = []
                for station_name, count in ka_analysis['station_contacts'].items():
                    station_data.append({
                        'Station': station_name,
                        'Ka Contacts': count,
                    })
                if station_data:
                    pd.DataFrame(station_data).to_excel(writer, sheet_name='Ka_By_Station', index=False)

    print(f"Excel report saved to: {output_path}")
    return output_path
