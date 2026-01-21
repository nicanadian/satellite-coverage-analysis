"""
Plotting functions matching the original notebook's styling.

Uses matplotlib + cartopy for geographic plots, with consistent styling
from the original SAR notebook.
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from shapely.geometry import Point

from .config import AnalysisConfig, GroundStationConfig
from .orbits import TLEData, calculate_ground_track


def _get_map_features(config: AnalysisConfig) -> Tuple[Any, Any]:
    """
    Create land and ocean features with notebook styling.

    Returns
    -------
    Tuple[land_feature, ocean_feature]
    """
    land = cfeature.NaturalEarthFeature(
        category='physical',
        name='land',
        scale='50m',
        edgecolor='face',
        facecolor=config.land_color
    )

    ocean = cfeature.NaturalEarthFeature(
        category='physical',
        name='ocean',
        scale='50m',
        edgecolor='face',
        facecolor=config.ocean_color
    )

    return land, ocean


def _setup_map_axes(
    ax: plt.Axes,
    config: AnalysisConfig,
    add_features: bool = True,
) -> None:
    """
    Configure map axes with consistent styling.
    """
    if add_features:
        land, ocean = _get_map_features(config)
        ax.add_feature(ocean, zorder=0)
        ax.add_feature(land, zorder=1)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor=config.border_color, zorder=2)
    ax.gridlines()


def _get_satellite_color(sat_id: int, config: AnalysisConfig, sat_ids: List[int]) -> str:
    """Get color for a satellite from the color palette."""
    if sat_id in sat_ids:
        idx = sat_ids.index(sat_id)
    else:
        idx = hash(sat_id)
    return config.satellite_colors[idx % len(config.satellite_colors)]


def plot_ground_tracks(
    tle_data: Dict[int, TLEData],
    config: AnalysisConfig,
    duration_minutes: int = 100,
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot satellite ground tracks.

    Matches Cell 10 of the original notebook.

    Parameters
    ----------
    tle_data : Dict[int, TLEData]
        TLE data dictionary.
    config : AnalysisConfig
        Analysis configuration.
    duration_minutes : int
        Duration to plot ground tracks for.
    output_path : Path, optional
        Path to save the figure.

    Returns
    -------
    plt.Figure
        The matplotlib figure.
    """
    fig = plt.figure(figsize=config.plot_figsize)
    ax = plt.axes(projection=ccrs.PlateCarree())

    # Basic map features (simpler style for ground tracks)
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.gridlines()

    sat_ids = list(tle_data.keys())

    for sat_id in sat_ids:
        color = _get_satellite_color(sat_id, config, sat_ids)

        try:
            ground_track = calculate_ground_track(
                tle_data,
                sat_id,
                config.start_datetime,
                duration=timedelta(minutes=duration_minutes),
            )

            if not ground_track:
                continue

            lons, lats = zip(*ground_track)

            ax.plot(lons, lats,
                    color=color,
                    transform=ccrs.PlateCarree(),
                    label=f'Sat-{sat_id}',
                    linewidth=1)

            # Start/end markers
            ax.plot(lons[0], lats[0], 'o', color=color,
                    transform=ccrs.PlateCarree(), markersize=6)
            ax.plot(lons[-1], lats[-1], 'x', color=color,
                    transform=ccrs.PlateCarree(), markersize=6)

        except Exception as e:
            print(f"Warning: Could not plot ground track for satellite {sat_id}: {e}")

    ax.legend(loc='upper right')
    plt.title(f'Spacecraft Ground Tracks - first {duration_minutes} minutes of analysis')
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=config.plot_dpi, bbox_inches='tight')

    return fig


def plot_coverage_map(
    access_df: pd.DataFrame,
    targets_gdf: gpd.GeoDataFrame,
    config: AnalysisConfig,
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot access coverage map with statistics.

    Matches Cell 12 of the original notebook.

    Parameters
    ----------
    access_df : pd.DataFrame
        Access windows DataFrame.
    targets_gdf : gpd.GeoDataFrame
        Target AOIs.
    config : AnalysisConfig
        Analysis configuration.
    output_path : Path, optional
        Path to save the figure.

    Returns
    -------
    plt.Figure
        The matplotlib figure.
    """
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))

    # Plot 1: Accesses per spacecraft (bar chart)
    if not access_df.empty:
        accesses_per_spacecraft = access_df['Satellite'].value_counts().sort_index()
        accesses_per_spacecraft.plot(kind='bar', ax=ax1)
        ax1.set_title('Number of Accesses per Spacecraft')
        ax1.set_xlabel('Spacecraft ID')
        ax1.set_ylabel('Number of Accesses')
        ax1.tick_params(axis='x', rotation=45)

    # Plot 2: Accesses per day per spacecraft (line chart)
    if not access_df.empty:
        access_df_copy = access_df.copy()
        access_df_copy['Date'] = pd.to_datetime(access_df_copy['Start_Time']).dt.date
        accesses_per_day = access_df_copy.groupby(['Date', 'Satellite']).size().unstack(fill_value=0)
        accesses_per_day.plot(kind='line', marker='o', ax=ax2)
        ax2.set_title('Number of Accesses per Day per Spacecraft')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Number of Accesses')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True)
        ax2.legend(title='Spacecraft')

    # Plot 3: Map with access counts per AOI
    ax3 = plt.subplot(313, projection=ccrs.PlateCarree())
    land, ocean = _get_map_features(config)
    ax3.add_feature(ocean, zorder=0)
    ax3.add_feature(land, zorder=1)
    ax3.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor=config.border_color, zorder=2)
    ax3.gridlines()

    if not access_df.empty:
        accesses_per_aoi = access_df.groupby(['AOI_Lon', 'AOI_Lat']).size()

        if not accesses_per_aoi.empty:
            lons = [coord[0] for coord in accesses_per_aoi.index]
            lats = [coord[1] for coord in accesses_per_aoi.index]

            scatter = ax3.scatter(
                lons, lats,
                c=accesses_per_aoi.values,
                s=accesses_per_aoi.values * 50 + 10,
                cmap='viridis',
                alpha=0.6,
                transform=ccrs.PlateCarree()
            )

            cbar = plt.colorbar(scatter, ax=ax3)
            cbar.set_label('Number of Accesses')

            buffer = 5
            ax3.set_extent([
                min(lons) - buffer,
                max(lons) + buffer,
                min(lats) - buffer,
                max(lats) + buffer
            ])

    ax3.set_title('Number of Accesses per AOI Location')

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=config.plot_dpi, bbox_inches='tight')

    return fig


def plot_comm_cones(
    ground_stations: List[GroundStationConfig],
    tle_data: Dict[int, TLEData],
    targets_gdf: gpd.GeoDataFrame,
    config: AnalysisConfig,
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot ground station communication cones.

    Matches Cell 19 of the original notebook.

    Parameters
    ----------
    ground_stations : List[GroundStationConfig]
        Ground station configurations.
    tle_data : Dict[int, TLEData]
        TLE data dictionary.
    targets_gdf : gpd.GeoDataFrame
        Target AOIs.
    config : AnalysisConfig
        Analysis configuration.
    output_path : Path, optional
        Path to save the figure.

    Returns
    -------
    plt.Figure
        The matplotlib figure.
    """
    from .contacts import create_communication_cone

    fig = plt.figure(figsize=config.plot_figsize, dpi=config.plot_dpi)
    ax = plt.axes(projection=ccrs.PlateCarree())

    _setup_map_axes(ax, config)
    ax.set_global()
    ax.set_title("Ground Station Communication Cones")

    sat_ids = list(tle_data.keys())
    legend_handles = []
    all_cones = []

    for gs in ground_stations:
        for sat_id, tle in tle_data.items():
            sat_alt_m = tle.altitude_km * 1000
            color = _get_satellite_color(sat_id, config, sat_ids)

            try:
                comm_cone = create_communication_cone(
                    gs.lon, gs.lat, gs.min_elevation_deg, sat_alt_m
                )

                ax.add_geometries(
                    [comm_cone],
                    crs=ccrs.PlateCarree(),
                    facecolor='none',
                    edgecolor=color,
                    alpha=0.8,
                    linewidth=1.5
                )

                all_cones.append(comm_cone)

                label = f'GS: {gs.name} - Sat-{sat_id}'
                patch = mpatches.Patch(color=color, label=label)
                legend_handles.append(patch)

            except Exception as e:
                print(f"Warning: Could not create comm cone for {gs.name}/Sat-{sat_id}: {e}")

        # Plot ground station marker
        ax.plot(gs.lon, gs.lat, '^', color='black', markersize=6,
                transform=ccrs.PlateCarree())

    # Plot target AOIs
    if not targets_gdf.empty:
        ax.plot(
            targets_gdf.geometry.x,
            targets_gdf.geometry.y,
            'None',
            markeredgecolor='blue',
            markeredgewidth=1,
            marker='s',
            markersize=6,
            label='Target AOIs',
            transform=ccrs.PlateCarree()
        )

    # Set extent based on cones
    if all_cones:
        try:
            from shapely.ops import unary_union
            combined = unary_union(all_cones)
            if combined and not combined.is_empty:
                total_bounds = combined.bounds
                buffer = 5
                ax.set_extent([
                    total_bounds[0] - buffer,
                    total_bounds[2] + buffer,
                    total_bounds[1] - buffer,
                    total_bounds[3] + buffer
                ], crs=ccrs.PlateCarree())
        except Exception:
            # Fall back to global view if union fails
            ax.set_global()

    ax.legend(handles=legend_handles, loc='upper right', fontsize=8)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=config.plot_dpi, bbox_inches='tight')

    return fig


def plot_access_statistics(
    access_df: pd.DataFrame,
    config: AnalysisConfig,
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot access duration and revisit statistics.

    Parameters
    ----------
    access_df : pd.DataFrame
        Access windows DataFrame.
    config : AnalysisConfig
        Analysis configuration.
    output_path : Path, optional
        Path to save the figure.

    Returns
    -------
    plt.Figure
        The matplotlib figure.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    if access_df.empty:
        for ax in axes.flat:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
        return fig

    # Plot 1: Access duration distribution
    ax1 = axes[0, 0]
    access_df['Access_Duration'].hist(bins=30, ax=ax1, color='#1f77b4', edgecolor='black')
    ax1.set_xlabel('Access Duration (seconds)')
    ax1.set_ylabel('Count')
    ax1.set_title('Access Duration Distribution')
    ax1.axvline(access_df['Access_Duration'].mean(), color='red', linestyle='--',
                label=f'Mean: {access_df["Access_Duration"].mean():.0f}s')
    ax1.legend()

    # Plot 2: Accesses by satellite
    ax2 = axes[0, 1]
    sat_counts = access_df['Satellite'].value_counts().sort_index()
    sat_counts.plot(kind='bar', ax=ax2, color='#2ca02c')
    ax2.set_xlabel('Satellite ID')
    ax2.set_ylabel('Number of Accesses')
    ax2.set_title('Accesses per Satellite')
    ax2.tick_params(axis='x', rotation=45)

    # Plot 3: Off-nadir angle distribution
    ax3 = axes[1, 0]
    if 'OffNadir_Min' in access_df.columns:
        off_nadir_mean = (access_df['OffNadir_Min'] + access_df['OffNadir_Max']) / 2
        off_nadir_mean.hist(bins=30, ax=ax3, color='#ff7f0e', edgecolor='black')
        ax3.set_xlabel('Mean Off-Nadir Angle (degrees)')
        ax3.set_ylabel('Count')
        ax3.set_title('Off-Nadir Angle Distribution')

    # Plot 4: Access times over analysis window
    ax4 = axes[1, 1]
    access_df_sorted = access_df.sort_values('Start_Time')
    for sat_id in access_df['Satellite'].unique():
        sat_df = access_df_sorted[access_df_sorted['Satellite'] == sat_id]
        ax4.scatter(sat_df['Start_Time'], [sat_id] * len(sat_df), s=10,
                    label=f'Sat-{sat_id}', alpha=0.7)
    ax4.set_xlabel('Time')
    ax4.set_ylabel('Satellite ID')
    ax4.set_title('Access Timeline')
    ax4.tick_params(axis='x', rotation=45)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=config.plot_dpi, bbox_inches='tight')

    return fig


def plot_contact_validity(
    mode_dfs: Dict[str, pd.DataFrame],
    ground_stations: List[GroundStationConfig],
    config: AnalysisConfig,
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot valid vs invalid contacts per imaging mode.

    Matches Cell 22 of the original notebook.

    Parameters
    ----------
    mode_dfs : Dict[str, pd.DataFrame]
        Dictionary of mode DataFrames.
    ground_stations : List[GroundStationConfig]
        Ground station configurations.
    config : AnalysisConfig
        Analysis configuration.
    output_path : Path, optional
        Path to save the figure.

    Returns
    -------
    plt.Figure
        The matplotlib figure.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Collect data
    valid_invalid_counts = {}
    ground_station_success = {}

    for mode_name, df in mode_dfs.items():
        valid = df['Total_Valid_Contact'].sum()
        invalid = len(df) - valid
        valid_invalid_counts[mode_name] = {'Valid': valid, 'Invalid': invalid}

        # Calculate success rate across ground stations
        gs_success_rates = []
        for gs in ground_stations:
            valid_col = f'Valid_Contact_{gs.name}'
            contact_end_col = f'Next_{gs.name}_Contact_End'
            if valid_col in df.columns and contact_end_col in df.columns:
                valid_contacts = df[valid_col].sum()
                total_contacts = df[contact_end_col].notna().sum()
                if total_contacts > 0:
                    gs_success_rates.append((valid_contacts / total_contacts) * 100)

        ground_station_success[mode_name] = np.mean(gs_success_rates) if gs_success_rates else 0

    counts_df = pd.DataFrame(valid_invalid_counts).T.reset_index()
    counts_df.columns = ['Imaging_Mode', 'Valid', 'Invalid']

    success_df = pd.DataFrame(list(ground_station_success.items()),
                              columns=['Imaging_Mode', 'Success_Rate'])

    # Plot 1: Valid vs Invalid bar chart
    bar_width = 0.35
    indices = np.arange(len(counts_df))

    ax1.bar(indices, counts_df['Valid'], bar_width, label='Valid', color='#2ca02c')
    ax1.bar(indices + bar_width, counts_df['Invalid'], bar_width, label='Invalid', color='#d62728')

    ax1.set_xlabel('Imaging Mode')
    ax1.set_ylabel('Number of Contacts')
    ax1.set_title('Valid vs Invalid DDL Contacts per Imaging Mode')
    ax1.set_xticks(indices + bar_width / 2)
    ax1.set_xticklabels(counts_df['Imaging_Mode'], rotation=45, ha='right')
    ax1.legend()

    # Annotate bars
    for i in range(len(counts_df)):
        ax1.text(i, counts_df['Valid'].iloc[i] + 1, str(counts_df['Valid'].iloc[i]),
                 ha='center', va='bottom')
        ax1.text(i + bar_width, counts_df['Invalid'].iloc[i] + 1, str(counts_df['Invalid'].iloc[i]),
                 ha='center', va='bottom')

    # Plot 2: Success rate bar chart
    bars = ax2.bar(success_df['Imaging_Mode'], success_df['Success_Rate'], color='#1f77b4')
    ax2.set_xlabel('Imaging Mode')
    ax2.set_ylabel('Valid DDL Contact Percentage (%)')
    ax2.set_title('Valid DDL Contact Percentage per Imaging Mode')
    ax2.set_ylim(0, 100)
    ax2.tick_params(axis='x', rotation=45)

    # Annotate bars
    for bar in bars:
        height = bar.get_height()
        ax2.annotate(f'{height:.1f}%',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3),
                     textcoords="offset points",
                     ha='center', va='bottom')

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=config.plot_dpi, bbox_inches='tight')

    return fig


def plot_downlink_delay_distribution(
    downlink_delay_df: pd.DataFrame,
    config: AnalysisConfig,
    output_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot payload downlink delay distribution.

    Parameters
    ----------
    downlink_delay_df : pd.DataFrame
        Downlink delay DataFrame from calculate_downlink_delay.
    config : AnalysisConfig
        Analysis configuration.
    output_path : Path, optional
        Path to save the figure.

    Returns
    -------
    plt.Figure
        The matplotlib figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    if downlink_delay_df.empty or downlink_delay_df['Downlink_Delay_minutes'].isna().all():
        for ax in axes:
            ax.text(0.5, 0.5, 'No downlink delay data available', ha='center', va='center')
        return fig

    valid_delay = downlink_delay_df['Downlink_Delay_minutes'].dropna()

    # Plot 1: Histogram
    ax1 = axes[0]
    ax1.hist(valid_delay, bins=30, color='#1f77b4', edgecolor='black', alpha=0.7)
    ax1.axvline(valid_delay.median(), color='red', linestyle='--',
                label=f'Median: {valid_delay.median():.1f} min')
    ax1.axvline(valid_delay.quantile(0.95), color='orange', linestyle='--',
                label=f'P95: {valid_delay.quantile(0.95):.1f} min')
    ax1.set_xlabel('Payload Downlink Delay (minutes)')
    ax1.set_ylabel('Count')
    ax1.set_title('Downlink Delay Distribution')
    ax1.legend()

    # Plot 2: Box plot by imaging mode
    ax2 = axes[1]
    modes = downlink_delay_df['Imaging_Mode'].unique()
    data_by_mode = [downlink_delay_df[downlink_delay_df['Imaging_Mode'] == m]['Downlink_Delay_minutes'].dropna()
                    for m in modes]
    data_by_mode = [d for d in data_by_mode if len(d) > 0]

    if data_by_mode:
        bp = ax2.boxplot(data_by_mode, labels=[m for m, d in zip(modes, data_by_mode) if len(d) > 0])
        ax2.set_xlabel('Imaging Mode')
        ax2.set_ylabel('Payload Downlink Delay (minutes)')
        ax2.set_title('Downlink Delay by Imaging Mode')
        ax2.tick_params(axis='x', rotation=45)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=config.plot_dpi, bbox_inches='tight')

    return fig
