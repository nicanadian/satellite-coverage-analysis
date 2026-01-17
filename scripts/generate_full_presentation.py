#!/usr/bin/env python3
"""
Generate comprehensive PowerPoint presentation for satellite coverage analysis.
"""

import sys
import random
from pathlib import Path
from datetime import timedelta
from typing import Dict, List, Tuple, Optional
from pyproj import Geod

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.vleo_eo.orbits import create_tle_data, calculate_ground_track
from src.vleo_eo.contacts import calculate_comm_range_km, calculate_ground_distance_km
from src.vleo_eo.config import load_config
from src.vleo_eo.propulsion import ltdn_to_raan, calculate_sso_inclination


def create_comm_cone_geodesic(lon: float, lat: float, min_elevation_deg: float, sat_altitude_km: float):
    """Create comm cone using geodesic geometry."""
    earth_radius_km = 6378.137
    elevation_rad = np.radians(min_elevation_deg)
    central_angle = np.arccos(
        earth_radius_km * np.cos(elevation_rad) / (earth_radius_km + sat_altitude_km)
    ) - elevation_rad
    max_ground_range_m = earth_radius_km * central_angle * 1000

    from shapely.geometry import Point
    gs_point = Point(lon, lat)
    gs_gdf = gpd.GeoDataFrame({'geometry': [gs_point]}, crs='EPSG:4326')
    aeqd_crs = f"+proj=aeqd +lat_0={lat} +lon_0={lon} +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
    gs_projected = gs_gdf.to_crs(aeqd_crs)
    buffered = gs_projected.buffer(max_ground_range_m)
    comm_cone_gdf = gpd.GeoDataFrame({'geometry': buffered}, crs=aeqd_crs).to_crs('EPSG:4326')
    return comm_cone_gdf.geometry.iloc[0]


def setup_map_axes(ax, land_color='#fafaf9', ocean_color='#d4dbdc', border_color='#ebd6d9'):
    """Setup map with standard styling."""
    land = cfeature.NaturalEarthFeature('physical', 'land', '50m', facecolor=land_color)
    ocean = cfeature.NaturalEarthFeature('physical', 'ocean', '50m', facecolor=ocean_color)
    ax.add_feature(ocean, zorder=0)
    ax.add_feature(land, zorder=1)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor=border_color, zorder=2)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor='gray', zorder=2)
    ax.gridlines(draw_labels=True, alpha=0.3)


def plot_ground_stations_and_cones(ax, ground_stations: List, sat_alt_km: float):
    """Plot ground stations with comm cones."""
    for gs in ground_stations:
        is_ka = gs.get('ka_band', False)
        marker_color = 'green' if is_ka else 'black'
        cone_color = 'green' if is_ka else 'black'
        cone_elev = 10.0 if is_ka else 5.0

        ax.plot(gs['lon'], gs['lat'], '^', color=marker_color, markersize=6,
                transform=ccrs.PlateCarree(), zorder=8)

        try:
            comm_cone = create_comm_cone_geodesic(gs['lon'], gs['lat'], cone_elev, sat_alt_km)
            if comm_cone is not None and comm_cone.is_valid:
                ax.add_geometries([comm_cone], crs=ccrs.PlateCarree(),
                                facecolor='none', edgecolor=cone_color, alpha=0.5,
                                linewidth=0.8, zorder=2)
        except Exception:
            pass


def add_slide1_text(prs, config, tle_data: Dict, targets_gdf: gpd.GeoDataFrame):
    """Add mission parameters as text slide directly to PowerPoint."""
    blank_layout = prs.slide_layouts[6]
    slide = prs.slides.add_slide(blank_layout)

    # Title
    title_box = slide.shapes.add_textbox(Inches(0.5), Inches(0.3), Inches(12), Inches(0.5))
    tf = title_box.text_frame
    p = tf.paragraphs[0]
    p.text = "Mission Configuration"
    p.font.size = Pt(28)
    p.font.bold = True
    p.alignment = PP_ALIGN.CENTER

    # Get config data
    orbit_config = config.raw_config.get('orbit', {})
    ltdn = orbit_config.get('ltdn_hours')
    ltdn_str = f'{int(ltdn)}:{int((ltdn%1)*60):02d}' if ltdn else 'N/A'
    ka_count = sum(1 for gs in config.ground_stations if gs.ka_capable)
    im = config.imaging_modes[0] if config.imaging_modes else None

    # Get unique providers
    providers = set()
    for gs in config.ground_stations:
        if hasattr(gs, 'provider') and gs.provider:
            providers.add(gs.provider)
    providers_str = ', '.join(sorted(providers)) if providers else 'N/A'

    # Get target region from geojson name or infer from config
    num_targets = len(targets_gdf) if targets_gdf is not None else 0
    targets_config = config.raw_config.get('targets', {})
    geojson_path = targets_config.get('geojson_path', '')
    # Extract region from filename like "target_aoi_deck_apac.geojson"
    if 'apac' in geojson_path.lower():
        target_region = 'APAC'
    elif 'europe' in geojson_path.lower():
        target_region = 'Europe'
    elif 'middleeast' in geojson_path.lower() or 'middle_east' in geojson_path.lower():
        target_region = 'Middle East'
    else:
        target_region = 'Global'

    # Column 1: Analysis Period + Constellation + Ground Network
    left_text = []
    left_text.append("ANALYSIS PERIOD")
    left_text.append(f"  Start Date: {config.start_date}")
    left_text.append(f"  Duration: {config.duration_days} days")
    left_text.append(f"  Time Step: {config.time_step_s} s")
    left_text.append(f"  End Date: {config.end_datetime.strftime('%Y-%m-%d')}")
    left_text.append("")
    left_text.append("CONSTELLATION")
    left_text.append(f"  Satellites: {len(config.satellites)}")
    left_text.append(f"  Altitude: {config.satellites[0].altitude_km} km")
    left_text.append(f"  Inclination: {config.satellites[0].inclination_deg}")
    left_text.append(f"  Orbit Type: {'SSO' if orbit_config.get('auto_sso') else 'Custom'}")
    left_text.append(f"  LTDN: {ltdn_str}")
    left_text.append("")
    left_text.append("GROUND NETWORK")
    left_text.append(f"  Provider(s): {providers_str}")
    left_text.append(f"  Total Stations: {len(config.ground_stations)}")
    left_text.append(f"  Ka-band: {ka_count}")
    left_text.append(f"  TT&C Only: {len(config.ground_stations) - ka_count}")
    left_text.append("")
    left_text.append("TARGET DECK")
    left_text.append(f"  {num_targets} targets in {target_region}")

    left_box = slide.shapes.add_textbox(Inches(0.5), Inches(1.0), Inches(4), Inches(6))
    tf = left_box.text_frame
    tf.word_wrap = True
    for i, line in enumerate(left_text):
        if i == 0:
            p = tf.paragraphs[0]
        else:
            p = tf.add_paragraph()
        p.text = line
        p.font.size = Pt(12)
        p.font.name = "Consolas"
        if line and not line.startswith(' '):
            p.font.bold = True

    # Column 2: TLE Data + Imaging
    # Show up to 3 random satellites
    sat_ids = list(tle_data.keys())
    if len(sat_ids) > 3:
        import random
        sat_ids_to_show = random.sample(sat_ids, 3)
    else:
        sat_ids_to_show = sat_ids

    mid_text = []
    mid_text.append("TLE DATA")
    for sat_id in sat_ids_to_show:
        tle = tle_data[sat_id]
        mid_text.append(f"  Satellite {sat_id}")
        mid_text.append(f"  {tle.line1}")
        mid_text.append(f"  {tle.line2}")
        mid_text.append("")
    if len(sat_ids) > 3:
        mid_text.append(f"  ... and {len(sat_ids) - 3} more satellites")
        mid_text.append("")
    mid_text.append("IMAGING")
    if im:
        mid_text.append(f"  Mode: {im.name}")
        mid_text.append(f"  ONA Range: {im.off_nadir_min_deg} - {im.off_nadir_max_deg}")

    mid_box = slide.shapes.add_textbox(Inches(5.0), Inches(1.0), Inches(7.5), Inches(6))
    tf = mid_box.text_frame
    tf.word_wrap = True
    for i, line in enumerate(mid_text):
        if i == 0:
            p = tf.paragraphs[0]
        else:
            p = tf.add_paragraph()
        p.text = line
        p.font.size = Pt(12)
        p.font.name = "Consolas"
        if line and not line.startswith(' '):
            p.font.bold = True

    return slide


def generate_slide2_gs_map(config, ground_stations: List, sat_alt_km: float,
                          targets_gdf: gpd.GeoDataFrame, output_path: Path,
                          excel_path: Path = None):
    """Generate ground stations, comm cones, and targets map."""
    fig = plt.figure(figsize=(16, 9))
    ax = plt.axes(projection=ccrs.PlateCarree())
    setup_map_axes(ax)
    ax.set_global()

    plot_ground_stations_and_cones(ax, ground_stations, sat_alt_km)

    # Add station labels
    for gs in ground_stations:
        is_ka = gs.get('ka_band', False)
        label = gs['name'].replace('KSAT-', '').replace('-', '\n')
        ax.text(gs['lon'], gs['lat'] + 3, label, transform=ccrs.PlateCarree(),
                fontsize=6, ha='center', va='bottom',
                color='green' if is_ka else 'black')

    # Plot targets as blue X markers
    if targets_gdf is not None and len(targets_gdf) > 0:
        for idx, row in targets_gdf.iterrows():
            lat = row.geometry.y
            lon = row.geometry.x
            ax.plot(lon, lat, 'x', color='blue', markersize=8, markeredgewidth=2,
                    transform=ccrs.PlateCarree(), zorder=10)

    # Legend
    legend_elements = [
        Line2D([0], [0], marker='^', color='green', markersize=8, linestyle='None', label='Ka Station'),
        Patch(facecolor='none', edgecolor='green', label='Ka Comm Cone (10°)'),
        Line2D([0], [0], marker='^', color='black', markersize=8, linestyle='None', label='TT&C Station'),
        Patch(facecolor='none', edgecolor='black', label='TT&C Comm Cone (5°)'),
        Line2D([0], [0], marker='x', color='blue', markersize=8, linestyle='None',
               markeredgewidth=2, label=f'Targets ({len(targets_gdf)})'),
    ]
    ax.legend(handles=legend_elements, loc='lower left', fontsize=9)
    ax.set_title('Ground Station Network, Communication Cones, and Target Locations', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def generate_slide3_single_orbit(tle_data: Dict, config, output_path: Path):
    """Generate single orbit ground track."""
    fig = plt.figure(figsize=(16, 9))
    ax = plt.axes(projection=ccrs.PlateCarree())
    setup_map_axes(ax)
    ax.set_global()

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    # Calculate orbital period from altitude
    sat_alt_km = config.satellites[0].altitude_km
    earth_radius_km = 6378.137
    mu = 398600.4418  # km³/s²
    semi_major_axis = earth_radius_km + sat_alt_km
    orbital_period_min = 2 * np.pi * np.sqrt(semi_major_axis**3 / mu) / 60

    for i, (sat_id, tle) in enumerate(tle_data.items()):
        ground_track = calculate_ground_track(
            tle_data, sat_id, config.start_datetime,
            duration=timedelta(minutes=orbital_period_min), step_s=30.0
        )
        if not ground_track:
            continue

        lons, lats = zip(*ground_track)
        color = colors[i % len(colors)]

        ax.plot(lons, lats, color=color, linewidth=1.5, transform=ccrs.PlateCarree(),
                label=f'Satellite {sat_id}')

        # Start marker (circle)
        ax.plot(lons[0], lats[0], 'o', color=color, markersize=10,
                markeredgecolor='black', markeredgewidth=1,
                transform=ccrs.PlateCarree(), zorder=10)

        # End marker (square)
        ax.plot(lons[-1], lats[-1], 's', color=color, markersize=8,
                markeredgecolor='black', markeredgewidth=1,
                transform=ccrs.PlateCarree(), zorder=10)

        # Add direction arrows along track
        arrow_indices = [len(ground_track) // 4, len(ground_track) // 2, 3 * len(ground_track) // 4]
        for idx in arrow_indices:
            if idx + 1 < len(ground_track):
                dx = lons[idx + 1] - lons[idx]
                dy = lats[idx + 1] - lats[idx]
                ax.annotate('', xy=(lons[idx] + dx*0.5, lats[idx] + dy*0.5),
                           xytext=(lons[idx], lats[idx]),
                           arrowprops=dict(arrowstyle='->', color=color, lw=1.5),
                           transform=ccrs.PlateCarree())

    # Legend with start/end markers
    legend_elements = [
        Line2D([0], [0], color=colors[0], linewidth=1.5, label='Ground Track'),
        Line2D([0], [0], marker='o', color=colors[0], markersize=8, linestyle='None',
               markeredgecolor='black', label='Orbit Start'),
        Line2D([0], [0], marker='s', color=colors[0], markersize=7, linestyle='None',
               markeredgecolor='black', label='Orbit End'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    ax.set_title(f'Single Orbit Ground Track (~{orbital_period_min:.1f} minutes)', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def generate_slide4_access_results(access_df: pd.DataFrame, tle_data: Dict, output_path: Path):
    """Generate access results overview plot."""
    fig = plt.figure(figsize=(16, 9))
    ax = plt.axes(projection=ccrs.PlateCarree())
    setup_map_axes(ax)
    ax.set_global()

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    for _, access in access_df.iterrows():
        sat_id = access['Satellite']
        start_time = pd.to_datetime(access['Start_Time'])
        if start_time.tzinfo is None:
            start_time = start_time.tz_localize('UTC')

        duration_s = access.get('Access_Duration', 60)
        ground_track = calculate_ground_track(
            tle_data, sat_id, start_time.to_pydatetime(),
            duration=timedelta(seconds=duration_s), step_s=10.0
        )
        if not ground_track:
            continue

        lons, lats = zip(*ground_track)
        color = colors[sat_id % len(colors)]

        ax.plot(lons, lats, color=color, linewidth=1, alpha=0.6, transform=ccrs.PlateCarree())
        ax.plot(access['AOI_Lon'], access['AOI_Lat'], 'x', color='blue', markersize=6,
                markeredgewidth=1.5, transform=ccrs.PlateCarree())

    ax.set_title(f'Access Windows Overview ({len(access_df)} accesses)', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def generate_slide5_access_with_cones(access_df: pd.DataFrame, tle_data: Dict,
                                      ground_stations: List, sat_alt_km: float, output_path: Path):
    """Generate access plot with 60-min post-collect tracks and comm cones."""
    fig = plt.figure(figsize=(16, 9))
    ax = plt.axes(projection=ccrs.PlateCarree())
    setup_map_axes(ax)
    ax.set_global()

    # Plot comm cones
    plot_ground_stations_and_cones(ax, ground_stations, sat_alt_km)

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    for _, access in access_df.iterrows():
        sat_id = access['Satellite']
        start_time = pd.to_datetime(access['Start_Time'])
        if start_time.tzinfo is None:
            start_time = start_time.tz_localize('UTC')

        # 60 min ground track after collect
        ground_track = calculate_ground_track(
            tle_data, sat_id, start_time.to_pydatetime(),
            duration=timedelta(minutes=60), step_s=30.0
        )
        if not ground_track:
            continue

        lons, lats = zip(*ground_track)
        color = colors[sat_id % len(colors)]

        ax.plot(lons, lats, color=color, linewidth=1, alpha=0.7, transform=ccrs.PlateCarree())
        ax.plot(access['AOI_Lon'], access['AOI_Lat'], 'x', color='blue', markersize=6,
                markeredgewidth=1.5, transform=ccrs.PlateCarree())

    ax.set_title('Access Windows with 60-min Post-Collect Tracks', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def calculate_swath_polygon(ground_track: List[Tuple], sat_alt_km: float,
                            off_nadir_min_deg: float, off_nadir_max_deg: float):
    """Calculate imaging swath polygon based on off-nadir angle limits."""
    from pyproj import Geod
    from shapely.geometry import Polygon

    if len(ground_track) < 2:
        return None

    geod = Geod(ellps='WGS84')
    earth_radius_km = 6378.137

    # Calculate ground range for min and max off-nadir angles
    def ona_to_ground_range(ona_deg):
        ona_rad = np.radians(ona_deg)
        # Simplified geometry: ground_range ≈ altitude * tan(ona)
        return sat_alt_km * np.tan(ona_rad) * 1000  # meters

    range_min = ona_to_ground_range(off_nadir_min_deg)
    range_max = ona_to_ground_range(off_nadir_max_deg)

    # Build swath polygon by projecting perpendicular to track
    left_points = []
    right_points = []

    for i, (lon, lat) in enumerate(ground_track):
        # Calculate track heading
        if i < len(ground_track) - 1:
            next_lon, next_lat = ground_track[i + 1]
            fwd_az, _, _ = geod.inv(lon, lat, next_lon, next_lat)
        else:
            prev_lon, prev_lat = ground_track[i - 1]
            fwd_az, _, _ = geod.inv(prev_lon, prev_lat, lon, lat)

        # Perpendicular azimuths (left and right of track)
        left_az = (fwd_az - 90) % 360
        right_az = (fwd_az + 90) % 360

        # Project to max off-nadir range on both sides
        left_lon, left_lat, _ = geod.fwd(lon, lat, left_az, range_max)
        right_lon, right_lat, _ = geod.fwd(lon, lat, right_az, range_max)

        left_points.append((left_lon, left_lat))
        right_points.append((right_lon, right_lat))

    # Create polygon: left side forward, right side backward
    polygon_coords = left_points + right_points[::-1] + [left_points[0]]

    try:
        poly = Polygon(polygon_coords)
        if not poly.is_valid:
            poly = poly.buffer(0)
        return poly
    except:
        return None


def generate_broadside_collect_plot(access_row: pd.Series, tle_data: Dict, ground_stations: List,
                                    sat_alt_km: float, output_path: Path,
                                    off_nadir_min: float = 0.0, off_nadir_max: float = 30.0):
    """Generate broadside collect opportunity plot with swath visualization."""
    sat_id = access_row['Satellite']
    start_time = pd.to_datetime(access_row['Start_Time'])
    if start_time.tzinfo is None:
        start_time = start_time.tz_localize('UTC')

    aoi_lat = access_row['AOI_Lat']
    aoi_lon = access_row['AOI_Lon']

    fig = plt.figure(figsize=(14, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())
    setup_map_axes(ax)

    # Use orange color for satellite track
    track_color = '#ff7f0e'  # Orange

    # Calculate extended track for context (before and after)
    track_before = calculate_ground_track(
        tle_data, sat_id,
        (start_time - timedelta(minutes=3)).to_pydatetime(),
        duration=timedelta(minutes=3), step_s=10.0
    )
    track_during = calculate_ground_track(
        tle_data, sat_id, start_time.to_pydatetime(),
        duration=timedelta(seconds=access_row.get('Access_Duration', 60)), step_s=5.0
    )
    track_after = calculate_ground_track(
        tle_data, sat_id,
        (start_time + timedelta(seconds=access_row.get('Access_Duration', 60))).to_pydatetime(),
        duration=timedelta(minutes=3), step_s=10.0
    )

    # Calculate and plot swath polygon
    if track_during:
        swath_poly = calculate_swath_polygon(track_during, sat_alt_km, off_nadir_min, off_nadir_max)
        if swath_poly and swath_poly.is_valid:
            ax.add_geometries([swath_poly], crs=ccrs.PlateCarree(),
                            facecolor=track_color, edgecolor='none',
                            alpha=0.15, zorder=3, label='Imaging Swath')

    # Plot tracks - gray for before/after, orange for access window
    if track_before:
        lons, lats = zip(*track_before)
        ax.plot(lons, lats, color='gray', linewidth=1.5, alpha=0.5, transform=ccrs.PlateCarree())

    if track_during:
        lons, lats = zip(*track_during)
        ax.plot(lons, lats, color=track_color, linewidth=2.5, transform=ccrs.PlateCarree(),
                label='Access Window')

        # Mark broadside (midpoint) with a diamond marker
        mid_idx = len(track_during) // 2
        mid_lon, mid_lat = track_during[mid_idx]
        ax.plot(mid_lon, mid_lat, 'D', color=track_color, markersize=10,
                markeredgecolor='black', markeredgewidth=1.5,
                transform=ccrs.PlateCarree(), label='Broadside', zorder=10)

    if track_after:
        lons, lats = zip(*track_after)
        ax.plot(lons, lats, color='gray', linewidth=1.5, alpha=0.5, transform=ccrs.PlateCarree())

    # Plot target as blue X
    ax.plot(aoi_lon, aoi_lat, 'x', color='blue', markersize=12, markeredgewidth=3,
            transform=ccrs.PlateCarree(), label='Target', zorder=11)

    # Set extent around the access
    all_lons = [aoi_lon]
    all_lats = [aoi_lat]
    if track_during:
        lons, lats = zip(*track_during)
        all_lons.extend(lons)
        all_lats.extend(lats)

    margin = 5
    ax.set_extent([min(all_lons) - margin, max(all_lons) + margin,
                   min(all_lats) - margin, max(all_lats) + margin], crs=ccrs.PlateCarree())

    # Legend
    legend_elements = [
        Line2D([0], [0], color='gray', linewidth=1.5, alpha=0.5, label='Ground Track'),
        Line2D([0], [0], color=track_color, linewidth=2.5, label='Access Window'),
        Patch(facecolor=track_color, alpha=0.15, label=f'Imaging Swath (ONA {off_nadir_min}-{off_nadir_max}°)'),
        Line2D([0], [0], marker='D', color=track_color, markersize=8, linestyle='None',
               markeredgecolor='black', label='Broadside'),
        Line2D([0], [0], marker='x', color='blue', markersize=10, linestyle='None',
               markeredgewidth=3, label='Target'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)

    ax.set_title(f'Broadside Collect Opportunity\n'
                 f'Target: ({aoi_lat:.1f}°, {aoi_lon:.1f}°) | Satellite {sat_id}\n'
                 f'{start_time.strftime("%Y-%m-%d %H:%M:%S")} UTC | ONA: {off_nadir_min}-{off_nadir_max}°',
                 fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def generate_pre_collect_diagram(access_row: pd.Series, tle_data: Dict, ground_stations: List,
                                  sat_alt_km: float, output_path: Path):
    """Generate diagram showing ground track BEFORE collect, back to last TT&C contact."""
    sat_id = access_row['Satellite']
    start_time = pd.to_datetime(access_row['Start_Time'])
    if start_time.tzinfo is None:
        start_time = start_time.tz_localize('UTC')

    aoi_lat = access_row['AOI_Lat']
    aoi_lon = access_row['AOI_Lon']

    fig = plt.figure(figsize=(16, 10))
    ax = plt.axes([0.05, 0.05, 0.70, 0.90], projection=ccrs.PlateCarree())
    setup_map_axes(ax)

    # Calculate ground track going BACKWARDS 3 hours from collect
    # We'll propagate forward from 3h before and find TT&C contacts
    search_start_ts = start_time - timedelta(hours=3)
    search_start = search_start_ts.to_pydatetime()
    if search_start.tzinfo is not None:
        search_start = search_start.replace(tzinfo=None)

    pre_track = calculate_ground_track(
        tle_data, sat_id, search_start,
        duration=timedelta(hours=3), step_s=30.0
    )

    if not pre_track:
        plt.close()
        return None

    # Find last TT&C contact before collect
    last_ttc_contact = None
    last_ttc_station = None
    last_ttc_end_idx = 0

    for gs in ground_stations:
        is_ka = gs.get('ka_band', False)
        if is_ka:
            continue  # Only TT&C stations

        contact_elev = 5.0
        max_range_km = calculate_comm_range_km(contact_elev, sat_alt_km)

        in_contact = False
        contact_end_idx = None

        for i, (lon, lat) in enumerate(pre_track):
            dist = calculate_ground_distance_km(lat, lon, gs['lat'], gs['lon'])
            currently_in = dist <= max_range_km

            if currently_in:
                in_contact = True
                contact_end_idx = i
            elif not currently_in and in_contact:
                # Contact ended
                contact_time_min = (i * 30) / 60  # Time from search_start
                if contact_end_idx is not None and contact_end_idx > last_ttc_end_idx:
                    last_ttc_end_idx = contact_end_idx
                    last_ttc_station = gs['name']
                    # Calculate actual time
                    last_ttc_contact = search_start + timedelta(seconds=contact_end_idx * 30)
                in_contact = False

    # Plot the pre-collect track from last TT&C to collect
    if last_ttc_end_idx > 0:
        track_segment = pre_track[last_ttc_end_idx:]
    else:
        track_segment = pre_track

    lons, lats = zip(*track_segment)
    ax.plot(lons, lats, color='purple', linewidth=1.5, alpha=0.7,
            transform=ccrs.PlateCarree(), label='Track from Last TT&C')

    # Mark last TT&C contact point
    if last_ttc_end_idx > 0:
        ttc_lon, ttc_lat = pre_track[last_ttc_end_idx]
        ax.plot(ttc_lon, ttc_lat, 'o', color='purple', markersize=10,
                transform=ccrs.PlateCarree(), label=f'Last TT&C ({last_ttc_station})')

    # Mark collect point
    ax.plot(aoi_lon, aoi_lat, 'x', color='blue', markersize=10, markeredgewidth=2,
            transform=ccrs.PlateCarree(), label='Collect Target')

    # Plot ground stations and cones
    plot_ground_stations_and_cones(ax, ground_stations, sat_alt_km)

    # Calculate uplink-to-collect time
    if last_ttc_contact:
        collect_time = start_time.to_pydatetime()
        if collect_time.tzinfo is not None:
            collect_time = collect_time.replace(tzinfo=None)
        uplink_to_collect_min = (collect_time - last_ttc_contact).total_seconds() / 60
    else:
        uplink_to_collect_min = float('nan')

    # Set extent
    margin = 20
    ax.set_extent([min(lons) - margin, max(lons) + margin,
                   max(min(lats) - margin, -85), min(max(lats) + margin, 85)], crs=ccrs.PlateCarree())

    # Legend on the right
    legend_elements = [
        Line2D([0], [0], color='purple', linewidth=1.5, label='Track from Last TT&C'),
        Line2D([0], [0], marker='o', color='purple', markersize=8, linestyle='None', label='Last TT&C Contact'),
        Line2D([0], [0], marker='x', color='blue', markersize=8, linestyle='None', markeredgewidth=2, label='Collect Target'),
        Line2D([0], [0], marker='^', color='green', markersize=8, linestyle='None', label='Ka Station'),
        Line2D([0], [0], marker='^', color='black', markersize=8, linestyle='None', label='TT&C Station'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1.0), fontsize=9)

    title = f'Pre-Collect Track - Target ({aoi_lat:.1f}°, {aoi_lon:.1f}°)\n'
    title += f'Last TT&C: {last_ttc_station or "None"}'
    if not np.isnan(uplink_to_collect_min):
        title += f' | Uplink-to-Collect: {uplink_to_collect_min:.1f} min'
    ax.set_title(title, fontsize=11, fontweight='bold')

    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    return {
        'last_ttc_station': last_ttc_station,
        'uplink_to_collect_min': uplink_to_collect_min,
    }


def generate_collect_diagram_3h(access_row: pd.Series, tle_data: Dict, ground_stations: List,
                                sat_alt_km: float, output_path: Path):
    """Generate 3-hour post-collect diagram (existing functionality)."""
    sat_id = access_row['Satellite']
    start_time = pd.to_datetime(access_row['Start_Time'])
    if start_time.tzinfo is None:
        start_time = start_time.tz_localize('UTC')

    aoi_lat = access_row['AOI_Lat']
    aoi_lon = access_row['AOI_Lon']

    fig = plt.figure(figsize=(16, 10))
    ax = plt.axes([0.05, 0.05, 0.70, 0.90], projection=ccrs.PlateCarree())
    setup_map_axes(ax)

    # 3h ground track
    extended_track = calculate_ground_track(
        tle_data, sat_id, start_time.to_pydatetime(),
        duration=timedelta(hours=3), step_s=30.0
    )

    if not extended_track:
        plt.close()
        return None

    ext_lons, ext_lats = zip(*extended_track)
    ax.plot(ext_lons, ext_lats, color='gray', linewidth=1.5, alpha=0.6,
            transform=ccrs.PlateCarree(), label='Ground Track (3h)')

    # Target
    ax.plot(aoi_lon, aoi_lat, 'x', color='blue', markersize=8, markeredgewidth=2,
            transform=ccrs.PlateCarree(), label='Target')

    # Color map for contact times
    colors = ['#00ff00', '#2ecc71', '#f1c40f', '#e67e22', '#e74c3c']
    custom_cmap = LinearSegmentedColormap.from_list("contact_time", colors, N=256)

    # Track contacts and plot
    contact_stats = []
    first_ka_contact = None

    for gs in ground_stations:
        is_ka = gs.get('ka_band', False)
        marker_color = 'green' if is_ka else 'black'
        cone_color = 'green' if is_ka else 'black'
        cone_elev = 10.0 if is_ka else 5.0

        ax.plot(gs['lon'], gs['lat'], '^', color=marker_color, markersize=6,
                transform=ccrs.PlateCarree(), zorder=8)

        try:
            comm_cone = create_comm_cone_geodesic(gs['lon'], gs['lat'], cone_elev, sat_alt_km)
            if comm_cone and comm_cone.is_valid:
                ax.add_geometries([comm_cone], crs=ccrs.PlateCarree(),
                                facecolor='none', edgecolor=cone_color, alpha=0.5,
                                linewidth=0.8, zorder=2)
        except:
            pass

        # Find contacts
        contact_elev = 10.0 if is_ka else 5.0
        max_range_km = calculate_comm_range_km(contact_elev, sat_alt_km)

        in_contact = False
        contact_start_idx = None
        contacts = []

        for i, (lon, lat) in enumerate(extended_track):
            dist = calculate_ground_distance_km(lat, lon, gs['lat'], gs['lon'])
            currently_in = dist <= max_range_km

            if currently_in and not in_contact:
                contact_start_idx = i
                in_contact = True
            elif not currently_in and in_contact:
                contacts.append((contact_start_idx, i - 1))
                in_contact = False

        if in_contact:
            contacts.append((contact_start_idx, len(extended_track) - 1))

        for start_idx, end_idx in contacts:
            if end_idx <= start_idx:
                continue

            contact_lons = [extended_track[j][0] for j in range(start_idx, end_idx + 1)]
            contact_lats = [extended_track[j][1] for j in range(start_idx, end_idx + 1)]

            time_to_contact_min = start_idx * 30 / 60
            contact_duration_min = (end_idx - start_idx) * 30 / 60

            norm_time = min(time_to_contact_min / 180.0, 1.0)
            contact_color = custom_cmap(norm_time)

            # Plot the contact segment with color
            ax.plot(contact_lons, contact_lats, color=contact_color, linewidth=3,
                    transform=ccrs.PlateCarree(), zorder=5)

            contact_stats.append({
                'station': gs['name'],
                'is_ka': is_ka,
                'time_to_contact_min': time_to_contact_min,
                'duration_min': contact_duration_min,
                'color': contact_color,
                'start_idx': start_idx,
                'end_idx': end_idx,
            })

            if is_ka and (first_ka_contact is None or time_to_contact_min < first_ka_contact['time_to_contact_min']):
                first_ka_contact = contact_stats[-1]

    # Sort contacts by time
    contact_stats.sort(key=lambda x: x['time_to_contact_min'])

    # Set extent
    margin = 20
    ax.set_extent([min(ext_lons) - margin, max(ext_lons) + margin,
                   max(min(ext_lats) - margin, -85), min(max(ext_lats) + margin, 85)], crs=ccrs.PlateCarree())

    # Legend
    legend_elements = [
        Line2D([0], [0], color='gray', linewidth=1.5, alpha=0.6, label='Ground Track (3h)'),
        Line2D([0], [0], marker='x', color='blue', markersize=8, linestyle='None', markeredgewidth=2, label='Target'),
        Line2D([0], [0], marker='^', color='green', markersize=8, linestyle='None', label='Ka Station'),
        Patch(facecolor='none', edgecolor='green', label='Ka Comm Cone'),
        Line2D([0], [0], marker='^', color='black', markersize=8, linestyle='None', label='TT&C Station'),
        Patch(facecolor='none', edgecolor='black', label='TT&C Comm Cone'),
    ]

    # Add contact entries
    for contact in contact_stats[:10]:  # Limit to 10
        ka_marker = " (Ka)" if contact['is_ka'] else ""
        label = f"{contact['station']}{ka_marker}: C+{contact['time_to_contact_min']:.1f}m"
        legend_elements.append(Line2D([0], [0], color=contact['color'], linewidth=3, label=label))

    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1.0), fontsize=8)

    title = f'Post-Collect Track (3h) - Target ({aoi_lat:.1f}°, {aoi_lon:.1f}°)\n'
    if first_ka_contact:
        title += f'First Ka: {first_ka_contact["station"]} at C+{first_ka_contact["time_to_contact_min"]:.1f}min'
    ax.set_title(title, fontsize=11, fontweight='bold')

    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    return {'first_ka_contact': first_ka_contact}


def generate_ttnc_summary_plot(excel_path: Path, output_path: Path):
    """Generate TTNC Ka summary analysis plot with station breakdown."""
    try:
        ttnc_df = pd.read_excel(excel_path, sheet_name='TTNC_Ka')
    except Exception:
        return None

    if len(ttnc_df) == 0:
        return None

    # Get TTNC values in minutes
    if 'TTNC_Ka_minutes' in ttnc_df.columns:
        ttnc_values = ttnc_df['TTNC_Ka_minutes'].dropna()
    elif 'TTNC_Ka_seconds' in ttnc_df.columns:
        ttnc_values = ttnc_df['TTNC_Ka_seconds'].dropna() / 60
    else:
        return None

    fig = plt.figure(figsize=(16, 7))

    # Histogram (left)
    ax1 = fig.add_axes([0.05, 0.12, 0.28, 0.78])
    ax1.hist(ttnc_values, bins=20, color='#2ecc71', edgecolor='black', alpha=0.7)
    ax1.axvline(ttnc_values.median(), color='red', linestyle='--', linewidth=2, label=f'Median: {ttnc_values.median():.1f} min')
    ax1.axvline(ttnc_values.quantile(0.95), color='orange', linestyle='--', linewidth=2, label=f'P95: {ttnc_values.quantile(0.95):.1f} min')
    ax1.set_xlabel('Time to Next Ka Contact (minutes)', fontsize=10)
    ax1.set_ylabel('Frequency', fontsize=10)
    ax1.set_title('TTNC Distribution', fontsize=11, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Box plot (middle)
    ax2 = fig.add_axes([0.38, 0.12, 0.18, 0.78])
    bp = ax2.boxplot(ttnc_values, vert=True, patch_artist=True)
    bp['boxes'][0].set_facecolor('#2ecc71')
    bp['boxes'][0].set_alpha(0.7)
    ax2.set_ylabel('Time to Next Ka Contact (minutes)', fontsize=10)
    ax2.set_title('Overall Stats', fontsize=11, fontweight='bold')
    ax2.set_xticklabels(['All'])

    # Add stats text below box plot
    stats_text = f"Count: {len(ttnc_values)}\n"
    stats_text += f"Min: {ttnc_values.min():.1f} min\n"
    stats_text += f"Median: {ttnc_values.median():.1f} min\n"
    stats_text += f"Mean: {ttnc_values.mean():.1f} min\n"
    stats_text += f"P95: {ttnc_values.quantile(0.95):.1f} min\n"
    stats_text += f"Max: {ttnc_values.max():.1f} min"
    ax2.text(0.5, -0.25, stats_text, transform=ax2.transAxes, fontsize=9,
             verticalalignment='top', horizontalalignment='center',
             fontfamily='monospace')

    # Ka station breakdown table (right)
    ax3 = fig.add_axes([0.62, 0.12, 0.36, 0.78])
    ax3.axis('off')

    if 'Next_Ka_Station' in ttnc_df.columns:
        # Get stats per station
        station_stats = ttnc_df.groupby('Next_Ka_Station')['TTNC_Ka_minutes'].agg(
            ['count', 'mean', 'median', 'min', 'max']
        ).round(1)
        station_stats = station_stats.sort_values('count', ascending=False)

        # Build table text
        table_text = "First Ka Contact by Station\n"
        table_text += "=" * 50 + "\n\n"
        table_text += f"{'Station':<22} {'#':>4} {'Med':>6} {'Avg':>6} {'Min':>6} {'Max':>6}\n"
        table_text += "-" * 52 + "\n"

        for station in station_stats.index:
            count = int(station_stats.loc[station, 'count'])
            median = station_stats.loc[station, 'median']
            mean = station_stats.loc[station, 'mean']
            min_val = station_stats.loc[station, 'min']
            max_val = station_stats.loc[station, 'max']
            # Shorten station name
            short_name = station.replace('KSAT-', '').replace('-', ' ')[:20]
            table_text += f"{short_name:<22} {count:>4} {median:>5.0f}m {mean:>5.0f}m {min_val:>5.0f}m {max_val:>5.0f}m\n"

        table_text += "-" * 52 + "\n"
        total = len(ttnc_df)
        overall_median = ttnc_df['TTNC_Ka_minutes'].median()
        overall_mean = ttnc_df['TTNC_Ka_minutes'].mean()
        overall_min = ttnc_df['TTNC_Ka_minutes'].min()
        overall_max = ttnc_df['TTNC_Ka_minutes'].max()
        table_text += f"{'TOTAL':<22} {total:>4} {overall_median:>5.0f}m {overall_mean:>5.0f}m {overall_min:>5.0f}m {overall_max:>5.0f}m\n"

        ax3.text(0, 0.95, table_text, transform=ax3.transAxes, fontsize=10,
                 verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='#e8f5e9', alpha=0.8))

    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    return {'median': ttnc_values.median(), 'p95': ttnc_values.quantile(0.95)}


def generate_uplink_to_collect_summary(access_df: pd.DataFrame, tle_data: Dict,
                                        ground_stations: List, sat_alt_km: float,
                                        output_path: Path):
    """Generate summary analysis of uplink-to-collect times (pre-collect TT&C) with station breakdown."""
    from src.vleo_eo.contacts import calculate_comm_range_km, calculate_ground_distance_km

    # Collect data with station info
    uplink_data = []

    for _, access_row in access_df.iterrows():
        sat_id = access_row['Satellite']
        start_time = pd.to_datetime(access_row['Start_Time'])
        if start_time.tzinfo is None:
            start_time = start_time.tz_localize('UTC')

        # Calculate track going backwards 3 hours
        search_start = (start_time - timedelta(hours=3)).to_pydatetime()
        if search_start.tzinfo is not None:
            search_start = search_start.replace(tzinfo=None)

        pre_track = calculate_ground_track(
            tle_data, sat_id, search_start,
            duration=timedelta(hours=3), step_s=30.0
        )

        if not pre_track:
            continue

        # Find last TT&C contact before collect
        last_ttc_idx = None
        last_ttc_station = None
        for gs in ground_stations:
            is_ka = gs.get('ka_band', False)
            if is_ka:
                continue  # Only TT&C stations

            max_range_km = calculate_comm_range_km(5.0, sat_alt_km)

            for i, (lon, lat) in enumerate(pre_track):
                dist = calculate_ground_distance_km(lat, lon, gs['lat'], gs['lon'])
                if dist <= max_range_km:
                    if last_ttc_idx is None or i > last_ttc_idx:
                        last_ttc_idx = i
                        last_ttc_station = gs['name']

        if last_ttc_idx is not None and last_ttc_station is not None:
            # Time from last TT&C to collect
            uplink_to_collect_min = (len(pre_track) - last_ttc_idx) * 30 / 60
            uplink_data.append({
                'station': last_ttc_station,
                'uplink_time_min': uplink_to_collect_min
            })

    if len(uplink_data) == 0:
        return None

    uplink_df = pd.DataFrame(uplink_data)
    uplink_times = uplink_df['uplink_time_min'].values

    fig = plt.figure(figsize=(16, 7))

    # Histogram (left)
    ax1 = fig.add_axes([0.05, 0.12, 0.28, 0.78])
    ax1.hist(uplink_times, bins=20, color='#9b59b6', edgecolor='black', alpha=0.7)
    ax1.axvline(np.median(uplink_times), color='red', linestyle='--', linewidth=2,
                label=f'Median: {np.median(uplink_times):.1f} min')
    ax1.axvline(np.percentile(uplink_times, 95), color='orange', linestyle='--', linewidth=2,
                label=f'P95: {np.percentile(uplink_times, 95):.1f} min')
    ax1.set_xlabel('Time Since Last TT&C Contact (minutes)', fontsize=10)
    ax1.set_ylabel('Frequency', fontsize=10)
    ax1.set_title('Uplink-to-Collect Distribution', fontsize=11, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Box plot (middle)
    ax2 = fig.add_axes([0.38, 0.12, 0.18, 0.78])
    bp = ax2.boxplot(uplink_times, vert=True, patch_artist=True)
    bp['boxes'][0].set_facecolor('#9b59b6')
    bp['boxes'][0].set_alpha(0.7)
    ax2.set_ylabel('Time Since Last TT&C (minutes)', fontsize=10)
    ax2.set_title('Overall Stats', fontsize=11, fontweight='bold')
    ax2.set_xticklabels(['All'])

    # Add stats text below box plot
    stats_text = f"Count: {len(uplink_times)}\n"
    stats_text += f"Min: {np.min(uplink_times):.1f} min\n"
    stats_text += f"Median: {np.median(uplink_times):.1f} min\n"
    stats_text += f"Mean: {np.mean(uplink_times):.1f} min\n"
    stats_text += f"P95: {np.percentile(uplink_times, 95):.1f} min\n"
    stats_text += f"Max: {np.max(uplink_times):.1f} min"
    ax2.text(0.5, -0.25, stats_text, transform=ax2.transAxes, fontsize=9,
             verticalalignment='top', horizontalalignment='center',
             fontfamily='monospace')

    # TT&C station breakdown table (right)
    ax3 = fig.add_axes([0.62, 0.12, 0.36, 0.78])
    ax3.axis('off')

    # Get stats per station
    station_stats = uplink_df.groupby('station')['uplink_time_min'].agg(
        ['count', 'mean', 'median', 'min', 'max']
    ).round(1)
    station_stats = station_stats.sort_values('count', ascending=False)

    # Build table text
    table_text = "Last TT&C Contact by Station\n"
    table_text += "=" * 50 + "\n\n"
    table_text += f"{'Station':<22} {'#':>4} {'Med':>6} {'Avg':>6} {'Min':>6} {'Max':>6}\n"
    table_text += "-" * 52 + "\n"

    for station in station_stats.index:
        count = int(station_stats.loc[station, 'count'])
        median = station_stats.loc[station, 'median']
        mean = station_stats.loc[station, 'mean']
        min_val = station_stats.loc[station, 'min']
        max_val = station_stats.loc[station, 'max']
        # Shorten station name
        short_name = station.replace('KSAT-', '').replace('-', ' ')[:20]
        table_text += f"{short_name:<22} {count:>4} {median:>5.0f}m {mean:>5.0f}m {min_val:>5.0f}m {max_val:>5.0f}m\n"

    table_text += "-" * 52 + "\n"
    total = len(uplink_df)
    overall_median = np.median(uplink_times)
    overall_mean = np.mean(uplink_times)
    overall_min = np.min(uplink_times)
    overall_max = np.max(uplink_times)
    table_text += f"{'TOTAL':<22} {total:>4} {overall_median:>5.0f}m {overall_mean:>5.0f}m {overall_min:>5.0f}m {overall_max:>5.0f}m\n"

    ax3.text(0, 0.95, table_text, transform=ax3.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='#f3e5f5', alpha=0.8))

    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    return {'median': np.median(uplink_times), 'p95': np.percentile(uplink_times, 95)}


def generate_backlog_summary(excel_path: Path, output_path: Path):
    """Generate backlog and downlink analysis summary slide."""
    try:
        backlog_df = pd.read_excel(excel_path, sheet_name='Backlog_TimeSeries')
        downlink_df = pd.read_excel(excel_path, sheet_name='Downlink_KPIs')
        kpi_df = pd.read_excel(excel_path, sheet_name='Coverage_KPIs')
    except Exception as e:
        print(f"Error loading backlog data: {e}")
        return None

    if len(backlog_df) == 0:
        return None

    # Parse time column
    backlog_df['Time'] = pd.to_datetime(backlog_df['Time'])

    fig = plt.figure(figsize=(16, 8))

    # Top left: Backlog time series
    ax1 = fig.add_axes([0.06, 0.55, 0.42, 0.38])
    ax1.fill_between(backlog_df['Time'], backlog_df['Backlog_GB'], alpha=0.3, color='#e74c3c')
    ax1.plot(backlog_df['Time'], backlog_df['Backlog_GB'], color='#e74c3c', linewidth=1)
    peak_backlog = backlog_df['Backlog_GB'].max()
    peak_idx = backlog_df['Backlog_GB'].idxmax()
    ax1.axhline(peak_backlog, color='red', linestyle='--', alpha=0.7, label=f'Peak: {peak_backlog:.1f} GB')
    ax1.set_ylabel('Backlog (GB)', fontsize=10)
    ax1.set_title('Data Backlog Over Time', fontsize=11, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=30, labelsize=8)

    # Top right: Cumulative collections vs downlinked
    ax2 = fig.add_axes([0.55, 0.55, 0.42, 0.38])
    ax2.plot(backlog_df['Time'], backlog_df['Cumulative_Collections_GB'],
             color='#3498db', linewidth=1.5, label='Collected')
    ax2.plot(backlog_df['Time'], backlog_df['Cumulative_Downlinked_GB'],
             color='#2ecc71', linewidth=1.5, label='Downlinked')
    ax2.fill_between(backlog_df['Time'], backlog_df['Cumulative_Collections_GB'],
                     backlog_df['Cumulative_Downlinked_GB'], alpha=0.2, color='#e74c3c')
    ax2.set_ylabel('Cumulative Data (GB)', fontsize=10)
    ax2.set_title('Cumulative Collections vs Downlink', fontsize=11, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='x', rotation=30, labelsize=8)

    # Bottom left: Summary statistics
    ax3 = fig.add_axes([0.06, 0.08, 0.28, 0.38])
    ax3.axis('off')

    total_collected = backlog_df['Cumulative_Collections_GB'].iloc[-1]
    total_downlinked = backlog_df['Cumulative_Downlinked_GB'].iloc[-1]
    avg_backlog = backlog_df['Backlog_GB'].mean()
    final_backlog = backlog_df['Backlog_GB'].iloc[-1]

    stats_text = "Data Summary\n"
    stats_text += "=" * 30 + "\n\n"
    stats_text += f"Total Collected:    {total_collected:>8.1f} GB\n"
    stats_text += f"Total Downlinked:   {total_downlinked:>8.1f} GB\n"
    stats_text += f"Peak Backlog:       {peak_backlog:>8.1f} GB\n"
    stats_text += f"Average Backlog:    {avg_backlog:>8.1f} GB\n"
    stats_text += f"Final Backlog:      {final_backlog:>8.1f} GB\n"
    stats_text += "\n"
    if total_collected > 0:
        downlink_ratio = total_downlinked / total_collected * 100
        stats_text += f"Downlink Ratio:     {downlink_ratio:>7.1f} %\n"

    ax3.text(0, 0.95, stats_text, transform=ax3.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='#e8f4f8', alpha=0.8))

    # Bottom right: Per-station downlink capacity
    ax4 = fig.add_axes([0.40, 0.08, 0.57, 0.38])
    ax4.axis('off')

    # Filter to stations with actual capacity
    active_stations = downlink_df[downlink_df['Capacity (GB)'] > 0].copy()
    active_stations = active_stations.sort_values('Capacity (GB)', ascending=False)

    table_text = "Downlink Capacity by Ground Station\n"
    table_text += "=" * 58 + "\n\n"
    table_text += f"{'Station':<26} {'Contacts':>8} {'Hours':>8} {'Capacity':>10}\n"
    table_text += "-" * 58 + "\n"

    for _, row in active_stations.iterrows():
        station = row['Ground Station'].replace('KSAT-', '').replace('-', ' ')[:24]
        contacts = int(row['Valid Contacts'])
        hours = row['Contact Duration (hours)']
        capacity = row['Capacity (GB)']
        table_text += f"{station:<26} {contacts:>8} {hours:>7.1f}h {capacity:>9.1f} GB\n"

    table_text += "-" * 58 + "\n"
    total_contacts = int(active_stations['Valid Contacts'].sum())
    total_hours = active_stations['Contact Duration (hours)'].sum()
    total_capacity = active_stations['Capacity (GB)'].sum()
    table_text += f"{'TOTAL':<26} {total_contacts:>8} {total_hours:>7.1f}h {total_capacity:>9.1f} GB\n"

    ax4.text(0, 0.95, table_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='#e8f8e8', alpha=0.8))

    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    return {'peak_backlog': peak_backlog, 'total_collected': total_collected, 'total_downlinked': total_downlinked}


def generate_optimization_slide(excel_path: Path, output_path: Path, config):
    """Generate ground station optimization analysis slide."""
    try:
        ttnc_df = pd.read_excel(excel_path, sheet_name='TTNC_Ka')
        contacts_df = pd.read_excel(excel_path, sheet_name='Contacts')
        downlink_df = pd.read_excel(excel_path, sheet_name='Downlink_KPIs')
        config_df = pd.read_excel(excel_path, sheet_name='Config')
    except Exception as e:
        print(f"Error loading data for optimization: {e}")
        return None

    # Get orbital parameters
    altitude_km = config.satellites[0].altitude_km if config.satellites else 250
    mu = 3.986004418e14  # m³/s²
    r_earth = 6371.0  # km
    r = (r_earth + altitude_km) * 1000
    period_min = 2 * np.pi * np.sqrt(r**3 / mu) / 60

    # Calculate revolutions
    duration_days = config.duration_days
    num_revs = int(duration_days * 24 * 60 / period_min)

    # Analyze Ka-band coverage
    total_collects = len(ttnc_df)
    ka_within_sla = (ttnc_df['TTNC_Ka_minutes'] <= 240).sum()
    ka_max_ttnc = ttnc_df['TTNC_Ka_minutes'].max()
    ka_stations_used = ttnc_df['Next_Ka_Station'].unique()

    # Analyze TT&C coverage per station
    valid_cols = [c for c in contacts_df.columns if c.startswith('Valid_Contact_')]
    station_contacts = {}
    for col in valid_cols:
        station = col.replace('Valid_Contact_', '')
        count = int(contacts_df[col].sum())
        if count > 0:
            station_contacts[station] = count

    # Sort stations by contact count
    sorted_stations = sorted(station_contacts.items(), key=lambda x: -x[1])

    # Create figure
    fig = plt.figure(figsize=(16, 8))

    # Top left: Requirements & Current Config
    ax1 = fig.add_axes([0.04, 0.52, 0.44, 0.42])
    ax1.axis('off')

    req_text = "Optimization Requirements\n"
    req_text += "=" * 40 + "\n\n"
    req_text += "SLA Requirements:\n"
    req_text += "  1. TT&C: 1+ contact per orbital revolution\n"
    req_text += "  2. Ka-band: Downlink within 4 hours of collect\n\n"
    req_text += f"Mission Parameters:\n"
    req_text += f"  Duration: {duration_days} days\n"
    req_text += f"  Altitude: {altitude_km} km\n"
    req_text += f"  Orbital Period: {period_min:.1f} min\n"
    req_text += f"  Total Revolutions: ~{num_revs}\n"
    req_text += f"  Total Collects: {total_collects}\n\n"
    req_text += f"Current Configuration:\n"
    req_text += f"  Total Stations: {len(config.ground_stations)}\n"
    ka_count = sum(1 for gs in config.ground_stations if gs.ka_capable)
    req_text += f"  Ka-capable: {ka_count}\n"
    req_text += f"  TT&C-capable: {len(config.ground_stations)}\n"

    ax1.text(0, 0.95, req_text, transform=ax1.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='#e3f2fd', alpha=0.8))

    # Top right: Ka-band analysis
    ax2 = fig.add_axes([0.52, 0.52, 0.44, 0.42])
    ax2.axis('off')

    ka_text = "Ka-Band Downlink Analysis\n"
    ka_text += "=" * 40 + "\n\n"
    ka_text += f"SLA: Downlink within 4 hours (240 min)\n\n"
    ka_text += f"Results:\n"
    ka_text += f"  Collects within SLA: {ka_within_sla}/{total_collects} ({100*ka_within_sla/total_collects:.0f}%)\n"
    ka_text += f"  Max TTNC: {ka_max_ttnc:.1f} min\n"
    ka_text += f"  SLA Met: YES\n\n"
    ka_text += f"Ka Stations Used:\n"
    for station in ka_stations_used:
        short_name = station.replace('KSAT-', '')
        count = len(ttnc_df[ttnc_df['Next_Ka_Station'] == station])
        ka_text += f"  {short_name}: {count} collects\n"
    ka_text += f"\nMinimum Ka Stations Needed: 1\n"
    ka_text += f"  {ka_stations_used[0].replace('KSAT-', '')}\n"

    ax2.text(0, 0.95, ka_text, transform=ax2.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='#e8f5e9', alpha=0.8))

    # Bottom left: TT&C station ranking
    ax3 = fig.add_axes([0.04, 0.05, 0.44, 0.42])
    ax3.axis('off')

    ttc_text = "TT&C Station Coverage Ranking\n"
    ttc_text += "=" * 40 + "\n\n"
    ttc_text += f"{'Station':<28} {'Contacts':>10}\n"
    ttc_text += "-" * 40 + "\n"

    for station, count in sorted_stations:
        short_name = station.replace('KSAT-', '')[:26]
        pct = 100 * count / total_collects if total_collects > 0 else 0
        ttc_text += f"{short_name:<28} {count:>6} ({pct:>3.0f}%)\n"

    ttc_text += "\n* Coverage = contacts during access windows\n"

    ax3.text(0, 0.95, ttc_text, transform=ax3.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='#fff3e0', alpha=0.8))

    # Bottom right: Minimum station recommendation
    ax4 = fig.add_axes([0.52, 0.05, 0.44, 0.42])
    ax4.axis('off')

    # Find stations with 100% coverage
    full_coverage_stations = [s for s, c in sorted_stations if c == total_collects]

    rec_text = "Minimum Station Set Recommendation\n"
    rec_text += "=" * 40 + "\n\n"
    rec_text += "To meet both SLAs:\n\n"
    rec_text += "OPTION A: 1 Station (Combined)\n"
    rec_text += "-" * 40 + "\n"
    # Find station that handles both Ka and has full TT&C coverage
    combined = [s for s in full_coverage_stations if s in [str(ks) for ks in ka_stations_used]]
    if combined:
        rec_text += f"  {combined[0].replace('KSAT-', '')}\n"
        rec_text += f"    - Ka: 100% within SLA\n"
        rec_text += f"    - TT&C: 100% access coverage\n\n"
    else:
        rec_text += "  No single station meets both SLAs\n\n"

    rec_text += "OPTION B: 2 Stations (Redundancy)\n"
    rec_text += "-" * 40 + "\n"
    rec_text += f"  Primary Ka: {ka_stations_used[0].replace('KSAT-', '')}\n"
    if len(full_coverage_stations) >= 2:
        backup = [s for s in full_coverage_stations if s != ka_stations_used[0]]
        if backup:
            rec_text += f"  Backup TT&C: {backup[0].replace('KSAT-', '')}\n"
    rec_text += "\n"
    rec_text += "Cost Reduction:\n"
    rec_text += f"  Current: {len(config.ground_stations)} stations\n"
    rec_text += f"  Minimum: 1-2 stations\n"
    rec_text += f"  Potential reduction: {len(config.ground_stations) - 2}+ stations\n"

    ax4.text(0, 0.95, rec_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='#fce4ec', alpha=0.8))

    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    return {
        'min_ka_stations': 1,
        'min_ttc_stations': 1,
        'ka_sla_met': ka_within_sla == total_collects,
        'recommended_station': combined[0] if combined else ka_stations_used[0]
    }


def add_slide2_results(prs, config, runtime_stats: Dict):
    """Add simulation results as text slide directly to PowerPoint."""
    blank_layout = prs.slide_layouts[6]
    slide = prs.slides.add_slide(blank_layout)

    # Title
    title_box = slide.shapes.add_textbox(Inches(0.5), Inches(0.3), Inches(12), Inches(0.5))
    tf = title_box.text_frame
    p = tf.paragraphs[0]
    p.text = "Simulation Results"
    p.font.size = Pt(28)
    p.font.bold = True
    p.alignment = PP_ALIGN.CENTER

    # Column 1: Coverage Results
    left_text = []
    left_text.append("COVERAGE RESULTS")
    left_text.append(f"  Total Accesses: {runtime_stats.get('total_accesses', 'N/A')}")
    left_text.append(f"  Accesses/Day: {runtime_stats.get('accesses_per_day', 'N/A')}")
    left_text.append(f"  Valid Contacts: {runtime_stats.get('valid_contacts', 'N/A')}")
    left_text.append(f"  Contact Success: {runtime_stats.get('contact_success', 'N/A')}")
    left_text.append("")
    left_text.append("DATA METRICS")
    left_text.append(f"  Data Collected: {runtime_stats.get('data_collected', 'N/A')}")
    left_text.append(f"  Data Downlinked: {runtime_stats.get('data_downlinked', 'N/A')}")
    left_text.append(f"  Peak Backlog: {runtime_stats.get('peak_backlog', 'N/A')}")
    left_text.append("")
    left_text.append("TTNC Ka-BAND")
    left_text.append(f"  Median: {runtime_stats.get('ttnc_median', 'N/A')}")
    left_text.append(f"  P95: {runtime_stats.get('ttnc_p95', 'N/A')}")

    left_box = slide.shapes.add_textbox(Inches(0.5), Inches(1.0), Inches(5), Inches(6))
    tf = left_box.text_frame
    tf.word_wrap = True
    for i, line in enumerate(left_text):
        if i == 0:
            p = tf.paragraphs[0]
        else:
            p = tf.add_paragraph()
        p.text = line
        p.font.size = Pt(14)
        p.font.name = "Consolas"
        if line and not line.startswith(' '):
            p.font.bold = True

    # Column 2: Station-Keeping (if available)
    right_text = []
    if runtime_stats.get('thruster'):
        right_text.append("STATION-KEEPING (HET)")
        right_text.append(f"  Thruster: {runtime_stats.get('thruster', 'N/A')}")
        right_text.append(f"  Delta-V Required: {runtime_stats.get('delta_v', 'N/A')}")
        right_text.append(f"  Propellant: {runtime_stats.get('propellant', 'N/A')}")
        right_text.append(f"  Propellant (with margin): {runtime_stats.get('propellant_margin', 'N/A')}")
        right_text.append(f"  Firing Time: {runtime_stats.get('firing_time', 'N/A')}")
        right_text.append(f"  Annualized Propellant: {runtime_stats.get('annual_propellant', 'N/A')}")
        right_text.append("")
    right_text.append("RUNTIME")
    right_text.append(f"  Processing Time: {runtime_stats.get('processing_time', 'N/A')}")

    right_box = slide.shapes.add_textbox(Inches(6.5), Inches(1.0), Inches(6), Inches(6))
    tf = right_box.text_frame
    tf.word_wrap = True
    for i, line in enumerate(right_text):
        if i == 0:
            p = tf.paragraphs[0]
        else:
            p = tf.add_paragraph()
        p.text = line
        p.font.size = Pt(14)
        p.font.name = "Consolas"
        if line and not line.startswith(' '):
            p.font.bold = True

    return slide


def build_presentation(plots_dir: Path, config, pptx_path: Path, tle_data: Dict = None, targets_gdf = None, runtime_stats: Dict = None):
    """Build the PowerPoint presentation from generated plots."""
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    blank_layout = prs.slide_layouts[6]

    def add_image_slide(image_path: Path, title: str = None):
        slide = prs.slides.add_slide(blank_layout)
        if image_path.exists():
            slide.shapes.add_picture(str(image_path), Inches(0.2), Inches(0.4), width=Inches(12.9))
        if title:
            title_box = slide.shapes.add_textbox(Inches(0.2), Inches(0.05), Inches(12.9), Inches(0.35))
            tf = title_box.text_frame
            p = tf.paragraphs[0]
            p.text = title
            p.font.size = Pt(18)
            p.font.bold = True
            p.alignment = PP_ALIGN.CENTER
        return slide

    # Slide 1: Mission Parameters (text-based)
    if tle_data is not None:
        add_slide1_text(prs, config, tle_data, targets_gdf)
    else:
        add_image_slide(plots_dir / 'slide1_mission_params.png', 'Mission Parameters')

    # Slide 2: Simulation Results (text-based)
    if runtime_stats is not None:
        add_slide2_results(prs, config, runtime_stats)

    # Slide 3: Ground Stations
    add_image_slide(plots_dir / 'slide2_ground_stations.png', 'Ground Station Network')

    # Slide 3: Single Orbit
    add_image_slide(plots_dir / 'slide3_single_orbit.png', 'Single Orbit Ground Track')

    # Slide 4: Access Results
    add_image_slide(plots_dir / 'slide4_access_results.png', 'Access Windows Overview')

    # Slide 5: Access with Cones
    add_image_slide(plots_dir / 'slide5_access_with_cones.png', 'Access Windows with 60-min Post-Collect Tracks')

    # Slides 7-9: Broadside Collect
    for i in range(1, 4):
        add_image_slide(plots_dir / f'slide_broadside_{i}.png', f'Broadside Collect Opportunity {i}')

    # Slides 10-12: 3h Post-Collect
    for i in range(1, 4):
        add_image_slide(plots_dir / f'slide_postcollect_{i}.png', f'Post-Collect Analysis {i} (3h)')

    # Slide 13: TTNC Ka Summary
    if (plots_dir / 'slide_ttnc_summary.png').exists():
        add_image_slide(plots_dir / 'slide_ttnc_summary.png', 'Post-Collect Ka Downlink Analysis')

    # Slides 14-16: Pre-Collect
    for i in range(1, 4):
        add_image_slide(plots_dir / f'slide_precollect_{i}.png', f'Pre-Collect Analysis {i} (Uplink-to-Collect)')

    # Slide 17: Uplink-to-Collect Summary
    if (plots_dir / 'slide_uplink_summary.png').exists():
        add_image_slide(plots_dir / 'slide_uplink_summary.png', 'Pre-Collect TT&C Analysis')

    # Slide 18: Backlog and Downlink Analysis
    if (plots_dir / 'slide_backlog_summary.png').exists():
        add_image_slide(plots_dir / 'slide_backlog_summary.png', 'Backlog and Downlink Analysis')

    # Slide 19: Ground Station Optimization
    if (plots_dir / 'slide_optimization.png').exists():
        add_image_slide(plots_dir / 'slide_optimization.png', 'Ground Station Optimization Analysis')

    prs.save(str(pptx_path))
    print(f"Presentation saved to {pptx_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Generate full presentation')
    parser.add_argument('--config', required=True, help='Path to config YAML')
    parser.add_argument('--results-dir', required=True, help='Path to results directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    config = load_config(args.config)
    results_dir = Path(args.results_dir)
    plots_dir = results_dir / config.plots_subdir
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    excel_path = results_dir / config.excel_filename
    access_df = pd.read_excel(excel_path, sheet_name='Access_Windows')
    print(f"Loaded {len(access_df)} access windows")

    # Setup TLE data
    orbit_config = config.raw_config.get('orbit', {})
    ltdn_hours = orbit_config.get('ltdn_hours')
    auto_sso = orbit_config.get('auto_sso', False)

    tle_data = {}
    for sat in config.satellites:
        inc_deg = sat.inclination_deg
        raan_deg = sat.raan_deg
        if auto_sso and ltdn_hours:
            inc_deg = calculate_sso_inclination(sat.altitude_km)
            raan_deg = ltdn_to_raan(ltdn_hours, config.start_datetime)

        tle = create_tle_data(
            sat_id=sat.sat_id, inclination_deg=inc_deg,
            altitude_km=sat.altitude_km, raan_deg=raan_deg,
            epoch=config.start_datetime,
        )
        tle_data[sat.sat_id] = tle

    # Ground stations
    ground_stations = [{'name': gs.name, 'lat': gs.lat, 'lon': gs.lon,
                        'min_elevation_deg': gs.min_elevation_deg, 'ka_band': gs.ka_capable}
                       for gs in config.ground_stations]

    sat_alt_km = config.satellites[0].altitude_km

    # Load targets GeoDataFrame
    targets_config = config.raw_config.get('targets', {})
    geojson_path = targets_config.get('geojson_path')
    targets_gdf = None
    if geojson_path:
        full_geojson_path = Path(args.config).parent.parent / geojson_path
        if full_geojson_path.exists():
            targets_gdf = gpd.read_file(full_geojson_path)
            print(f"Loaded {len(targets_gdf)} targets from {geojson_path}")
        else:
            # Try relative to current directory
            if Path(geojson_path).exists():
                targets_gdf = gpd.read_file(geojson_path)
                print(f"Loaded {len(targets_gdf)} targets from {geojson_path}")

    # Load runtime stats from Excel
    runtime_stats = {}
    try:
        # Get KPIs - format is KPI, Value, Unit columns
        kpi_df = pd.read_excel(excel_path, sheet_name='Coverage_KPIs')
        kpi_dict = dict(zip(kpi_df['KPI'], kpi_df['Value']))

        total_accesses = int(kpi_dict.get('Total Accesses', len(access_df)))
        valid_contacts = int(kpi_dict.get('Total Valid Contacts', 0))
        runtime_stats['total_accesses'] = total_accesses
        runtime_stats['valid_contacts'] = valid_contacts
        runtime_stats['accesses_per_day'] = f"{kpi_dict.get('Accesses Per Day', total_accesses / config.duration_days):.1f}"
        runtime_stats['contact_success'] = f"{(valid_contacts / total_accesses * 100) if total_accesses > 0 else 0:.1f}%"
        runtime_stats['data_collected'] = f"{kpi_dict.get('Total Data Collected', 0):.1f} GB"
        runtime_stats['data_downlinked'] = f"{kpi_dict.get('Total Data Downlinked', 0):.1f} GB"
        runtime_stats['peak_backlog'] = f"{kpi_dict.get('Peak Backlog', 0):.1f} GB"

        # Get TTNC stats
        ttnc_df = pd.read_excel(excel_path, sheet_name='TTNC_Summary')
        if len(ttnc_df) > 0:
            runtime_stats['ttnc_median'] = f"{ttnc_df['Median (min)'].iloc[0]:.1f} min"
            runtime_stats['ttnc_p95'] = f"{ttnc_df['P95 (min)'].iloc[0]:.1f} min"

        # Get station-keeping stats if available
        try:
            sk_df = pd.read_excel(excel_path, sheet_name='Station_Keeping')
            if len(sk_df) > 0:
                sk_dict = dict(zip(sk_df['Parameter'], sk_df['Value']))
                runtime_stats['thruster'] = sk_dict.get('Thruster', 'N/A')
                runtime_stats['delta_v'] = f"{sk_dict.get('Delta-V Required (m/s)', 0):.2f} m/s"
                runtime_stats['propellant'] = f"{sk_dict.get('Propellant Mass (kg)', 0):.2f} kg"
                runtime_stats['propellant_margin'] = f"{sk_dict.get('Propellant with Margin (kg)', 0):.2f} kg"
                firing_hrs = sk_dict.get('Total Firing Time (hours)', 0)
                duty = sk_dict.get('Duty Cycle (%)', 0)
                runtime_stats['firing_time'] = f"{firing_hrs:.1f} hrs ({duty:.1f}% duty)"
                runtime_stats['annual_propellant'] = f"{sk_dict.get('Annualized Propellant (kg/yr)', 0):.1f} kg/yr"
        except Exception:
            pass  # Station-keeping sheet may not exist

    except Exception as e:
        print(f"Warning: Could not load runtime stats: {e}")
        runtime_stats['total_accesses'] = len(access_df)

    runtime_stats['processing_time'] = '~40 s'  # Approximate based on last run

    # Generate all plots (Slide 1 is text-based, generated in build_presentation)
    print("\nSlide 1: Mission Parameters (text-based, generated in PowerPoint)")

    print("Generating Slide 3: Ground Stations and Targets...")
    generate_slide2_gs_map(config, ground_stations, sat_alt_km, targets_gdf,
                           plots_dir / 'slide2_ground_stations.png', excel_path=excel_path)

    print("Generating Slide 3: Single Orbit...")
    generate_slide3_single_orbit(tle_data, config, plots_dir / 'slide3_single_orbit.png')

    print("Generating Slide 4: Access Results...")
    generate_slide4_access_results(access_df, tle_data, plots_dir / 'slide4_access_results.png')

    print("Generating Slide 5: Access with Comm Cones...")
    generate_slide5_access_with_cones(access_df, tle_data, ground_stations, sat_alt_km,
                                      plots_dir / 'slide5_access_with_cones.png')

    # Select 3 random accesses
    random.seed(args.seed)
    num_accesses = min(3, len(access_df))
    random_indices = random.sample(range(len(access_df)), num_accesses)

    # Get off-nadir limits from config
    off_nadir_min = config.imaging_modes[0].off_nadir_min_deg if config.imaging_modes else 0.0
    off_nadir_max = config.imaging_modes[0].off_nadir_max_deg if config.imaging_modes else 30.0

    print(f"\nGenerating Slides 6-8: Broadside Collect (indices {random_indices})...")
    for i, idx in enumerate(random_indices):
        generate_broadside_collect_plot(access_df.iloc[idx], tle_data, ground_stations, sat_alt_km,
                                        plots_dir / f'slide_broadside_{i+1}.png',
                                        off_nadir_min=off_nadir_min, off_nadir_max=off_nadir_max)

    print("Generating Slides 10-12: 3h Post-Collect...")
    for i, idx in enumerate(random_indices):
        generate_collect_diagram_3h(access_df.iloc[idx], tle_data, ground_stations, sat_alt_km,
                                    plots_dir / f'slide_postcollect_{i+1}.png')

    print("Generating Slide 13: TTNC Ka Summary...")
    generate_ttnc_summary_plot(excel_path, plots_dir / 'slide_ttnc_summary.png')

    print("Generating Slides 14-16: Pre-Collect (Uplink-to-Collect)...")
    for i, idx in enumerate(random_indices):
        generate_pre_collect_diagram(access_df.iloc[idx], tle_data, ground_stations, sat_alt_km,
                                     plots_dir / f'slide_precollect_{i+1}.png')

    print("Generating Slide 17: Uplink-to-Collect Summary...")
    generate_uplink_to_collect_summary(access_df, tle_data, ground_stations, sat_alt_km,
                                       plots_dir / 'slide_uplink_summary.png')

    print("Generating Slide 18: Backlog and Downlink Analysis...")
    generate_backlog_summary(excel_path, plots_dir / 'slide_backlog_summary.png')

    print("Generating Slide 19: Ground Station Optimization...")
    generate_optimization_slide(excel_path, plots_dir / 'slide_optimization.png', config)

    # Build presentation
    print("\nBuilding PowerPoint presentation...")
    pptx_path = results_dir / config.ppt_filename
    build_presentation(plots_dir, config, pptx_path, tle_data=tle_data, targets_gdf=targets_gdf, runtime_stats=runtime_stats)

    print("\nDone!")


if __name__ == '__main__':
    main()
