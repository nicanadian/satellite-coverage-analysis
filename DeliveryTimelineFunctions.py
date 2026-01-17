# In DeliveryTimelineFunctions.py

import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import pytz
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from shapely.geometry import Point
from skyfield.api import load, EarthSatellite, wgs84
from pyproj import Geod
from shapely.geometry import Polygon
from datetime import timezone
from shapely.prepared import prep

def generate_tle(epoch, inclination_deg, altitude_km, sat_num, raan_deg=0):
    """
    Generate a TLE for a satellite given basic orbital parameters
    
    Parameters:
    -----------
    epoch : datetime
        Epoch time for the TLE
    inclination_deg : float
        Orbital inclination in degrees
    altitude_km : float
        Mean orbital altitude in kilometers
    sat_num : str
        Satellite catalog number (5 digits)
    raan_deg : float, optional
        Right Ascension of Ascending Node in degrees (default 0)
    
    Returns:
    --------
    line1, line2 : tuple of strings
        The two lines of the TLE
    """
    
    # Constants
    earth_radius = 6378.137  # km
    earth_mu = 398600.4418  # km^3/s^2
    
    # Calculate semi-major axis and mean motion
    semi_major_axis = earth_radius + altitude_km
    mean_motion = np.sqrt(earth_mu / semi_major_axis**3) * 86400 / (2 * np.pi)  # revs per day
    
    # Convert epoch to TLE format
    year = epoch.year % 100
    day_of_year = epoch.timetuple().tm_yday
    epoch_str = f"{year:02d}{day_of_year:03d}.00000000"
    
    # Line 1 components
    line1_template = "1 {sat_num}U 20001A   {epoch} .00000000  00000-0  50000-4 0  9990"
    line1 = line1_template.format(
        sat_num=sat_num,
        epoch=epoch_str
    )
    
    # Line 2 components
    line2_template = "2 {sat_num} {inc:8.4f} {raan:8.4f} 0001000   0.0000 {arg_perigee:8.4f} {mean_motion:11.8f}    10"
    line2 = line2_template.format(
        sat_num=sat_num,
        inc=inclination_deg,
        raan=raan_deg,
        arg_perigee=0.0,  # Circular orbit
        mean_motion=mean_motion
    )
    
    # Calculate checksums
    line1 = line1[:-1] + str(calculate_checksum(line1))
    line2 = line2[:-1] + str(calculate_checksum(line2))
    
    return line1, line2

def calculate_checksum(line):
    """Calculate the checksum for a TLE line"""
    sum = 0
    for i in range(68):  # TLE lines are 69 characters including checksum
        if line[i].isdigit():
            sum += int(line[i])
        elif line[i] == '-':
            sum += 1
    return sum % 10

def find_next_contact(spacecraft_tles, sat_id, start_time, comm_cone_wgs84, max_search_time=pd.Timedelta(hours=3)):
    """
    Find the next ground station contact start and end times for a satellite.

    Parameters
    ----------
    spacecraft_tles : dict
        Dictionary containing TLE data for satellites.
    sat_id : int or str
        Satellite ID number.
    start_time : datetime.datetime
        Start time to begin searching for contact (must be timezone-aware).
    comm_cone_wgs84 : shapely.geometry
        Communication cone geometry in WGS84 coordinates.
    max_search_time : pd.Timedelta, optional
        Maximum time to search for contact, default is 3 hours.

    Returns
    -------
    tuple of (pd.Timestamp or None, pd.Timestamp or None)
        Start and end times of the next contact, or (None, None) if no contact is found.
    """
    if start_time.tzinfo is None:
        raise ValueError("start_time must be timezone-aware")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Convert start_time to Pandas Timestamp and floor to the nearest second
        start_time_ts = pd.Timestamp(start_time).floor('s')
        end_time_ts = (start_time_ts + max_search_time).floor('s')
        
        # Create a date range from start_time to end_time with 15-second intervals
        times = pd.date_range(start=start_time_ts,
                              end=end_time_ts, 
                              freq='15s')
        
        # Ensure all times are in UTC
        if times.tz is None:
            times = times.tz_localize('UTC')
        elif times.tz != timezone.utc:
            times = times.tz_convert('UTC')
        
        ts = load.timescale()
        # Convert Pandas Timestamps to Skyfield Time objects
        t = ts.from_datetimes([t.to_pydatetime() for t in times])

    # Initialize the satellite object
    sat = EarthSatellite(spacecraft_tles[sat_id]['line1'],
                        spacecraft_tles[sat_id]['line2'],
                        'sat', ts)
    positions = sat.at(t)
    
    # Get geographic positions
    subpoints = wgs84.geographic_position_of(positions)
    lats = subpoints.latitude.degrees
    lons = subpoints.longitude.degrees
    
    # Prepare coordinates for geometric operations
    coords = np.column_stack((lons, lats)).astype(np.float64)
    
    # Prepare the communication cone geometry
    prepared_cone = prep(comm_cone_wgs84)
    
    # Create Point objects for each coordinate
    points = [Point(x, y) for x, y in coords]
    
    # Determine which points are within the communication cone
    in_range = np.array([prepared_cone.contains(point) for point in points])
    
    # Find all contact periods
    contacts = []
    in_contact = False
    for idx, val in enumerate(in_range):
        if val and not in_contact:
            # Start of a contact
            contact_start_time = times[idx]
            in_contact = True
        elif not val and in_contact:
            # End of a contact
            contact_end_time = times[idx]
            contacts.append((contact_start_time, contact_end_time))
            in_contact = False
    if in_contact:
        # If contact is ongoing at the end of the search window
        contact_end_time = times[-1]
        contacts.append((contact_start_time, contact_end_time))
    
    if not contacts:
        return (None, None)
    
    # Return the first contact
    return contacts[0]

def get_satellite_position(spacecraft_tles, sat_id, time):
    """
    Get satellite position at a specific time.
    
    Args:
        spacecraft_tles: Dictionary of spacecraft TLEs
        sat_id: Satellite ID
        time: Timestamp (timezone-aware)
    
    Returns:
        lat, lon, alt of satellite
    """
    from skyfield.api import utc, EarthSatellite, load
    import pytz
    
    # Initialize timescale
    ts = load.timescale()
    
    # Convert to Python datetime if it's a pandas Timestamp
    if isinstance(time, pd.Timestamp):
        time = time.to_pydatetime()

    # Ensure the datetime is timezone-aware using Skyfield's utc
    if time.tzinfo is None:
        time = time.replace(tzinfo=utc)
    elif time.tzinfo != utc:
        time = time.astimezone(utc)

    # Initialize timescale
    ts = load.timescale()
    # Get time from timescale
    t = ts.from_datetime(time)
    
    # Create satellite from TLE
    line1 = spacecraft_tles[sat_id]['line1']
    line2 = spacecraft_tles[sat_id]['line2']
    satellite = EarthSatellite(line1, line2, f'Capella-{sat_id}', ts)
    
    # Convert timestamp to datetime with UTC timezone
    if isinstance(time, pd.Timestamp):
        # Convert to Python datetime and ensure UTC timezone
        time = time.to_pydatetime().replace(tzinfo=pytz.UTC)
    
    # Calculate position
    geocentric = satellite.at(t)
    subpoint = geocentric.subpoint()
    
    return subpoint.latitude.degrees, subpoint.longitude.degrees, subpoint.elevation.km

def calculate_slew_duration(tle_dict, sat_id, imaging_stop_time, target_lat, target_lon, gs_lat, gs_lon, slew_rate, mode_switch_time):
    """Calculate slew duration based on angular change between target and ground station"""
    
    # Get satellite position at imaging stop
    sat_pos = get_satellite_position(tle_dict, sat_id, imaging_stop_time)
    sat_lat, sat_lon, sat_alt = sat_pos
    
    # Convert positions to 3D vectors
    def to_unit_vector(lat, lon):
        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)
        x = np.cos(lat_rad) * np.cos(lon_rad)
        y = np.cos(lat_rad) * np.sin(lon_rad)
        z = np.sin(lat_rad)
        return np.array([x, y, z])
    
    # Calculate vectors
    sat_to_target = to_unit_vector(target_lat, target_lon) - to_unit_vector(sat_lat, sat_lon)
    sat_to_gs = to_unit_vector(gs_lat, gs_lon) - to_unit_vector(sat_lat, sat_lon)
    
    # Normalize vectors
    sat_to_target = sat_to_target / np.linalg.norm(sat_to_target)
    sat_to_gs = sat_to_gs / np.linalg.norm(sat_to_gs)
    
    # Calculate angle between vectors (in degrees)
    angle = np.arccos(np.clip(np.dot(sat_to_target, sat_to_gs), -1.0, 1.0))
    angle_deg = np.degrees(angle)
    
    # Calculate slew duration
    slew_time = angle_deg / slew_rate  # degrees / (degrees/second) = seconds

    total_duration = max(slew_time, mode_switch_time)
    
    return total_duration

def create_communication_cone(lon, lat, min_elevation_angle, sat_alt=None):
    """
    Create a communication cone for a ground station
    
    Parameters
    ----------
    lon : float
        Ground station longitude
    lat : float
        Ground station latitude
    min_elevation_angle : float
        Minimum elevation angle in degrees
    sat_alt : float, optional
        Satellite altitude in meters
        
    Returns
    -------
    shapely.geometry
        Communication cone geometry in WGS84 coordinates
    """
    if sat_alt is None:
        sat_alt = 600000  # Default 600km in meters

    geod = Geod(ellps='WGS84')

    # Calculate maximum ground range
    earth_radius = 6378137  # meters
    elevation_rad = np.radians(min_elevation_angle)
    # Calculate central angle
    central_angle = np.arccos(earth_radius * np.cos(elevation_rad) / 
                            (earth_radius + sat_alt)) - elevation_rad
    
    # Calculate ground range
    max_ground_range = earth_radius * central_angle
    
    azimuths = np.linspace(0, 360, 360)
    lats = []
    lons = []

    for azimuth in azimuths:
        lon2, lat2, _ = geod.fwd(float(lon), float(lat), azimuth, max_ground_range)
        # Handle longitude wrapping for Eastern hemisphere
        if lon > 0:  # If input longitude is in Eastern hemisphere
            if lon2 < 0:
                lon2 += 360
        else:  # If input longitude is in Western hemisphere
            if lon2 > 180:
                lon2 -= 360
        lats.append(lat2)
        lons.append(lon2)

    # Close the polygon
    lats.append(lats[0])
    lons.append(lons[0])

    return Polygon(zip(lons, lats)) 

def convert_wgs_to_utm(lon, lat):
    """Convert WGS84 coordinates to UTM zone number"""
    utm_zone = int((lon + 180) / 6) + 1
    if lat >= 0:
        return 32600 + utm_zone
    return 32700 + utm_zone

def calculate_ground_track(spacecraft_tles, sat_id, start_time, duration=pd.Timedelta(minutes=10)):
    """
    Calculate satellite ground track for a given time period
    
    Parameters
    ----------
    spacecraft_tles : dict
        Dictionary containing TLE data for satellites
    sat_id : int or str
        Satellite ID number
    start_time : datetime or Timestamp
        Start time for ground track calculation
    duration : Timedelta, optional
        Duration to calculate ground track for, default 10 minutes
        
    Returns
    -------
    list
        List of (longitude, latitude) tuples representing ground track
    """
    # Input validation
    if sat_id not in spacecraft_tles:
        raise KeyError(f"Satellite ID {sat_id} not found in TLE data")
    
    if not isinstance(start_time, (pd.Timestamp, pd.DatetimeIndex)):
        try:
            start_time = pd.Timestamp(start_time)
        except:
            raise ValueError("start_time must be a datetime or pandas Timestamp")

    # Generate array of times to check (every 10 seconds)
    times = pd.date_range(start=start_time, 
                         end=start_time + duration, 
                         freq='10s')
    
    # Convert times to Skyfield format
    ts = load.timescale()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t = ts.from_datetimes([pd.Timestamp(t).floor('s').to_pydatetime() for t in times])
    
    # Get satellite positions
    sat = EarthSatellite(spacecraft_tles[sat_id]['line1'],
                        spacecraft_tles[sat_id]['line2'],
                        'sat', ts)
    positions = sat.at(t)
    
    # Convert to geographic coordinates
    subpoints = wgs84.geographic_position_of(positions)
    lats = subpoints.latitude.degrees
    lons = subpoints.longitude.degrees
    
    return list(zip(lons, lats))

def calculate_look_angle(sat_lats, sat_lons, sat_alts, aoi_lat, aoi_lon):
    """
    Calculate look angles between satellite positions and an AOI,
    accounting for Earth's obstruction
    
    Parameters
    ----------
    sat_lats, sat_lons, sat_alts : array-like or float
        Satellite position coordinates and altitudes
    aoi_lat, aoi_lon : float
        Target coordinates
    
    Returns
    -------
    float or array
        Look angles in degrees (NaN if beyond horizon)
    """
    # Convert inputs to numpy arrays if they aren't already
    sat_lats = np.atleast_1d(sat_lats)
    sat_lons = np.atleast_1d(sat_lons)
    sat_alts = np.atleast_1d(sat_alts)
    
    # Convert to radians for calculations
    sat_lats_rad = np.radians(sat_lats)
    sat_lons_rad = np.radians(sat_lons)
    aoi_lat_rad = np.radians(aoi_lat)
    aoi_lon_rad = np.radians(aoi_lon)
    
    # Earth radius in km
    R = 6378.137
    
    # Calculate great circle distance
    dlat = sat_lats_rad - aoi_lat_rad
    dlon = sat_lons_rad - aoi_lon_rad
    
    # Haversine formula
    a = np.sin(dlat/2)**2 + np.cos(sat_lats_rad) * np.cos(aoi_lat_rad) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    ground_distance = R * c
    
    # Calculate horizon distance for each satellite position
    horizon_distance = R * np.arccos(R / (R + sat_alts))
    
    # Create mask for points beyond horizon
    beyond_horizon = ground_distance > horizon_distance
    
    # Calculate look angle
    look_angles = np.degrees(np.arctan2(sat_alts, ground_distance))
    
    # Mask out points beyond horizon
    look_angles = np.where(beyond_horizon, np.nan, look_angles)
    
    # If input was a single value, return a single value
    if len(look_angles) == 1:
        return float(look_angles[0])
    return look_angles

def overpass_to_swath(ops_gdf, aoi_pol, la_min, la_max):
    import geopandas as gpd
    import pandas as pd
    
    lon = aoi_pol.centroid.x
    lat = aoi_pol.centroid.y
        
    epsg_code = convert_wgs_to_utm(lon, lat)

    swath_list = []  # Use a list to collect swaths

    for idx in range(0, len(ops_gdf)):
        opty_ls = ops_gdf.geometry[idx]
        opty_elev = ops_gdf.Elevation[idx]*1000

        near_edge_dist, ia_near = la_to_ia_to_swathdist(opty_elev, la_min)
        near_edge = gpd.GeoSeries(opty_ls, crs=4326).to_crs(int(epsg_code)).buffer(near_edge_dist).to_crs(4326)

        far_edge_dist, ia_far = la_to_ia_to_swathdist(opty_elev, la_max)
        far_edge = gpd.GeoSeries(opty_ls, crs=4326).to_crs(int(epsg_code)).buffer(far_edge_dist, cap_style=2).to_crs(4326)

        swath_tmp = far_edge.difference(near_edge, align=False)
        swath_list.append(swath_tmp)  # Add to list instead of appending

    # Combine all swaths using concat
    swath_gsr = pd.concat(swath_list) if swath_list else gpd.GeoSeries()

    # Convert to GeoJSON and GDF
    swath_geoj = swath_gsr.__geo_interface__
    swath_gdf = gpd.GeoDataFrame(swath_gsr, columns=['geometry'], crs=4326)
    
    return swath_gdf, swath_geoj, far_edge

def la_to_ia_to_swathdist(elev, la):
    from math import tan, pi, asin, sin, degrees, radians

    radius_earth = 6378145
    la_rad = radians(la)
    ia = asin((1 + elev/radius_earth)*sin(la_rad))
    theta = ia-la_rad
    swath_dist = theta * radius_earth

    return(swath_dist, degrees(ia))

def calculate_squint_angles(spacecraft_tles, sat_id, target_lat, target_lon, times, plot=False):
    """
    Calculate squint angles at specified times using 3D vectors.
    """
    ts = load.timescale()
    
    def get_track_direction(time, dt=1):
        """Get ground track direction at a point by looking slightly ahead"""
        t1 = ts.from_datetime(time - pd.Timedelta(seconds=dt))
        t2 = ts.from_datetime(time + pd.Timedelta(seconds=dt))
        
        pos1 = wgs84.geographic_position_of(satellite.at(t1))
        pos2 = wgs84.geographic_position_of(satellite.at(t2))
        
        dlat = pos2.latitude.degrees - pos1.latitude.degrees
        dlon = pos2.longitude.degrees - pos1.longitude.degrees
        
        # Handle longitude wrap-around
        if dlon > 180:
            dlon -= 360
        elif dlon < -180:
            dlon += 360
            
        direction = np.array([dlon, dlat])
        return direction / np.linalg.norm(direction)
    
    def plot_geometry(time, sat_lat, sat_lon, target_lat, target_lon, track_direction, broadside, squint_deg):
        plt.figure(figsize=(12, 8))
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.coastlines()
        ax.gridlines(draw_labels=True)
        
        # Plot ground track (±1 minute)
        track_times = pd.date_range(
            time - pd.Timedelta(minutes=2),
            time + pd.Timedelta(minutes=2),
            freq='10S'
        )
        t_track = ts.from_datetimes(track_times.to_pydatetime())
        positions = satellite.at(t_track)
        subpoints = wgs84.geographic_position_of(positions)
        track_lats = subpoints.latitude.degrees
        track_lons = subpoints.longitude.degrees
        ax.plot(track_lons, track_lats, 'b-', transform=ccrs.PlateCarree(), label='Ground Track')
        
        # Plot satellite nadir point
        ax.plot(sat_lon, sat_lat, 'ro', transform=ccrs.PlateCarree(), label='Satellite Nadir')
        
        # Plot target
        ax.plot(target_lon, target_lat, 'go', transform=ccrs.PlateCarree(), label='Target')
        
        # Plot velocity direction
        scale = 2.0
        vel_lon = sat_lon + track_direction[0] * scale
        vel_lat = sat_lat + track_direction[1] * scale
        ax.plot([sat_lon, vel_lon], [sat_lat, vel_lat], 'y--', 
                transform=ccrs.PlateCarree(), label='Velocity Direction')
        
        # Plot broadside direction
        broadside_lon = sat_lon + broadside[0] * scale
        broadside_lat = sat_lat + broadside[1] * scale
        ax.plot([sat_lon, broadside_lon], [sat_lat, broadside_lat], 'k--', 
                transform=ccrs.PlateCarree(), label='Broadside Direction')
        
        # Plot line to target
        ax.plot([sat_lon, target_lon], [sat_lat, target_lat], 'r--', 
                transform=ccrs.PlateCarree(), label='To Target')
        
        # Set extent
        buffer = 5  # degrees
        ax.set_extent([
            min(sat_lon, target_lon) - buffer,
            max(sat_lon, target_lon) + buffer,
            min(sat_lat, target_lat) - buffer,
            max(sat_lat, target_lat) + buffer
        ])
        
        plt.title(f'Squint Geometry for Capella-{sat_id} at {time}\nCalculated Squint: {squint_deg:.2f}°')
        plt.legend()
        plt.show()
    
    # Initialize satellite
    satellite = EarthSatellite(spacecraft_tles[sat_id]['line1'],
                             spacecraft_tles[sat_id]['line2'],
                             'sat', ts)
    
    squint_angles = {}
    
    for time in times:
        # Get satellite position at this time
        t = ts.from_datetime(time)
        geocentric = satellite.at(t)
        
        # Get satellite position in geographic coordinates
        subpoint = wgs84.geographic_position_of(geocentric)
        sat_lat = subpoint.latitude.degrees
        sat_lon = subpoint.longitude.degrees
        
        # Get ground track direction
        track_direction = get_track_direction(time)
        
        # Calculate target direction vector
        target_direction = np.array([target_lon - sat_lon, target_lat - sat_lat])
        target_direction = target_direction / np.linalg.norm(target_direction)
        
        # Calculate broadside options (perpendicular to track direction)
        broadside1 = np.array([-track_direction[1], track_direction[0]])
        broadside2 = np.array([track_direction[1], -track_direction[0]])
        
        # Choose broadside that's on same side as target
        dot1 = np.dot(broadside1, target_direction)
        dot2 = np.dot(broadside2, target_direction)
        broadside = broadside1 if dot1 > dot2 else broadside2
        
        # Calculate squint angle
        squint_angle = np.arccos(np.clip(np.dot(broadside, target_direction), -1.0, 1.0))
        squint_deg = np.degrees(squint_angle)
        
        # Determine sign based on whether target is forward or backward of broadside
        if np.dot(target_direction, track_direction) < 0:
            squint_deg = -squint_deg
        
        squint_angles[time] = squint_deg
        
        # Plot geometry if requested
        if plot:
            plot_geometry(time, sat_lat, sat_lon, target_lat, target_lon, 
                         track_direction, broadside, squint_deg)
    
    return squint_angles

def find_broadside_and_target_squint(spacecraft_tles, sat_id, target_lat, target_lon, start_time, end_time, target_squint):
    """
    Find the times of broadside (minimum squint) and target maximum squint during an access period.
    
    Parameters
    ----------
    spacecraft_tles : dict
        Dictionary containing TLE data for satellites
    sat_id : int
        Satellite ID number
    target_lat : float
        Target latitude in degrees
    target_lon : float
        Target longitude in degrees
    start_time : datetime
        Start time of access
    end_time : datetime
        End time of access
    target_squint : float
        Target maximum squint angle in degrees
        
    Returns
    -------
    dict
        Dictionary containing broadside and target squint information
    """
    # Create time array with 1s intervals
    times = pd.date_range(start=start_time, end=end_time, freq='s')
    
    # Calculate squint angles for all times
    squint_angles = calculate_squint_angles(
        spacecraft_tles,
        sat_id,
        target_lat,
        target_lon,
        times,
        plot=False
    )
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame.from_dict(squint_angles, orient='index', columns=['squint'])
    df.index.name = 'time'
    
    # Find minimum absolute squint (closest to broadside)
    broadside_idx = df['squint'].abs().idxmin()
    broadside_squint = df.loc[broadside_idx, 'squint']
    
    # Find time closest to target squint angle
    df['target_diff'] = (df['squint'] - target_squint).abs()
    target_idx = df['target_diff'].idxmin()
    target_squint = df.loc[target_idx, 'squint']
    
    return {
        'broadside_time': broadside_idx,
        'broadside_angle': broadside_squint,
        'target_squint_time': target_idx,
        'target_squint_angle': target_squint
    }

def format_duration(seconds):
    """Convert seconds to a readable duration string"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def analyze_contact_timelines(df_no_ksat, df_with_ksat):
    # Function to calculate time difference in minutes
    def get_contact_delay(row, gs_name):
        start_time = pd.to_datetime(row['Start_Time'])
        contact_start = pd.to_datetime(row[f'Next_{gs_name}_Contact_Start'])
        if pd.notna(contact_start):
            return (contact_start - start_time).total_seconds() / 60  # Convert to minutes
        return None

    # Initialize results for both datasets
    results_no_ksat = []
    results_with_ksat = []

    # Process each dataset
    for dataset_name, df in [('No KSAT', df_no_ksat), ('With KSAT', df_with_ksat)]:
        # Get list of ground stations for this dataset
        gs_cols = [col.replace('Valid_Contact_', '') 
                  for col in df.columns 
                  if col.startswith('Valid_Contact_')]
        
        # Process each valid contact
        for _, row in df.iterrows():
            for gs in gs_cols:
                if row[f'Valid_Contact_{gs}']:
                    delay = get_contact_delay(row, gs)
                    if delay is not None:
                        result = {
                            'Dataset': dataset_name,
                            'Imaging_Mode': row['Imaging_Mode'],
                            'Ground_Station': gs,
                            'Satellite': row['Satellite'],
                            'Collection_Start': row['Start_Time'],
                            'Contact_Delay_Minutes': delay,
                            'Total_Timeline_Minutes': row['Collect_to_Delivery_Timeline'] / 60  # Convert to minutes
                        }
                        
                        if dataset_name == 'No KSAT':
                            results_no_ksat.append(result)
                        else:
                            results_with_ksat.append(result)

    # Convert to DataFrames
    df_results_no_ksat = pd.DataFrame(results_no_ksat)
    df_results_with_ksat = pd.DataFrame(results_with_ksat)

    # Print summary statistics
    print("Summary Statistics for Contact Delays (minutes):")
    print("\nWithout KSAT Ground Stations:")
    print(df_results_no_ksat.groupby('Imaging_Mode')['Contact_Delay_Minutes'].describe().round(2))
    
    print("\nWith KSAT Ground Stations:")
    print(df_results_with_ksat.groupby('Imaging_Mode')['Contact_Delay_Minutes'].describe().round(2))

    # Calculate statistics by ground station
    print("\nContact Delay Statistics by Ground Station (minutes):")
    gs_stats_no_ksat = df_results_no_ksat.groupby('Ground_Station')['Contact_Delay_Minutes'].describe().round(2)
    gs_stats_with_ksat = df_results_with_ksat.groupby('Ground_Station')['Contact_Delay_Minutes'].describe().round(2)
    
    print("\nWithout KSAT:")
    print(gs_stats_no_ksat)
    print("\nWith KSAT:")
    print(gs_stats_with_ksat)

    # Calculate percentage of contacts within different time windows
    time_windows = [30, 60, 90, 120]
    
    print("\nPercentage of Contacts Within Time Windows:")
    for dataset_name, df_results in [('Without KSAT', df_results_no_ksat), 
                                   ('With KSAT', df_results_with_ksat)]:
        print(f"\n{dataset_name}:")
        for window in time_windows:
            pct = (df_results['Contact_Delay_Minutes'] <= window).mean() * 100
            print(f"Contacts within {window} minutes: {pct:.1f}%")

    return df_results_no_ksat, df_results_with_ksat

def analyze_sequential_contacts(df_no_ksat, df_with_ksat):
    def get_contact_info(row, gs_name):
        start_time = pd.to_datetime(row['Start_Time'])
        contact_start = pd.to_datetime(row[f'Next_{gs_name}_Contact_Start'])
        contact_end = pd.to_datetime(row[f'Next_{gs_name}_Contact_End'])
        if pd.notna(contact_start) and row[f'Valid_Contact_{gs_name}']:
            return {
                'contact_start': contact_start,
                'contact_end': contact_end,
                'delay_minutes': (contact_start - start_time).total_seconds() / 60,
                'ground_station': gs_name
            }
        return None

    results_no_ksat = []
    results_with_ksat = []

    # Process each dataset
    for dataset_name, df in [('No KSAT', df_no_ksat), ('With KSAT', df_with_ksat)]:
        # Get list of ground stations for this dataset
        gs_cols = [col.replace('Valid_Contact_', '') 
                  for col in df.columns 
                  if col.startswith('Valid_Contact_')]
        
        # Process each collection opportunity
        for idx, row in df.iterrows():
            # Get all valid contacts for this collection
            valid_contacts = []
            for gs in gs_cols:
                contact_info = get_contact_info(row, gs)
                if contact_info:
                    valid_contacts.append(contact_info)
            
            # Sort contacts by start time
            valid_contacts.sort(key=lambda x: x['contact_start'])
            
            # Record information for each sequential contact
            for contact_order, contact in enumerate(valid_contacts, 1):
                result = {
                    'Dataset': dataset_name,
                    'Collection_ID': idx,
                    'Imaging_Mode': row['Imaging_Mode'],
                    'Satellite': row['Satellite'],
                    'Collection_Start': row['Start_Time'],
                    'Contact_Order': contact_order,
                    'Ground_Station': contact['ground_station'],
                    'Contact_Delay_Minutes': contact['delay_minutes'],
                    'Contact_Start': contact['contact_start'],
                    'Contact_End': contact['contact_end']
                }
                
                if dataset_name == 'No KSAT':
                    results_no_ksat.append(result)
                else:
                    results_with_ksat.append(result)

    # Convert to DataFrames
    df_seq_no_ksat = pd.DataFrame(results_no_ksat)
    df_seq_with_ksat = pd.DataFrame(results_with_ksat)
    
    # Generate statistics for each contact order
    def generate_order_stats(df, max_order=5):
        all_stats = []
        
        for order in range(1, max_order + 1):
            order_data = df[df['Contact_Order'] == order]
            
            if len(order_data) == 0:
                continue
                
            # Overall statistics
            overall_stats = order_data.groupby('Imaging_Mode')['Contact_Delay_Minutes'].agg([
                'count',
                'mean',
                'median',
                'std',
                'min',
                'max'
            ]).round(2)
            
            # Ground station breakdown
            gs_stats = order_data.groupby('Ground_Station').agg({
                'Contact_Delay_Minutes': ['count', 'mean', 'median']
            }).round(2)
            
            # Time windows
            time_windows = [30, 60, 90, 120]
            window_stats = {}
            for window in time_windows:
                pct = (order_data['Contact_Delay_Minutes'] <= window).mean() * 100
                window_stats[f'within_{window}min'] = f"{pct:.1f}%"
            
            all_stats.append({
                'contact_order': order,
                'overall_stats': overall_stats,
                'ground_station_stats': gs_stats,
                'window_stats': window_stats
            })
            
        return all_stats

    # Generate and save statistics
    no_ksat_stats = generate_order_stats(df_seq_no_ksat)
    with_ksat_stats = generate_order_stats(df_seq_with_ksat)

    # Save detailed sequential contact data
    df_seq_no_ksat.to_csv('sequential_contacts_no_ksat.csv', index=False)
    df_seq_with_ksat.to_csv('sequential_contacts_with_ksat.csv', index=False)

    # Save statistics for each contact order
    for i, (no_ksat_stat, with_ksat_stat) in enumerate(zip(no_ksat_stats, with_ksat_stats), 1):
        # Overall statistics
        pd.concat({
            'No_KSAT': no_ksat_stat['overall_stats'],
            'With_KSAT': with_ksat_stat['overall_stats']
        }, axis=1).to_csv(f'contact_{i}_overall_stats.csv')
        
        # Ground station statistics
        pd.concat({
            'No_KSAT': no_ksat_stat['ground_station_stats'],
            'With_KSAT': with_ksat_stat['ground_station_stats']
        }, axis=1).to_csv(f'contact_{i}_ground_station_stats.csv')
        
        # Time window statistics
        pd.DataFrame({
            'No_KSAT': no_ksat_stat['window_stats'],
            'With_KSAT': with_ksat_stat['window_stats']
        }).to_csv(f'contact_{i}_time_windows.csv')

    return df_seq_no_ksat, df_seq_with_ksat