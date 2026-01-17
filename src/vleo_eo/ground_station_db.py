"""
Ground Station Database for RGT-Min Optimization

Contains location data for commercial ground station providers:
- ATLAS Space Operations
- Viasat Real-Time Earth (RTE)
- KSAT (Kongsberg Satellite Services)
- AWS Ground Station

Data compiled from publicly available sources as of 2026.
"""

from typing import List, Dict, Any

# Ground station entries: name, lat, lon, provider, bands (s_band, x_band, ka_band)
# Band capabilities based on publicly available information

GROUND_STATION_DATABASE: List[Dict[str, Any]] = [
    # ==========================================================================
    # ATLAS Space Operations
    # Federated network with 50+ antennas across 34+ locations globally
    # ==========================================================================
    # North America
    {"name": "ATLAS-Paumalu-HI", "lat": 21.67, "lon": -158.02, "provider": "ATLAS",
     "s_band": True, "x_band": True, "ka_band": True, "min_elevation_deg": 5.0},
    {"name": "ATLAS-Brewster-WA", "lat": 48.14, "lon": -119.68, "provider": "ATLAS",
     "s_band": True, "x_band": True, "ka_band": True, "min_elevation_deg": 5.0},
    {"name": "ATLAS-Austin-TX", "lat": 30.27, "lon": -97.74, "provider": "ATLAS",
     "s_band": True, "x_band": True, "ka_band": False, "min_elevation_deg": 5.0},
    {"name": "ATLAS-Miami-FL", "lat": 25.76, "lon": -80.19, "provider": "ATLAS",
     "s_band": True, "x_band": True, "ka_band": False, "min_elevation_deg": 5.0},
    {"name": "ATLAS-Fairbanks-AK", "lat": 64.84, "lon": -147.72, "provider": "ATLAS",
     "s_band": True, "x_band": True, "ka_band": True, "min_elevation_deg": 5.0},
    # Europe
    {"name": "ATLAS-Guildford-UK", "lat": 51.24, "lon": -0.57, "provider": "ATLAS",
     "s_band": True, "x_band": True, "ka_band": False, "min_elevation_deg": 5.0},
    {"name": "ATLAS-Athens-GR", "lat": 37.98, "lon": 23.73, "provider": "ATLAS",
     "s_band": True, "x_band": True, "ka_band": False, "min_elevation_deg": 5.0},
    # Asia-Pacific
    {"name": "ATLAS-Tokyo-JP", "lat": 35.68, "lon": 139.69, "provider": "ATLAS",
     "s_band": True, "x_band": True, "ka_band": True, "min_elevation_deg": 5.0},
    {"name": "ATLAS-Singapore-SG", "lat": 1.35, "lon": 103.82, "provider": "ATLAS",
     "s_band": True, "x_band": True, "ka_band": False, "min_elevation_deg": 5.0},
    {"name": "ATLAS-Perth-AU", "lat": -31.95, "lon": 115.86, "provider": "ATLAS",
     "s_band": True, "x_band": True, "ka_band": True, "min_elevation_deg": 5.0},
    # South America
    {"name": "ATLAS-Santiago-CL", "lat": -33.45, "lon": -70.67, "provider": "ATLAS",
     "s_band": True, "x_band": True, "ka_band": False, "min_elevation_deg": 5.0},
    # Africa
    {"name": "ATLAS-Johannesburg-ZA", "lat": -26.20, "lon": 28.05, "provider": "ATLAS",
     "s_band": True, "x_band": True, "ka_band": False, "min_elevation_deg": 5.0},

    # ==========================================================================
    # AWS Ground Station
    # 12 locations as of 2026
    # ==========================================================================
    {"name": "AWS-US-East-Ohio", "lat": 40.42, "lon": -82.91, "provider": "AWS",
     "s_band": True, "x_band": True, "ka_band": False, "min_elevation_deg": 5.0},
    {"name": "AWS-US-East-Virginia", "lat": 38.95, "lon": -77.45, "provider": "AWS",
     "s_band": True, "x_band": True, "ka_band": False, "min_elevation_deg": 5.0},
    {"name": "AWS-US-West-Oregon", "lat": 45.84, "lon": -119.29, "provider": "AWS",
     "s_band": True, "x_band": True, "ka_band": False, "min_elevation_deg": 5.0},
    {"name": "AWS-US-West-Alaska", "lat": 64.84, "lon": -147.72, "provider": "AWS",
     "s_band": True, "x_band": True, "ka_band": True, "min_elevation_deg": 5.0},
    {"name": "AWS-US-West-Hawaii", "lat": 21.31, "lon": -157.86, "provider": "AWS",
     "s_band": True, "x_band": True, "ka_band": False, "min_elevation_deg": 5.0},
    {"name": "AWS-Europe-Ireland", "lat": 53.43, "lon": -7.94, "provider": "AWS",
     "s_band": True, "x_band": True, "ka_band": False, "min_elevation_deg": 5.0},
    {"name": "AWS-Europe-Stockholm", "lat": 59.33, "lon": 18.07, "provider": "AWS",
     "s_band": True, "x_band": True, "ka_band": False, "min_elevation_deg": 5.0},
    {"name": "AWS-Europe-Frankfurt", "lat": 50.11, "lon": 8.68, "provider": "AWS",
     "s_band": True, "x_band": True, "ka_band": False, "min_elevation_deg": 5.0},
    {"name": "AWS-Asia-Singapore", "lat": 1.35, "lon": 103.82, "provider": "AWS",
     "s_band": True, "x_band": True, "ka_band": False, "min_elevation_deg": 5.0},
    {"name": "AWS-Asia-Seoul", "lat": 37.57, "lon": 126.98, "provider": "AWS",
     "s_band": True, "x_band": True, "ka_band": False, "min_elevation_deg": 5.0},
    {"name": "AWS-Asia-Sydney", "lat": -33.87, "lon": 151.21, "provider": "AWS",
     "s_band": True, "x_band": True, "ka_band": False, "min_elevation_deg": 5.0},
    {"name": "AWS-Africa-CapeTown", "lat": -33.93, "lon": 18.42, "provider": "AWS",
     "s_band": True, "x_band": True, "ka_band": False, "min_elevation_deg": 5.0},

    # ==========================================================================
    # KSAT (Kongsberg Satellite Services)
    # 28+ locations with 300+ antennas, including polar stations
    # ==========================================================================
    # Polar Stations (critical for LEO coverage)
    {"name": "KSAT-Svalbard-SvalSat", "lat": 78.23, "lon": 15.41, "provider": "KSAT",
     "s_band": True, "x_band": True, "ka_band": True, "min_elevation_deg": 3.0},
    {"name": "KSAT-Troll-Antarctica", "lat": -72.01, "lon": 2.53, "provider": "KSAT",
     "s_band": True, "x_band": True, "ka_band": True, "min_elevation_deg": 3.0},
    {"name": "KSAT-Inuvik-Canada", "lat": 68.36, "lon": -133.72, "provider": "KSAT",
     "s_band": True, "x_band": True, "ka_band": False, "min_elevation_deg": 5.0},
    {"name": "KSAT-Puertollano-Spain", "lat": 38.68, "lon": -4.11, "provider": "KSAT",
     "s_band": True, "x_band": True, "ka_band": True, "min_elevation_deg": 5.0},
    # North America
    {"name": "KSAT-Fairbanks-AK", "lat": 64.98, "lon": -147.51, "provider": "KSAT",
     "s_band": True, "x_band": True, "ka_band": True, "min_elevation_deg": 5.0},
    {"name": "KSAT-Miami-FL", "lat": 25.79, "lon": -80.29, "provider": "KSAT",
     "s_band": True, "x_band": True, "ka_band": False, "min_elevation_deg": 5.0},
    {"name": "KSAT-Punta-Arenas-CL", "lat": -53.16, "lon": -70.91, "provider": "KSAT",
     "s_band": True, "x_band": True, "ka_band": True, "min_elevation_deg": 5.0},
    # Europe
    {"name": "KSAT-Tromso-Norway", "lat": 69.65, "lon": 18.96, "provider": "KSAT",
     "s_band": True, "x_band": True, "ka_band": True, "min_elevation_deg": 5.0},
    {"name": "KSAT-Athens-Greece", "lat": 38.05, "lon": 23.86, "provider": "KSAT",
     "s_band": True, "x_band": True, "ka_band": False, "min_elevation_deg": 5.0},
    {"name": "KSAT-Grimstad-Norway", "lat": 58.34, "lon": 8.59, "provider": "KSAT",
     "s_band": True, "x_band": True, "ka_band": False, "min_elevation_deg": 5.0},
    # Asia-Pacific
    {"name": "KSAT-Singapore", "lat": 1.35, "lon": 103.82, "provider": "KSAT",
     "s_band": True, "x_band": True, "ka_band": False, "min_elevation_deg": 5.0},
    {"name": "KSAT-Dubai-UAE", "lat": 25.08, "lon": 55.31, "provider": "KSAT",
     "s_band": True, "x_band": True, "ka_band": False, "min_elevation_deg": 5.0},
    {"name": "KSAT-Mauritius", "lat": -20.17, "lon": 57.50, "provider": "KSAT",
     "s_band": True, "x_band": True, "ka_band": False, "min_elevation_deg": 5.0},
    {"name": "KSAT-Awarua-NZ", "lat": -46.53, "lon": 168.37, "provider": "KSAT",
     "s_band": True, "x_band": True, "ka_band": False, "min_elevation_deg": 5.0},
    {"name": "KSAT-Hartebeesthoek-ZA", "lat": -25.89, "lon": 27.69, "provider": "KSAT",
     "s_band": True, "x_band": True, "ka_band": False, "min_elevation_deg": 5.0},

    # ==========================================================================
    # Viasat Real-Time Earth (RTE)
    # Global network focused on real-time data delivery
    # ==========================================================================
    # North America
    {"name": "Viasat-Brewster-WA", "lat": 48.14, "lon": -119.68, "provider": "Viasat",
     "s_band": True, "x_band": True, "ka_band": True, "min_elevation_deg": 5.0},
    {"name": "Viasat-Dulles-VA", "lat": 38.95, "lon": -77.45, "provider": "Viasat",
     "s_band": True, "x_band": True, "ka_band": True, "min_elevation_deg": 5.0},
    {"name": "Viasat-Alaska", "lat": 64.97, "lon": -147.52, "provider": "Viasat",
     "s_band": True, "x_band": True, "ka_band": True, "min_elevation_deg": 5.0},
    # Europe
    {"name": "Viasat-Kiruna-Sweden", "lat": 67.86, "lon": 20.23, "provider": "Viasat",
     "s_band": True, "x_band": True, "ka_band": True, "min_elevation_deg": 5.0},
    {"name": "Viasat-Farnborough-UK", "lat": 51.28, "lon": -0.78, "provider": "Viasat",
     "s_band": True, "x_band": True, "ka_band": True, "min_elevation_deg": 5.0},
    # Asia-Pacific
    {"name": "Viasat-Tokyo-Japan", "lat": 35.68, "lon": 139.69, "provider": "Viasat",
     "s_band": True, "x_band": True, "ka_band": True, "min_elevation_deg": 5.0},
    {"name": "Viasat-Alice-Springs-AU", "lat": -23.70, "lon": 133.88, "provider": "Viasat",
     "s_band": True, "x_band": True, "ka_band": True, "min_elevation_deg": 5.0},
    {"name": "Viasat-Singapore", "lat": 1.35, "lon": 103.82, "provider": "Viasat",
     "s_band": True, "x_band": True, "ka_band": False, "min_elevation_deg": 5.0},
    # Africa
    {"name": "Viasat-Gaborone-Botswana", "lat": -24.66, "lon": 25.91, "provider": "Viasat",
     "s_band": True, "x_band": True, "ka_band": False, "min_elevation_deg": 5.0},
    # South America
    {"name": "Viasat-Santiago-Chile", "lat": -33.45, "lon": -70.67, "provider": "Viasat",
     "s_band": True, "x_band": True, "ka_band": True, "min_elevation_deg": 5.0},
]


def get_stations_by_provider(provider: str) -> List[Dict[str, Any]]:
    """Get all stations for a specific provider."""
    return [s for s in GROUND_STATION_DATABASE if s["provider"] == provider]


def get_ttc_capable_stations(provider: str = None) -> List[Dict[str, Any]]:
    """Get stations with TT&C capability (S-band or X-band)."""
    stations = GROUND_STATION_DATABASE
    if provider:
        stations = [s for s in stations if s["provider"] == provider]
    return [s for s in stations if s["s_band"] or s["x_band"]]


def get_ka_capable_stations(provider: str = None) -> List[Dict[str, Any]]:
    """Get stations with Ka-band capability for payload downlink."""
    stations = GROUND_STATION_DATABASE
    if provider:
        stations = [s for s in stations if s["provider"] == provider]
    return [s for s in stations if s["ka_band"]]


def get_all_providers() -> List[str]:
    """Get list of all unique providers."""
    return list(set(s["provider"] for s in GROUND_STATION_DATABASE))


def station_to_config(station: Dict[str, Any], downlink_rate_mbps: float = 1200.0) -> Dict[str, Any]:
    """Convert a station entry to the config format used by the analysis."""
    return {
        "name": station["name"],
        "lat": station["lat"],
        "lon": station["lon"],
        "min_elevation_deg": station["min_elevation_deg"],
        "s_band": station["s_band"],
        "x_band": station["x_band"],
        "ka_band": station["ka_band"],
        "provider": station["provider"],
        "downlink_rate_mbps": downlink_rate_mbps,
    }
