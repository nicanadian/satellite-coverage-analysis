# Ground Station Database
# =======================
# 36 stations total: 14 KSAT, 11 ATLAS, 11 ViaSat RTE
#
# Comment/uncomment stations to include/exclude from analysis.
# Each station is one dict on a single line for easy toggling.
#
# Fields: name, lat, lon, s_band, x_band, ka_band, provider
#
# Usage in config YAML:
#   ground_stations_file: "configs/ground_stations.py"
#
# Or import directly:
#   from configs.ground_stations import GROUND_STATIONS

GROUND_STATIONS = [
    # =========================================================================
    # KSAT (Kongsberg Satellite Services) - 14 stations
    # =========================================================================
    # Polar
    {"name": "KSAT-Svalbard-SvalSat",     "lat":  78.23, "lon":   15.41, "s_band": True,  "x_band": True,  "ka_band": True,  "provider": "KSAT"},
    {"name": "KSAT-Troll-Antarctica",     "lat": -72.01, "lon":    2.53, "s_band": True,  "x_band": True,  "ka_band": False, "provider": "KSAT"},
    # North America
    {"name": "KSAT-Prudhoe-Bay-AK",       "lat":  70.20, "lon": -148.47, "s_band": True,  "x_band": True,  "ka_band": False, "provider": "KSAT"},
    {"name": "KSAT-Inuvik-Canada",        "lat":  68.36, "lon": -133.72, "s_band": True,  "x_band": True,  "ka_band": False, "provider": "KSAT"},
    # South America
    {"name": "KSAT-Punta-Arenas-CL",      "lat": -52.94, "lon":  -70.87, "s_band": True,  "x_band": True,  "ka_band": True,  "provider": "KSAT"},
    # Europe
    {"name": "KSAT-Tromso-Norway",        "lat":  69.65, "lon":   18.96, "s_band": True,  "x_band": True,  "ka_band": False, "provider": "KSAT"},
    {"name": "KSAT-Athens-Greece",        "lat":  38.05, "lon":   23.86, "s_band": True,  "x_band": True,  "ka_band": False, "provider": "KSAT"},
    {"name": "KSAT-Puertollano-Spain",    "lat":  38.68, "lon":   -4.11, "s_band": True,  "x_band": True,  "ka_band": True,  "provider": "KSAT"},
    # Asia-Pacific
    {"name": "KSAT-Singapore",            "lat":   1.35, "lon":  103.82, "s_band": True,  "x_band": True,  "ka_band": False, "provider": "KSAT"},
    {"name": "KSAT-Jeju-South-Korea",     "lat":  33.39, "lon":  126.31, "s_band": True,  "x_band": True,  "ka_band": False, "provider": "KSAT"},
    {"name": "KSAT-Mingenew-Australia",   "lat": -29.00, "lon":  115.30, "s_band": True,  "x_band": True,  "ka_band": False, "provider": "KSAT"},
    {"name": "KSAT-Awarua-NZ",            "lat": -46.53, "lon":  168.37, "s_band": True,  "x_band": True,  "ka_band": True,  "provider": "KSAT"},
    # Africa
    {"name": "KSAT-Mauritius",            "lat": -20.17, "lon":   57.50, "s_band": True,  "x_band": True,  "ka_band": False, "provider": "KSAT"},
    {"name": "KSAT-Hartebeesthoek-ZA",    "lat": -25.89, "lon":   27.69, "s_band": True,  "x_band": True,  "ka_band": False, "provider": "KSAT"},

    # =========================================================================
    # ATLAS Space Operations - 11 stations (unique, not shared with ViaSat RTE)
    # =========================================================================
    # North America
    {"name": "ATLAS-Barrow-AK",           "lat":  71.29, "lon": -156.79, "s_band": True,  "x_band": True,  "ka_band": False, "provider": "ATLAS"},
    {"name": "ATLAS-North-Pole-AK",       "lat":  64.75, "lon": -147.35, "s_band": True,  "x_band": True,  "ka_band": True,  "provider": "ATLAS"},
    {"name": "ATLAS-Pendergrass-GA",      "lat":  34.24, "lon":  -83.63, "s_band": True,  "x_band": True,  "ka_band": False, "provider": "ATLAS"},
    {"name": "ATLAS-Paumalu-HI",          "lat":  21.67, "lon": -158.02, "s_band": True,  "x_band": True,  "ka_band": False, "provider": "ATLAS"},
    {"name": "ATLAS-Harmon-Guam",         "lat":  13.48, "lon":  144.80, "s_band": True,  "x_band": True,  "ka_band": False, "provider": "ATLAS"},
    # Europe
    {"name": "ATLAS-Muonio-Finland",      "lat":  67.94, "lon":   23.68, "s_band": True,  "x_band": True,  "ka_band": False, "provider": "ATLAS"},
    {"name": "ATLAS-Dundee-Scotland",     "lat":  56.46, "lon":   -2.97, "s_band": True,  "x_band": True,  "ka_band": False, "provider": "ATLAS"},
    # Asia-Pacific
    {"name": "ATLAS-Chitose-Japan",       "lat":  42.82, "lon":  141.65, "s_band": True,  "x_band": True,  "ka_band": False, "provider": "ATLAS"},
    {"name": "ATLAS-Mingenew-AU",         "lat": -29.01, "lon":  115.34, "s_band": True,  "x_band": True,  "ka_band": True,  "provider": "ATLAS"},
    {"name": "ATLAS-Awarua-NZ",           "lat": -46.53, "lon":  168.37, "s_band": True,  "x_band": True,  "ka_band": False, "provider": "ATLAS"},
    # Africa
    {"name": "ATLAS-Mwulire-Rwanda",      "lat":  -1.94, "lon":   30.06, "s_band": True,  "x_band": True,  "ka_band": False, "provider": "ATLAS"},

    # =========================================================================
    # ViaSat RTE (Real-Time Earth) - 11 stations
    # =========================================================================
    # Africa
    {"name": "ViaSat-Accra-Ghana",        "lat":   5.56, "lon":   -0.19, "s_band": True,  "x_band": True,  "ka_band": True,  "provider": "ViaSat"},
    {"name": "ViaSat-Pretoria-ZA",        "lat": -25.74, "lon":   28.25, "s_band": True,  "x_band": True,  "ka_band": True,  "provider": "ViaSat"},
    # Asia-Pacific
    {"name": "ViaSat-Alice-Springs-AU",   "lat": -23.70, "lon":  133.80, "s_band": True,  "x_band": True,  "ka_band": True,  "provider": "ViaSat"},
    {"name": "ViaSat-Obihiro-Japan",      "lat":  42.92, "lon":  143.20, "s_band": True,  "x_band": True,  "ka_band": True,  "provider": "ViaSat"},
    {"name": "ViaSat-Dubai-UAE",          "lat":  25.08, "lon":   55.31, "s_band": True,  "x_band": True,  "ka_band": False, "provider": "ViaSat"},
    # North America
    {"name": "ViaSat-Fairbanks-AK",       "lat":  64.84, "lon": -147.72, "s_band": True,  "x_band": True,  "ka_band": True,  "provider": "ViaSat"},
    {"name": "ViaSat-Pendergrass-GA",     "lat":  34.12, "lon":  -83.02, "s_band": True,  "x_band": True,  "ka_band": False, "provider": "ViaSat"},
    # Europe
    {"name": "ViaSat-Guildford-UK",       "lat":  51.24, "lon":   -0.57, "s_band": True,  "x_band": True,  "ka_band": False, "provider": "ViaSat"},
    {"name": "ViaSat-Ojebyn-Sweden",      "lat":  65.29, "lon":   21.07, "s_band": True,  "x_band": True,  "ka_band": True,  "provider": "ViaSat"},
    # South America
    {"name": "ViaSat-Cordoba-Argentina",  "lat": -31.42, "lon":  -64.18, "s_band": True,  "x_band": True,  "ka_band": False, "provider": "ViaSat"},
    {"name": "ViaSat-Ushuaia-Argentina",  "lat": -54.81, "lon":  -68.31, "s_band": True,  "x_band": True,  "ka_band": True,  "provider": "ViaSat"},
]


# =============================================================================
# Helper functions for filtering
# =============================================================================

def get_stations(providers=None, ka_band_only=False, regions=None):
    """
    Filter ground stations by criteria.

    Args:
        providers: List of provider names to include, e.g. ["KSAT", "ATLAS"]
        ka_band_only: If True, only return stations with Ka-band capability
        regions: Not implemented yet, but could filter by lat/lon bounds

    Returns:
        List of station dicts matching criteria
    """
    stations = GROUND_STATIONS

    if providers:
        stations = [s for s in stations if s["provider"] in providers]

    if ka_band_only:
        stations = [s for s in stations if s["ka_band"]]

    return stations


def stations_to_yaml_list(stations):
    """Convert station list to YAML-compatible list of dicts."""
    return [
        {
            "name": s["name"],
            "lat": s["lat"],
            "lon": s["lon"],
            "s_band": s["s_band"],
            "x_band": s["x_band"],
            "ka_band": s["ka_band"],
            "provider": s["provider"],
        }
        for s in stations
    ]


# Quick stats when run directly
if __name__ == "__main__":
    from collections import Counter

    providers = Counter(s["provider"] for s in GROUND_STATIONS)
    ka_capable = sum(1 for s in GROUND_STATIONS if s["ka_band"])

    print(f"Total stations: {len(GROUND_STATIONS)}")
    print(f"By provider: {dict(providers)}")
    print(f"Ka-band capable: {ka_capable}")
