"""
POI (Points of Interest) Spatial Enrichment Module
Calculates geodesic distances between property coordinates and critical local amenities
that drive property valuation in Melbourne's Eastern Suburbs (Burwood East & surrounds).
"""

import math

# Coordinates of key local educational institutions
SCHOOLS = {
    "mwsc": {
        "name": "Mount Waverley Secondary College",
        "lat": -37.8687,
        "lon": 145.1303,
        "catchment_radius_km": 2.2
    },
    "gwsc": {
        "name": "Glen Waverley Secondary College",
        "lat": -37.8798,
        "lon": 145.1633,
        "catchment_radius_km": 2.5
    },
    "box_hill_high": {
        "name": "Box Hill High School",
        "lat": -37.8222,
        "lon": 145.1287,
        "catchment_radius_km": 2.0
    },
    "deakin_uni": {
        "name": "Deakin University (Burwood Campus)",
        "lat": -37.8475,
        "lon": 145.1150
    },
    "burwood_east_primary": {
        "name": "Burwood East Primary School",
        "lat": -37.8545,
        "lon": 145.1485
    }
}

# Major shopping centers & retail hubs
SHOPPING_CENTRES = {
    "burwood_one": {
        "name": "Burwood One Shopping Centre (24h Kmart & Coles)",
        "lat": -37.8543,
        "lon": 145.1607
    },
    "burwood_brickworks": {
        "name": "Burwood Brickworks Shopping Centre",
        "lat": -37.8504,
        "lon": 145.1374
    },
    "the_glen": {
        "name": "The Glen Shopping Centre",
        "lat": -37.8765,
        "lon": 145.1648
    },
    "box_hill_central": {
        "name": "Box Hill Central Shopping Hub",
        "lat": -37.8193,
        "lon": 145.1228
    },
    "forest_hill_chase": {
        "name": "Forest Hill Chase Shopping Centre",
        "lat": -37.8378,
        "lon": 145.1678
    }
}

# Railway stations serving the area
TRAIN_STATIONS = [
    {"name": "Glen Waverley Station", "lat": -37.8795, "lon": 145.1610},
    {"name": "Mount Waverley Station", "lat": -37.8770, "lon": 145.1310},
    {"name": "Box Hill Station", "lat": -37.8193, "lon": 145.1215},
    {"name": "Blackburn Station", "lat": -37.8199, "lon": 145.1511},
    {"name": "Jordanville Station", "lat": -37.8720, "lon": 145.1180},
    {"name": "Syndal Station", "lat": -37.8748, "lon": 145.1485}
]

# Route 75 Tram line (runs along Burwood Highway through Burwood, Burwood East, and terminates at Vermont South)
TRAM_75_CORRIDOR = {
    "burwood_hwy_lat_approx": -37.8525,  # Latitude axis of Burwood Hwy
    "min_lon": 145.1000,
    "max_lon": 145.1850
}

# Melbourne CBD Reference (GPO)
MELBOURNE_CBD = {
    "name": "Melbourne CBD (GPO)",
    "lat": -37.8136,
    "lon": 144.9631
}

# Suburb centroid approximations for fallback coordinates if exact lat/lon is unparsed
SUBURB_CENTROIDS = {
    "burwood east": (-37.8541, 145.1500),
    "burwood": (-37.8480, 145.1090),
    "mount waverley": (-37.8760, 145.1320),
    "glen waverley": (-37.8810, 145.1660),
    "box hill": (-37.8190, 145.1240),
    "box hill south": (-37.8390, 145.1270),
    "blackburn": (-37.8180, 145.1530),
    "blackburn south": (-37.8380, 145.1510),
    "forest hill": (-37.8390, 145.1700),
    "vermont": (-37.8370, 145.1970),
    "vermont south": (-37.8630, 145.1850)
}


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Computes great-circle distance between two points on Earth in kilometers
    using the Haversine formula.
    """
    R = 6371.0  # Earth's mean radius in km
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return round(R * c, 3)


def get_distance_to_tram_75(lat: float, lon: float) -> float:
    """
    Estimates perpendicular distance (km) to the Route 75 Tram line along Burwood Highway.
    """
    # Route 75 operates along Burwood Highway between approx 145.100 and 145.185 longitude
    clamped_lon = max(TRAM_75_CORRIDOR["min_lon"], min(lon, TRAM_75_CORRIDOR["max_lon"]))
    return haversine_distance(lat, lon, TRAM_75_CORRIDOR["burwood_hwy_lat_approx"], clamped_lon)


def enrich_property_with_pois(lat: float, lon: float, suburb: str = "") -> dict:
    """
    Enriches a property record with spatial proximity features to key amenities.
    """
    # Fallback to suburb centroid if coordinates are missing or invalid
    if lat is None or lon is None or math.isnan(lat) or math.isnan(lon) or lat == 0:
        sub_key = suburb.lower().strip()
        if sub_key in SUBURB_CENTROIDS:
            lat, lon = SUBURB_CENTROIDS[sub_key]
        else:
            lat, lon = SUBURB_CENTROIDS["burwood east"]

    # 1. School Distances & Catchment Indicators
    dist_mwsc = haversine_distance(lat, lon, SCHOOLS["mwsc"]["lat"], SCHOOLS["mwsc"]["lon"])
    in_mwsc_zone = 1 if dist_mwsc <= SCHOOLS["mwsc"]["catchment_radius_km"] else 0

    dist_gwsc = haversine_distance(lat, lon, SCHOOLS["gwsc"]["lat"], SCHOOLS["gwsc"]["lon"])
    in_gwsc_zone = 1 if dist_gwsc <= SCHOOLS["gwsc"]["catchment_radius_km"] else 0

    dist_box_hill_high = haversine_distance(lat, lon, SCHOOLS["box_hill_high"]["lat"], SCHOOLS["box_hill_high"]["lon"])
    in_box_hill_high_zone = 1 if dist_box_hill_high <= SCHOOLS["box_hill_high"]["catchment_radius_km"] else 0

    dist_deakin = haversine_distance(lat, lon, SCHOOLS["deakin_uni"]["lat"], SCHOOLS["deakin_uni"]["lon"])

    # 2. Retail & Supermarket Hubs
    dist_burwood_one = haversine_distance(lat, lon, SHOPPING_CENTRES["burwood_one"]["lat"], SHOPPING_CENTRES["burwood_one"]["lon"])
    dist_burwood_brickworks = haversine_distance(lat, lon, SHOPPING_CENTRES["burwood_brickworks"]["lat"], SHOPPING_CENTRES["burwood_brickworks"]["lon"])
    dist_the_glen = haversine_distance(lat, lon, SHOPPING_CENTRES["the_glen"]["lat"], SHOPPING_CENTRES["the_glen"]["lon"])
    dist_box_hill_central = haversine_distance(lat, lon, SHOPPING_CENTRES["box_hill_central"]["lat"], SHOPPING_CENTRES["box_hill_central"]["lon"])
    dist_forest_hill_chase = haversine_distance(lat, lon, SHOPPING_CENTRES["forest_hill_chase"]["lat"], SHOPPING_CENTRES["forest_hill_chase"]["lon"])

    # Nearest major shopping center distance
    dist_nearest_major_shop = min(dist_burwood_one, dist_burwood_brickworks, dist_the_glen, dist_box_hill_central, dist_forest_hill_chase)

    # 3. Public Transport Distances
    train_dists = [haversine_distance(lat, lon, st["lat"], st["lon"]) for st in TRAIN_STATIONS]
    dist_nearest_train = min(train_dists)
    dist_tram_75 = get_distance_to_tram_75(lat, lon)

    # 4. CBD Distance
    dist_cbd = haversine_distance(lat, lon, MELBOURNE_CBD["lat"], MELBOURNE_CBD["lon"])

    return {
        "latitude": round(lat, 5),
        "longitude": round(lon, 5),
        "dist_to_mwsc_km": dist_mwsc,
        "in_mwsc_zone": in_mwsc_zone,
        "dist_to_gwsc_km": dist_gwsc,
        "in_gwsc_zone": in_gwsc_zone,
        "dist_to_box_hill_high_km": dist_box_hill_high,
        "in_box_hill_high_zone": in_box_hill_high_zone,
        "dist_to_deakin_km": dist_deakin,
        "dist_to_burwood_one_km": dist_burwood_one,
        "dist_to_burwood_brickworks_km": dist_burwood_brickworks,
        "dist_to_the_glen_km": dist_the_glen,
        "dist_to_box_hill_central_km": dist_box_hill_central,
        "dist_to_nearest_major_shop_km": dist_nearest_major_shop,
        "dist_to_nearest_train_km": dist_nearest_train,
        "dist_to_tram_75_km": dist_tram_75,
        "dist_to_cbd_km": dist_cbd
    }
