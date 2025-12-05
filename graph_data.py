# graph_data.py
import math
import pandas as pd
import numpy as np

# ============================================================
# 1. COORDINATES: PORTS (for coastal) + CAPITALS (for landlocked)
# ============================================================

# Main commercial PORT or sea gateway for each country
port_coords = {
    "Tunisia":        (36.81, 10.29),    # Tunis/Rades
    "United States":  (40.70, -74.00),   # New York / New Jersey
    "Germany":        (53.54, 9.99),     # Hamburg
    "China":          (31.23, 121.47),   # Shanghai
    "Japan":          (35.60, 139.78),   # Tokyo Bay
    "United Kingdom": (51.45, 0.00),     # Thames Estuary
    "France":         (43.30, 5.37),     # Marseille
    "Canada":         (45.50, -73.55),   # Montreal
    "South Korea":    (35.10, 129.04),   # Busan
    "Australia":      (-12.417105, 130.800923),  # (your original coords)
    "Italy":          (44.41, 8.93),     # Genoa
    "India":          (19.08, 72.88),    # Mumbai
    "Netherlands":    (51.95, 4.14),     # Rotterdam
    "Spain":          (36.13, -5.43),    # Algeciras
    "Brazil":         (-23.96, -46.33),  # Santos
    "Mexico":         (19.20, -96.10),   # Veracruz
    "Indonesia":      (-6.10, 106.80),   # Jakarta (Tanjung Priok)
    "Russia":         (59.93, 30.25),    # St Petersburg
    "Switzerland":    None,              # landlocked
    "Saudi Arabia":   (21.48, 39.18),    # Jeddah
    "Turkey":         (40.98, 28.95),    # Istanbul
    "Poland":         (54.36, 18.65),    # Gdansk
    "Sweden":         (57.70, 11.95),    # Gothenburg
    "South Africa":   (-29.88, 31.05),   # Durban
    "Argentina":      (-34.60, -58.37),  # Buenos Aires (port+capital)
    "UAE":            (25.00, 55.06),    # Jebel Ali
    "Thailand":       (13.10, 100.90),   # Laem Chabang
    "Chile":          (-33.05, -71.63),  # Valparaiso
    "Nigeria":        (6.45, 3.40),      # Lagos
    "New Zealand":    (-36.84, 174.77),  # Auckland
    "Egypt":          (31.26, 32.30),    # Port Said
}

# Capital coordinates (used especially when landlocked is involved)
capital_coords = {
    "Tunisia":        (36.81, 10.29),   # Tunis
    "United States":  (38.90, -77.04),  # Washington, DC
    "Germany":        (52.52, 13.41),   # Berlin
    "China":          (39.90, 116.41),  # Beijing
    "Japan":          (35.69, 139.69),  # Tokyo
    "United Kingdom": (51.51, -0.13),   # London
    "France":         (48.86, 2.35),    # Paris
    "Canada":         (45.42, -75.70),  # Ottawa
    "South Korea":    (37.57, 126.98),  # Seoul
    "Australia":      (-35.28, 149.13), # Canberra
    "Italy":          (41.90, 12.49),   # Rome
    "India":          (28.64, 77.23),   # New Delhi
    "Netherlands":    (52.37, 4.90),    # Amsterdam
    "Spain":          (40.42, -3.70),   # Madrid
    "Brazil":         (-15.79, -47.88), # Brasília
    "Mexico":         (19.43, -99.14),  # Mexico City
    "Indonesia":      (-6.19, 106.81),  # Jakarta
    "Russia":         (55.76, 37.62),   # Moscow
    "Switzerland":    (46.95, 7.45),    # Bern
    "Saudi Arabia":   (24.71, 46.68),   # Riyadh
    "Turkey":         (39.93, 32.86),   # Ankara
    "Poland":         (52.23, 21.01),   # Warsaw
    "Sweden":         (59.33, 18.07),   # Stockholm
    "South Africa":   (-25.75, 28.23),  # Pretoria
    "Argentina":      (-34.60, -58.38), # Buenos Aires
    "UAE":            (24.47, 54.38),   # Abu Dhabi
    "Thailand":       (13.76, 100.50),  # Bangkok
    "Chile":          (-33.45, -70.67), # Santiago
    "Nigeria":        (9.08, 7.40),     # Abuja
    "New Zealand":    (-41.29, 174.78), # Wellington
    "Egypt":          (30.04, 31.24),   # Cairo
}

# Landlocked countries in this set
landlocked_countries = {"Switzerland"}

countries = list(port_coords.keys())

# ============================================================
# 2. HAVERSINE + DISTANCE HELPERS
# ============================================================

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def coord_for_graph(country, mode="port_or_capital"):
    """
    For general distances (non-neighbor edges):
      - If country is landlocked -> use capital
      - Else -> use port
    """
    if country in landlocked_countries:
        return capital_coords[country]
    else:
        return port_coords[country]


def neighbor_distance(country1, country2):
    """
    Special rule for neighbor edges:
      - If at least one country is landlocked -> use CAPITALS for BOTH.
      - Else -> use PORTS for BOTH.
    """
    use_capitals = (country1 in landlocked_countries) or (country2 in landlocked_countries)
    if use_capitals:
        lat1, lon1 = capital_coords[country1]
        lat2, lon2 = capital_coords[country2]
    else:
        lat1, lon1 = port_coords[country1]
        lat2, lon2 = port_coords[country2]
    return haversine_km(lat1, lon1, lat2, lon2)

# ============================================================
# 3. MULTIPLIERS FOR REALISTIC SHIPPING DISTANCES
# ============================================================

MARITIME_MULTIPLIERS = {
    "short_sea": 1.2,
    "one_hop": 1.40,
    "intercontinental": 1.9,
    "oceanic": 2.4
}

LAND_MULTIPLIERS = {
    "border": 1.05,
    "ferry": 1.10,
    "landlocked_to_port": 1.10,
    "landlocked_to_landlocked": 1.20
}

# Simple continent mapping to detect Suez/Panama relevance
continent = {
    "Tunisia": "Africa",
    "Egypt": "Africa",
    "Nigeria": "Africa",
    "South Africa": "Africa",

    "France": "Europe",
    "Germany": "Europe",
    "Italy": "Europe",
    "Spain": "Europe",
    "United Kingdom": "Europe",
    "Netherlands": "Europe",
    "Poland": "Europe",
    "Switzerland": "Europe",
    "Russia": "Europe",
    "Turkey": "Europe",
    "Sweden": "Europe",

    "United States": "America",
    "Canada": "America",
    "Mexico": "America",
    "Brazil": "America",
    "Argentina": "America",
    "Chile": "America",

    "China": "Asia",
    "Japan": "Asia",
    "South Korea": "Asia",
    "India": "Asia",
    "Indonesia": "Asia",
    "Thailand": "Asia",

    "Australia": "Oceania",
    "New Zealand": "Oceania",

    "UAE": "Asia",
    "Saudi Arabia": "Asia"
}

# Major hub countries (by role in global shipping)
major_hubs = [
    "Italy",
    "Spain",
    "France",
    "Germany",
    "Egypt",
    "UAE",
    "China",
    "United States",
    "Brazil"
]

def needs_suez(c1, c2):
    # Europe <-> Asia or Africa <-> Asia
    return ((continent[c1] in ["Europe", "Africa"]) and continent[c2] == "Asia") or \
           ((continent[c2] in ["Europe", "Africa"]) and continent[c1] == "Asia")


def needs_panama(c1, c2):
    # Asia <-> America
    return (continent[c1] == "Asia" and continent[c2] == "America") or \
           (continent[c2] == "Asia" and continent[c1] == "America")


def realistic_distance(country1, country2, neighbor=False):
    """Return realistic distance (km) with maritime multipliers + hub penalties."""

    # ---- 1) NEIGHBOR ROUTES (land or short-sea)
    if neighbor:
        use_capitals = (country1 in landlocked_countries) or (country2 in landlocked_countries)

        if use_capitals:
            lat1, lon1 = capital_coords[country1]
            lat2, lon2 = capital_coords[country2]
            d = haversine_km(lat1, lon1, lat2, lon2)
            if (country1 in landlocked_countries or country2 in landlocked_countries):
                return d * LAND_MULTIPLIERS["landlocked_to_port"]
            else:
                return d * LAND_MULTIPLIERS["border"]
        else:
            lat1, lon1 = port_coords[country1]
            lat2, lon2 = port_coords[country2]
            d = haversine_km(lat1, lon1, lat2, lon2)
            return d * LAND_MULTIPLIERS["border"]

    # ---- 2) GENERAL MARITIME DISTANCE
    lat1, lon1 = coord_for_graph(country1)
    lat2, lon2 = coord_for_graph(country2)
    d = haversine_km(lat1, lon1, lat2, lon2)

    # Maritime multiplier by scale
    if d <= 2000:
        d *= MARITIME_MULTIPLIERS["short_sea"]
    elif d <= 8000:
        d *= MARITIME_MULTIPLIERS["one_hop"]
    elif d <= 15000:
        d *= MARITIME_MULTIPLIERS["intercontinental"]
    else:
        d *= MARITIME_MULTIPLIERS["oceanic"]

    # ---- 3) CANAL PENALTIES
    if needs_suez(country1, country2):
        d *= 1.10
    if needs_panama(country1, country2):
        d *= 1.12

    # ---- 4) HUB-TO-HUB PENALTY
    HUB_PENALTY_KM = 5000  # Equivalent to a huge shipping detour
    if (country1 in major_hubs) and (country2 in major_hubs):
        d += HUB_PENALTY_KM

    return d

# ============================================================
# 4. BUILD FULL DISTANCE MATRIX (PORT or CAPITAL for LANDLOCKED)
# ============================================================

dist_matrix = pd.DataFrame(index=countries, columns=countries, dtype=float)

for c1 in countries:
    lat1, lon1 = coord_for_graph(c1)
    for c2 in countries:
        if c1 == c2:
            dist = 0.0
        else:
            lat2, lon2 = coord_for_graph(c2)
            dist = haversine_km(lat1, lon1, lat2, lon2)
        dist_matrix.loc[c1, c2] = dist

# ============================================================
# 5. DEFINE REALISTIC REGIONAL NEIGHBORS
# ============================================================

regional_neighbors = {
    "Tunisia":        ["Italy", "Spain", "France", "Egypt"],
    "Italy":          ["France", "Switzerland", "Germany"],
    "Spain":          ["France"],
    "France":         ["Spain", "Italy", "Germany", "Switzerland", "United Kingdom"],
    "Germany":        ["France", "Poland", "Netherlands", "Switzerland"],
    "Poland":         ["Germany", "Russia", "Sweden"],
    "Sweden":         ["Poland"],  # short sea via Baltic
    "United Kingdom": ["France", "Netherlands"],
    "Netherlands":    ["Germany", "France", "United Kingdom"],
    "Switzerland":    ["France", "Germany", "Italy"],  # landlocked
    "Egypt":          ["Saudi Arabia"],
    "Saudi Arabia":   ["Egypt", "UAE"],
    "UAE":            ["Saudi Arabia", "India"],
    "Turkey":         ["Russia", "Egypt"],
    "Russia":         ["Poland", "Turkey", "Sweden"],
    "India":          ["UAE", "Thailand"],
    "Thailand":       ["India", "Indonesia"],
    "Indonesia":      ["Thailand", "Australia"],
    "South Korea":    ["China", "Japan"],
    "Japan":          ["China", "South Korea"],
    "Mexico":         ["United States"],
    "United States":  ["Canada", "Mexico"],
    "Canada":         ["United States"],
    "Brazil":         ["Argentina"],
    "Argentina":      ["Brazil", "Chile"],
    "Chile":          ["Argentina"],
    # South Africa / Nigeria connections are not direct here (reach via hubs)
}

# ============================================================
# 6. BUILD RESTRICTED DISTANCE MATRIX (HUB + NEIGHBOR LOGIC)
# ============================================================

dist_matrix_restricted = pd.DataFrame(
    index=countries,
    columns=countries,
    data=np.inf,
    dtype=float
)

# Diagonal = 0
for country in countries:
    dist_matrix_restricted.loc[country, country] = 0.0

connections_added = {c: [] for c in countries}

for country in countries:
    used_partners = set()

    # 1) Add neighbor connections first (real geography)
    if country in regional_neighbors:
        for neighbor in regional_neighbors[country]:
            if neighbor in countries and neighbor not in used_partners:
                d_neighbor = neighbor_distance(country, neighbor)
                dist_matrix_restricted.loc[country, neighbor] = d_neighbor
                dist_matrix_restricted.loc[neighbor, country] = d_neighbor
                connections_added[country].append(f"{neighbor} (neighbor)")
                used_partners.add(neighbor)

    # 2) Add hub connections
    if country in major_hubs:
        # hubs connect to closest 3 other hubs
        other_hubs = [h for h in major_hubs if h != country]
        hub_distances = sorted(other_hubs, key=lambda h: dist_matrix.loc[country, h])
        closest = hub_distances[:3]

        for hub in closest:
            if hub not in used_partners:
                d = realistic_distance(country, hub, neighbor=False)
                dist_matrix_restricted.loc[country, hub] = d
                dist_matrix_restricted.loc[hub, country] = d
                connections_added[country].append(f"{hub} (hub)")
                used_partners.add(hub)
    else:
        # non-hubs connect to 2–3 hubs
        hub_distances = sorted(major_hubs, key=lambda h: dist_matrix.loc[country, h])
        num = 3 if dist_matrix.loc[country, hub_distances[0]] > 1000 else 2
        closest = hub_distances[:num]

        for hub in closest:
            if hub not in used_partners:
                d = realistic_distance(country, hub, neighbor=False)
                dist_matrix_restricted.loc[country, hub] = d
                dist_matrix_restricted.loc[hub, country] = d
                connections_added[country].append(f"{hub} (hub)")
                used_partners.add(hub)

# 6.b BLOCK LANDLOCKED COUNTRIES FROM ACTING AS TRANSIT NODES
for ll in landlocked_countries:
    for c in countries:
        if c != ll:
            dist_matrix_restricted.loc[ll, c] = np.inf  # no outgoing from landlocked

# ============================================================
# 7. SHIPPING COST MODEL (COST = BASE + DIST * RATE) + NORMALIZATION
# ============================================================

LANES = {
    "short":  {"max": 2000,   "base": 250.0, "per_km": 0.18},
    "medium": {"max": 8000,   "base": 350.0, "per_km": 0.15},
    "long":   {"max": np.inf, "base": 450.0, "per_km": 0.13},
}

def classify_lane(dist_km):
    if dist_km <= LANES["short"]["max"]:
        return "short"
    elif dist_km <= LANES["medium"]["max"]:
        return "medium"
    else:
        return "long"


def compute_shipping_cost(dist_km):
    if not np.isfinite(dist_km) or dist_km <= 0:
        return np.inf
    lane = classify_lane(dist_km)
    params = LANES[lane]
    return params["base"] + params["per_km"] * dist_km


cost_matrix = pd.DataFrame(index=countries, columns=countries, dtype=float)

for c1 in countries:
    for c2 in countries:
        d = dist_matrix_restricted.loc[c1, c2]
        if np.isfinite(d) and d > 0:
            cost = compute_shipping_cost(d)
        elif d == 0:
            cost = 0.0
        else:
            cost = np.inf
        cost_matrix.loc[c1, c2] = cost

# Normalize costs (0–1) for edges that exist (finite, > 0)
finite_mask = np.isfinite(cost_matrix.values) & (cost_matrix.values > 0)
finite_costs = cost_matrix.values[finite_mask]

cost_min = finite_costs.min()
cost_max = finite_costs.max()

norm_cost_matrix = cost_matrix.copy()

for i in range(len(countries)):
    for j in range(len(countries)):
        val = cost_matrix.iat[i, j]
        if np.isfinite(val) and val > 0:
            norm_val = (val - cost_min) / (cost_max - cost_min)
            norm_cost_matrix.iat[i, j] = norm_val
        elif val == 0:
            norm_cost_matrix.iat[i, j] = 0.0
        else:
            norm_cost_matrix.iat[i, j] = np.inf

# ============================================================
# 8. NODE COORDINATES USED BY THE WEB APP
# ============================================================

country_coords = {}
for c in countries:
    if c in landlocked_countries:
        country_coords[c] = capital_coords[c]
    else:
        country_coords[c] = port_coords[c]

# ============================================================
# 9. PUBLIC HELPERS FOR THE WEB APP
# ============================================================

def get_countries():
    """
    Return list of dicts: [{'id': 'Tunisia', 'name': 'Tunisia', 'lat': ..., 'lon': ...}, ...]
    """
    data = []
    for c in countries:
        lat, lon = country_coords[c]
        data.append({
            "id": c,
            "name": c,
            "lat": float(lat),
            "lon": float(lon),
        })
    return data


def get_adjacency():
    """
    Build adjacency list from normalized cost matrix:
      adjacency[c1] = [
          {'to': c2, 'cost': normalized_cost, 'distanceKm': dist_km},
          ...
      ]
    Only include edges where a finite, nonzero cost exists.
    """
    adjacency = {c: [] for c in countries}
    for c1 in countries:
        for c2 in countries:
            if c1 == c2:
                continue
            cost = norm_cost_matrix.loc[c1, c2]
            if np.isfinite(cost) and cost > 0:
                distance_km = dist_matrix_restricted.loc[c1, c2]
                adjacency[c1].append({
                    "to": c2,
                    "cost": float(cost),
                    "distanceKm": float(distance_km),
                })
    return adjacency


# Optional: quick sanity check if you run this file directly
if __name__ == "__main__":
    print("Countries:", len(countries))
    print("Example country list entry:", get_countries()[0])
    adj = get_adjacency()
    print("Outgoing edges from Tunisia:", len(adj["Tunisia"]))
