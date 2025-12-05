import math
import pandas as pd
import numpy as np
import heapq

# ============================================================
# 1. COORDINATES: PORTS (for coastal) + CAPITALS (for landlocked)
# ============================================================

# Main commercial PORT or sea gateway for each country
# (approximate but realistic enough for distance & cost modeling)
port_coords = {
    "Tunisia":        (36.81, 10.29),    # Tunis/Rades (keep your original)
    "United States":  (40.70, -74.00),   # New York / New Jersey
    "Germany":        (53.54, 9.99),     # Hamburg
    "China":          (31.23, 121.47),   # Shanghai
    "Japan":          (35.60, 139.78),   # Tokyo Bay
    "United Kingdom": (51.45, 0.00),     # Thames Estuary (London area)
    "France":         (43.30, 5.37),     # Marseille
    "Canada":         (45.50, -73.55),   # Montreal (gateway for Atlantic)
    "South Korea":    (35.10, 129.04),   # Busan
    "Australia":      (-12.417105, 130.800923),  # Sydney
    "Italy":          (44.41, 8.93),     # Genoa
    "India":          (19.08, 72.88),    # Mumbai
    "Netherlands":    (51.95, 4.14),     # Rotterdam
    "Spain":          (36.13, -5.43),    # Algeciras
    "Brazil":         (-23.96, -46.33),  # Santos
    "Mexico":         (19.20, -96.10),   # Veracruz
    "Indonesia":      (-6.10, 106.80),   # Jakarta (Tanjung Priok)
    "Russia":         (59.93, 30.25),    # St Petersburg (one main sea access)
    "Switzerland":    None,              # landlocked, no port
    "Saudi Arabia":   (21.48, 39.18),    # Jeddah
    "Turkey":         (40.98, 28.95),    # Istanbul
    "Poland":         (54.36, 18.65),    # Gdansk
    "Sweden":         (57.70, 11.95),    # Gothenburg
    "South Africa":   (-29.88, 31.05),   # Durban
    "Argentina":      (-34.60, -58.37),  # Buenos Aires (port+capital)
    "UAE":            (25.00, 55.06),    # Jebel Ali (Dubai)
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
    a = math.sin(dphi / 2.0)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0)**2
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
# MULTIPLIERS FOR REALISTIC SHIPPING DISTANCES
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
            return d * LAND_MULTIPLIERS["landlocked_to_port"] if (country1 in landlocked_countries or country2 in landlocked_countries) else d * LAND_MULTIPLIERS["border"]
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

    # ---- 4) HUB-TO-HUB PENALTY (THE FIX)
    # This is the critical line: it prevents Germany, Italy, France
    # from acting as global routing shortcut nodes.
    HUB_PENALTY_KM = 5000   # Equivalent to a huge shipping detour

    if (country1 in major_hubs) and (country2 in major_hubs):
        d += HUB_PENALTY_KM

    return d


# ============================================================
# 3. BUILD FULL DISTANCE MATRIX (PORT or CAPITAL for LANDLOCKED)
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
# 4. DEFINE HUBS + REALISTIC REGIONAL NEIGHBORS
# ============================================================

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

# Realistic regional neighbor connections (land borders or short-sea)
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
    # South Africa / Nigeria connections are NOT direct here (reach via hubs)
}


# ============================================================
# 5. BUILD RESTRICTED DISTANCE MATRIX (HUB + NEIGHBOR LOGIC)
# ============================================================

print("="*70)
print("BUILDING RESTRICTED HUB NETWORK (MAX 2–3 HUBS PER COUNTRY)")
print("="*70)

# Initialize with infinity (no edge)
dist_matrix_restricted = pd.DataFrame(
    index=countries,
    columns=countries,
    data=np.inf
)

# Diagonal = 0
for country in countries:
    dist_matrix_restricted.loc[country, country] = 0.0

connections_added = {c: [] for c in countries}

# ============================================================
# CLEAN LOGIC: 1) Add neighbors (highest priority)
#              2) Add hubs ONLY if not already connected
# ============================================================

for country in countries:

    used_partners = set()   # Track who we already connected to

    # ---- 1. Add neighbor connections FIRST (real geography)
    if country in regional_neighbors:
        for neighbor in regional_neighbors[country]:
            if neighbor in countries and neighbor not in used_partners:
                d_neighbor = neighbor_distance(country, neighbor)
                dist_matrix_restricted.loc[country, neighbor] = d_neighbor
                dist_matrix_restricted.loc[neighbor, country] = d_neighbor
                connections_added[country].append(f"{neighbor} (neighbor)")
                used_partners.add(neighbor)

    # ---- 2. Add HUB connections SECOND (only if not already connected)
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
print("\nConnectivity Summary (each country: hubs + neighbors):")
for country in sorted(countries):
    num_connections = len(connections_added[country])
    conn_list = ', '.join(connections_added[country])
    print(f"  {country:20} → {num_connections:2d} connections: {conn_list}")

# ============================================================
# 5.b BLOCK LANDLOCKED COUNTRIES FROM ACTING AS TRANSIT NODES
# ============================================================

# Landlocked countries (e.g. Switzerland) should not be used
# as intermediate transit points in Tunisia → ... → destination paths.
# They can be DESTINATIONS, but we do not want paths like:
# Tunisia → France → Switzerland → Germany → China
# which are logistically nonsense.
for ll in landlocked_countries:
    for c in countries:
        if c != ll:
            # Block all OUTGOING edges from landlocked country ll
            dist_matrix_restricted.loc[ll, c] = np.inf

# ============================================================
# 6. SHIPPING COST MODEL (COST = BASE + DIST * RATE)
# ============================================================

LANES = {
    "short":  {"max": 2000,  "base": 250.0, "per_km": 0.18},
    "medium": {"max": 8000,  "base": 350.0, "per_km": 0.15},
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
# 7. DIJKSTRA ON NORMALIZED COSTS
# ============================================================

def dijkstra(cost_matrix_df, source_country):
    countries_list = list(cost_matrix_df.index)
    distances = {country: float('inf') for country in countries_list}
    previous = {country: None for country in countries_list}
    distances[source_country] = 0.0

    pq = [(0.0, source_country)]
    visited = set()

    while pq:
        current_dist, current = heapq.heappop(pq)

        if current in visited:
            continue
        visited.add(current)

        for neighbor in countries_list:
            if neighbor == current:
                continue

            edge_cost = cost_matrix_df.loc[current, neighbor]

            if not np.isfinite(edge_cost):
                continue  # no edge

# ---------------------------------------------------------    
            new_dist = current_dist + edge_cost
            if new_dist < distances[neighbor]:
                distances[neighbor] = new_dist
                previous[neighbor] = current
                heapq.heappush(pq, (new_dist, neighbor))

    return distances, previous

def reconstruct_path(previous, source, destination):
    path = []
    current = destination
    while current is not None:
        path.append(current)
        current = previous[current]
    path.reverse()
    return path

print("\n" + "="*70)
print("OPTIMAL PATHS FROM TUNISIA (BASED ON NORMALIZED LOGISTICS COST)")
print("="*70)

optimal_distances, previous_nodes = dijkstra(norm_cost_matrix, "Tunisia")

for destination in countries:
    if destination == "Tunisia":
        continue
    total_cost = optimal_distances[destination]
    if np.isfinite(total_cost):
        path_list = reconstruct_path(previous_nodes, "Tunisia", destination)
        path_str = " → ".join(path_list)
        print(f"{path_str}  (Normalized cost: {total_cost:.3f})")
    else:
        print(f"Tunisia → {destination}: NO ROUTE AVAILABLE")

# ============================================================
# 8. BASIC NETWORK STATISTICS
# ============================================================

print("\n" + "="*70)
print("NETWORK STATISTICS (ON NORMALIZED COST-GRAPH)")
print("="*70)

all_results = []
for destination in countries:
    if destination == "Tunisia":
        continue
    total_cost = optimal_distances[destination]
    if np.isfinite(total_cost):
        path_list = reconstruct_path(previous_nodes, "Tunisia", destination)
        num_hops = len(path_list) - 1
        all_results.append({
            "Destination": destination,
            "Hops": num_hops,
            "Norm_Path_Cost": round(total_cost, 3)
        })

results_df = pd.DataFrame(all_results)
print(f"Reachable destinations: {len(results_df)} / {len(countries)-1}")
if not results_df.empty:
    print(f"Average path length: {results_df['Hops'].mean():.1f} hops")
    print(f"Max path length: {results_df['Hops'].max():.0f} hops")
    print("\nTop 10 cheapest (normalized cost) destinations from Tunisia:")
    print(results_df.sort_values("Norm_Path_Cost").head(10).to_string(index=False))

import folium
import networkx as nx

# ============================================================
# 1. MERGE PORT + CAPITAL COORDS INTO A SINGLE MAP DICTIONARY
# ============================================================

country_coords = {}
for c in countries:
    if c in landlocked_countries:
        country_coords[c] = capital_coords[c]
    else:
        country_coords[c] = port_coords[c]


# ============================================================
# 2. BUILD NETWORKX GRAPH BASED ON DIJKSTRA OUTPUT
# ============================================================

G = nx.DiGraph()

# Add all nodes first
for country in countries:
    lat, lon = country_coords[country]

    # Determine tier by hops
    if country == "Tunisia":
        tier = "Source"
    else:
        total_cost = optimal_distances[country]

        if not np.isfinite(total_cost):
            tier = "Unreachable"
        else:
            path = reconstruct_path(previous_nodes, "Tunisia", country)
            hops = len(path) - 1

            if hops == 1:
                tier = "Tier 1"       # Direct link
            elif hops == 2:
                tier = "Tier 2"       # One hub
            elif hops == 3:
                tier = "Tier 3"       # Two hubs
            else:
                tier = "Tier 4"       # Complex path

    G.add_node(country, lat=lat, lon=lon, tier=tier)


# ============================================================
# 3. ADD EDGES BASED ON DIJKSTRA PATHS
# ============================================================

for country in countries:
    if country == "Tunisia":
        continue

    if not np.isfinite(optimal_distances[country]):
        continue  # skip unreachable

    path = reconstruct_path(previous_nodes, "Tunisia", country)
    hops = len(path) - 1

    for i in range(len(path) - 1):
        u = path[i]
        v = path[i + 1]

        # Edge classification
        if i == 0 and hops == 1:
            edge_type = "direct_export"
        elif i == 0:
            edge_type = "transit_hub"
        elif i == len(path) - 2:
            edge_type = "final_leg"
        else:
            edge_type = "intermediate"

        distance_km = dist_matrix_restricted.loc[u, v]

        G.add_edge(u, v, 
                   weight=distance_km,
                   type=edge_type,
                   hops=hops)


# ============================================================
# 4. BUILD FOLIUM MAP
# ============================================================

m = folium.Map(location=[20, 0], zoom_start=2, tiles="CartoDB positron")

tier_colors = {
    "Source": "green",
    "Tier 1": "red",
    "Tier 2": "orange",
    "Tier 3": "yellow",
    "Tier 4": "purple",
    "Unreachable": "gray"
}

edge_styles = {
    "direct_export": {"color": "#27ae60", "weight": 4, "dash": False},
    "transit_hub":   {"color": "#e74c3c", "weight": 3, "dash": "5,5"},
    "final_leg":     {"color": "#f39c12", "weight": 3, "dash": "10,5"},
    "intermediate":  {"color": "#9b59b6", "weight": 2, "dash": "2,4"},
}


# Add nodes with labels
for node, data in G.nodes(data=True):

    folium.CircleMarker(
        location=[data['lat'], data['lon']],
        radius=12 if node == "Tunisia" else 7,
        color=tier_colors[data['tier']],
        fill=True,
        fillColor=tier_colors[data['tier']],
        fillOpacity=0.85,
        popup=f"{node} ({data['tier']})",
        tooltip=f"{node} – {data['tier']}",
    ).add_to(m)

    # Add permanent label
    folium.Marker(
        location=[data['lat'], data['lon']],
        icon=folium.DivIcon(
            html=f"""
                <div style="
                    font-size: 11pt; 
                    color: black; 
                    font-weight: bold;
                    text-shadow: -1px -1px 0 white, 
                                 1px -1px 0 white, 
                                 -1px 1px 0 white, 
                                 1px 1px 0 white;
                    white-space: nowrap;">
                    {node}
                </div>
            """
        )
    ).add_to(m)


# Add edges
for u, v, data in G.edges(data=True):

    lat1, lon1 = G.nodes[u]['lat'], G.nodes[u]['lon']
    lat2, lon2 = G.nodes[v]['lat'], G.nodes[v]['lon']

    style = edge_styles.get(data['type'], {"color": "gray", "weight": 1, "dash": True})

    folium.PolyLine(
        locations=[[lat1, lon1], [lat2, lon2]],
        color=style["color"],
        weight=style["weight"],
        opacity=0.75,
        dashArray=style["dash"],
        tooltip=f"{u} → {v}<br>{int(data['weight'])} km<br>{data['type']}"
    ).add_to(m)


# ============================================================
# 5. ADD LEGEND
# ============================================================

legend = """
<div style="
position: fixed;
bottom: 40px;
left: 40px;
width: 240px;
height: 250px;
background-color: white;
border:2px solid grey;
z-index:9999;
font-size:13px;
padding: 10px;
opacity: 0.92;
">
<b>Tunisia Optimal Export Paths</b><br><br>

<b>Country Tiers</b><br>
<span style='color:green;'>●</span> Source (Tunisia)<br>
<span style='color:red;'>●</span> Tier 1 (Direct)<br>
<span style='color:orange;'>●</span> Tier 2 (1 Hub)<br>
<span style='color:yellow;'>●</span> Tier 3 (2 Hubs)<br>
<span style='color:purple;'>●</span> Tier 4 (≥3 Hubs)<br>
<span style='color:gray;'>●</span> Unreachable<br><br>

<b>Edge Types</b><br>
Green – Direct export<br>
Red – First hop to hub<br>
Orange – Final leg<br>
Purple – Intermediate hops<br>
</div>
"""

m.get_root().html.add_child(folium.Element(legend))

# ============================================================
# DISPLAY MAP
# ============================================================

m.save("tunisia_routes.html")
print("Map saved → tunisia_routes.html")
