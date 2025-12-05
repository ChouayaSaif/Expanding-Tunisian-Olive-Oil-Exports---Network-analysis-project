# app.py
from flask import Flask, jsonify, render_template, request
import math
from graph_data import get_countries, get_adjacency

app = Flask(__name__)

# Load graph data once at startup
COUNTRIES = get_countries()
ADJ = get_adjacency()

SOURCE_COUNTRY = "Tunisia"

# Build reverse adjacency for bidirectional Dijkstra
RADJ = {n: [] for n in ADJ}
for u, edges in ADJ.items():
    for e in edges:
        RADJ[e["to"]].append({
            "to": u,
            "cost": e["cost"],
            "distanceKm": e["distanceKm"]
        })


# ---------- Util: quick lookups ----------
country_index = {c["id"]: c for c in COUNTRIES}


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def heuristic(from_id, to_id):
    """Heuristic for A*: geographic distance scaled ~0–1."""
    cf = country_index.get(from_id)
    ct = country_index.get(to_id)
    if not cf or not ct:
        return 0.0
    d_km = haversine_km(cf["lat"], cf["lon"], ct["lat"], ct["lon"])
    return d_km / 20000.0  # rough scaling


# ============================================================
# 1. DIJKSTRA — MIN COST
# ============================================================
def dijkstra_min_cost(src, dest):
    import heapq
    INF = float("inf")

    dist = {n: INF for n in ADJ}
    prev = {n: None for n in ADJ}
    dist[src] = 0.0

    pq = [(0.0, src)]

    while pq:
        current_dist, u = heapq.heappop(pq)
        if u == dest:
            break
        if current_dist > dist[u]:
            continue

        for edge in ADJ[u]:
            v = edge["to"]
            new_dist = current_dist + edge["cost"]
            if new_dist < dist[v]:
                dist[v] = new_dist
                prev[v] = u
                heapq.heappush(pq, (new_dist, v))

    return dist, prev


# ============================================================
# 2. BIDIRECTIONAL DIJKSTRA — MIN COST
# ============================================================
def bidirectional_dijkstra_min_cost(src, dest):
    """
    Bidirectional Dijkstra:
      - one search from src (forward, using ADJ)
      - one search from dest (backward, using RADJ)
      - meet in the middle at 'meeting' node
    Returns: (best_cost, full_path)
    """
    import heapq
    INF = float("inf")

    if src == dest:
        return 0.0, [src]

    dist_f = {n: INF for n in ADJ}
    dist_b = {n: INF for n in ADJ}
    prev_f = {n: None for n in ADJ}
    prev_b = {n: None for n in ADJ}

    dist_f[src] = 0.0
    dist_b[dest] = 0.0

    pq_f = [(0.0, src)]
    pq_b = [(0.0, dest)]

    visited_f = set()
    visited_b = set()

    best = INF
    meeting = None

    while pq_f and pq_b:
        # ---- Forward step ----
        if pq_f:
            current_dist_f, u = heapq.heappop(pq_f)
            if current_dist_f <= dist_f[u]:
                visited_f.add(u)

                if u in visited_b:
                    total = dist_f[u] + dist_b[u]
                    if total < best:
                        best = total
                        meeting = u

                for edge in ADJ[u]:
                    v = edge["to"]
                    nd = dist_f[u] + edge["cost"]
                    if nd < dist_f[v]:
                        dist_f[v] = nd
                        prev_f[v] = u
                        heapq.heappush(pq_f, (nd, v))

        # ---- Backward step ----
        if pq_b:
            current_dist_b, u2 = heapq.heappop(pq_b)
            if current_dist_b <= dist_b[u2]:
                visited_b.add(u2)

                if u2 in visited_f:
                    total = dist_f[u2] + dist_b[u2]
                    if total < best:
                        best = total
                        meeting = u2

                for edge in RADJ[u2]:
                    v2 = edge["to"]
                    nd2 = dist_b[u2] + edge["cost"]
                    if nd2 < dist_b[v2]:
                        dist_b[v2] = nd2
                        prev_b[v2] = u2
                        heapq.heappush(pq_b, (nd2, v2))

        # Simple stopping condition: we already found a meeting point
        if meeting is not None:
            break

    if meeting is None or not math.isfinite(best):
        return float("inf"), []

    # Reconstruct path src -> meeting
    path_f = []
    cur = meeting
    while cur is not None:
        path_f.append(cur)
        cur = prev_f[cur]
    path_f.reverse()  # src ... meeting

    # Reconstruct path meeting -> dest using backward parents
    path_b = []
    cur = meeting
    while cur is not None:
        path_b.append(cur)
        cur = prev_b[cur]
    # path_b is meeting ... dest (because we started from dest), so:
    path_b = path_b[1:]  # remove meeting to avoid duplication

    full_path = path_f + path_b
    return best, full_path


# ============================================================
# 3. A* — MIN COST WITH GEOGRAPHIC HEURISTIC
# ============================================================
def astar_min_cost(src, dest):
    """
    A* search:
      g(n) = accumulated normalized cost
      h(n) = geographic heuristic (scaled haversine)
    """
    import heapq
    INF = float("inf")

    g = {n: INF for n in ADJ}
    f = {n: INF for n in ADJ}
    prev = {n: None for n in ADJ}

    g[src] = 0.0
    f[src] = heuristic(src, dest)

    open_set = [(f[src], src)]

    while open_set:
        _, current = heapq.heappop(open_set)
        if current == dest:
            break

        for edge in ADJ[current]:
            v = edge["to"]
            tentative_g = g[current] + edge["cost"]
            if tentative_g < g[v]:
                g[v] = tentative_g
                f[v] = tentative_g + heuristic(v, dest)
                prev[v] = current
                heapq.heappush(open_set, (f[v], v))

    return g, prev


# ============================================================
# Helpers
# ============================================================
def reconstruct_path(prev, src, dest):
    path = []
    cur = dest
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    return path if path and path[0] == src else []


def build_result_from_path(algorithm_name, src, dest, total_cost, path):
    if not path:
        return {
            "algorithm": algorithm_name,
            "source": src,
            "destination": dest,
            "path": [],
            "segments": [],
            "totalCost": float("inf"),
            "hops": 0,
            "reachable": False
        }

    segments = []
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        edge = next((e for e in ADJ[u] if e["to"] == v), None)
        if edge:
            segments.append({
                "from": u,
                "to": v,
                "segmentCost": edge["cost"],
                "distanceKm": edge["distanceKm"]
            })

    return {
        "algorithm": algorithm_name,
        "source": src,
        "destination": dest,
        "path": path,
        "segments": segments,
        "totalCost": total_cost,
        "hops": len(path) - 1,
        "reachable": True
    }


def build_result(algorithm_name, src, dest, dist_map, prev_map):
    best = dist_map.get(dest, float("inf"))
    path = reconstruct_path(prev_map, src, dest)
    return build_result_from_path(algorithm_name, src, dest, best, path)


# ============================================================
# API ROUTES
# ============================================================

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/countries")
def api_countries():
    return jsonify(COUNTRIES)


@app.route("/api/route")
def api_route():
    dest = request.args.get("destination")
    algo = request.args.get("algorithm", "DIJKSTRA_COST")

    if dest not in ADJ:
        return jsonify({"error": "Invalid destination"}), 400

    src = SOURCE_COUNTRY

    if algo == "DIJKSTRA_COST":
        dist, prev = dijkstra_min_cost(src, dest)
        res = build_result("DIJKSTRA_COST", src, dest, dist, prev)

    elif algo == "DIJKSTRA_BIDIR":
        total, path = bidirectional_dijkstra_min_cost(src, dest)
        res = build_result_from_path("DIJKSTRA_BIDIR", src, dest, total, path)

    elif algo == "ASTAR":
        dist, prev = astar_min_cost(src, dest)
        res = build_result("ASTAR", src, dest, dist, prev)

    else:
        return jsonify({"error": "Unknown algorithm"}), 400

    return jsonify(res)


if __name__ == "__main__":
    app.run(debug=True)
