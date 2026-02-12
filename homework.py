import math
import heapq
from collections import deque, defaultdict

INPUT = "input.txt"
OUTPUT = "output.txt"
PATHLEN_OUT = "pathlen.txt"

def euclid2(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

def euclid3(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)

def parse_input(path):
    with open(path, 'r') as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()!='']
    idx = 0
    algo = lines[idx].strip(); idx += 1
    energy_limit = int(lines[idx]); idx += 1
    momentum_limit = int(lines[idx]); idx += 1
    N = int(lines[idx]); idx += 1
    coords = {}
    for _ in range(N):
        parts = lines[idx].split()
        idx += 1
        name = parts[0]
        x,y,z = int(parts[1]), int(parts[2]), int(parts[3])
        coords[name] = (x,y,z)
    M = int(lines[idx]); idx += 1
    adj = defaultdict(list)
    for _ in range(M):
        a,b = lines[idx].split()
        idx += 1
        adj[a].append(b)
        adj[b].append(a)
    return algo, energy_limit, momentum_limit, coords, adj

def allowed_and_next_momentum(ucoord, vcoord, cur_momentum, energy_limit, momentum_limit):
    # energy needed (z2 - z1)
    e = vcoord[2] - ucoord[2]
    if e <= 0:
        # downhill or flat: always allowed
        next_m = min(momentum_limit, cur_momentum + (-e))
        return True, next_m
    else:
        # uphill: allowed if e <= cur_momentum + energy_limit
        if e <= cur_momentum + energy_limit:
            # momentum consumed first: used = min(cur_momentum, e)
            next_m = max(0, cur_momentum - e)
            return True, next_m
        else:
            return False, None

def reconstruct_path(parent, end_state):
    # parent: dict mapping (node, momentum) -> (prev_node, prev_momentum)
    path = []
    node, mom = end_state
    while True:
        path.append(node)
        if (node, mom) not in parent:
            break
        node, mom = parent[(node, mom)]
    return list(reversed(path))

def bfs(start, goal, energy_limit, momentum_limit, coords, adj):
    start_state = (start, 0)
    q = deque()
    q.append(start_state)
    visited = set()
    visited.add(start_state)
    parent = {}
    while q:
        node, mom = q.popleft()
        if node == goal:
            return reconstruct_path(parent, (node, mom))
        for nb in adj[node]:
            allowed, next_m = allowed_and_next_momentum(coords[node], coords[nb], mom, energy_limit, momentum_limit)
            if not allowed:
                continue
            s = (nb, next_m)
            if s in visited:
                continue
            visited.add(s)
            parent[s] = (node, mom)
            q.append(s)
    return None

def ucs(start, goal, energy_limit, momentum_limit, coords, adj):
    start_state = (start, 0)
    # heap of (cost, counter, node, momentum)
    heap = []
    counter = 0
    heapq.heappush(heap, (0.0, counter, start, 0))
    best = {}  # (node,mom) -> best_cost
    best[(start,0)] = 0.0
    parent = {}
    while heap:
        cost, _, node, mom = heapq.heappop(heap)
        if best.get((node,mom), float('inf')) < cost - 1e-12:
            continue
        if node == goal:
            return reconstruct_path(parent, (node,mom))
        for nb in adj[node]:
            allowed, next_m = allowed_and_next_momentum(coords[node], coords[nb], mom, energy_limit, momentum_limit)
            if not allowed:
                continue
            step = euclid2(coords[node], coords[nb])
            new_cost = cost + step
            key = (nb, next_m)
            if new_cost + 1e-12 < best.get(key, float('inf')):
                best[key] = new_cost
                counter += 1
                heapq.heappush(heap, (new_cost, counter, nb, next_m))
                parent[key] = (node, mom)
    return None

def astar(start, goal, energy_limit, momentum_limit, coords, adj):
    start_state = (start, 0)
    goal_coord = coords[goal]
    # heap of (f = g+h, g, counter, node, momentum)
    heap = []
    counter = 0
    h0 = euclid3(coords[start], goal_coord)
    heapq.heappush(heap, (h0, 0.0, counter, start, 0))
    best = {}  # (node,mom) -> best_g
    best[(start,0)] = 0.0
    parent = {}
    while heap:
        f, g, _, node, mom = heapq.heappop(heap)
        if best.get((node,mom), float('inf')) < g - 1e-12:
            continue
        if node == goal:
            return reconstruct_path(parent, (node,mom))
        for nb in adj[node]:
            allowed, next_m = allowed_and_next_momentum(coords[node], coords[nb], mom, energy_limit, momentum_limit)
            if not allowed:
                continue
            step = euclid3(coords[node], coords[nb])
            g2 = g + step
            key = (nb, next_m)
            if g2 + 1e-12 < best.get(key, float('inf')):
                best[key] = g2
                h = euclid3(coords[nb], goal_coord)
                counter += 1
                heapq.heappush(heap, (g2 + h, g2, counter, nb, next_m))
                parent[key] = (node, mom)
    return None

def compute_path_length(path, algo, coords):
    if path is None:
        return None
    if len(path) <= 1:
        return 0.0
    alg = algo.strip().upper()
    total = 0.0
    if alg == "BFS":
        # each move counts 1
        total = float(len(path)-1)
    elif alg == "UCS":
        # 2D Euclidean distances
        for i in range(len(path)-1):
            a = coords[path[i]]; b = coords[path[i+1]]
            total += euclid2(a,b)
    else:
        # treat anything starting with 'A' as A* (A* or astar)
        for i in range(len(path)-1):
            a = coords[path[i]]; b = coords[path[i+1]]
            total += euclid3(a,b)
    return total

def main():
    algo, energy_limit, momentum_limit, coords, adj = parse_input(INPUT)
    # find start and goal names
    start = None
    goal = None
    for name in coords:
        if name == "start":
            start = "start"
        if name == "goal":
            goal = "goal"
    if start is None or goal is None:
        with open(OUTPUT, 'w') as f:
            f.write("FAIL")
        with open(PATHLEN_OUT, 'w') as f:
            f.write("FAIL")
        return

    solution = None
    if algo.upper() == "BFS":
        solution = bfs(start, goal, energy_limit, momentum_limit, coords, adj)
    elif algo.upper() == "UCS":
        solution = ucs(start, goal, energy_limit, momentum_limit, coords, adj)
    elif algo.upper() in ("A*", "ASTAR", "ASTAR"):  # accept A* variants
        solution = astar(start, goal, energy_limit, momentum_limit, coords, adj)
    else:
        # unknown algorithm
        with open(OUTPUT, 'w') as f:
            f.write("FAIL")
        with open(PATHLEN_OUT, 'w') as f:
            f.write("FAIL")
        return

    # write path to output.txt
    with open(OUTPUT, 'w') as f:
        if solution is None:
            f.write("FAIL")
        else:
            f.write(" ".join(solution))

    # compute and write path length to pathlen.txt
    pathlen_value = compute_path_length(solution, algo, coords)
    with open(PATHLEN_OUT, 'w') as f:
        if pathlen_value is None:
            f.write("FAIL")
        else:
            # write full precision float (BFS will appear as e.g. 19.0)
            f.write(repr(float(pathlen_value)))

if __name__ == "__main__":
    main()
