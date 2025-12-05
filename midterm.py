import heapq
import math
import random
import concurrent.futures
import time
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Set, Optional
import matplotlib.pyplot as plt
import numpy as np
from collections import deque

# --- 1. CONFIGURACIÓN DEL ESCENARIO ---
GRID_SIZE = 35
OBSTACLE_RATIO = 0.30
RANDOM_SEED = 100

random.seed(RANDOM_SEED)


# --- 2. ESTRUCTURAS DE DATOS ---

@dataclass
class VectorCost:
    c1: float
    c2: float

    def __add__(self, other: "VectorCost") -> "VectorCost":
        return VectorCost(self.c1 + other.c1, self.c2 + other.c2)

    def __lt__(self, other: "VectorCost") -> bool:
        if abs(self.c1 - other.c1) > 1e-6:
            return self.c1 < other.c1
        return self.c2 < other.c2

    def __eq__(self, other):
        if not isinstance(other, VectorCost):
            return NotImplemented
        return abs(self.c1 - other.c1) < 1e-6 and abs(self.c2 - other.c2) < 1e-6

    def __repr__(self):
        return f"({self.c1:.1f}, {self.c2:.1f})"


@dataclass(order=True)
class Node:
    f_score: VectorCost
    g_score: VectorCost = field(compare=False)
    position: Tuple[int, int] = field(compare=False)
    parent: Optional["Node"] = field(default=None, compare=False)

    def __hash__(self):
        return hash(self.position)


# --- 3. PARETO LOGIC ---

def dominates(a: VectorCost, b: VectorCost) -> bool:
    return (a.c1 <= b.c1 and a.c2 <= b.c2) and (a.c1 < b.c1 or a.c2 < b.c2)

def is_dominated_by_set(candidate: VectorCost, pareto_set: List[VectorCost]) -> bool:
    return any(dominates(c, candidate) for c in pareto_set)

def remove_dominated_by_new(new: VectorCost, pareto_set: List[VectorCost]) -> List[VectorCost]:
    return [c for c in pareto_set if not dominates(new, c)]


# --- 4. MAPA Y CARGA ---

def heavy_computation():
    _ = [math.sin(x) * math.cos(x) for x in range(15000)]

class GridMap:
    def __init__(self, size: int, obstacles: Set[Tuple[int, int]]):
        self.size = size
        self.obstacles = obstacles
        self.danger_map = {
            (x, y): random.choice([1, 2, 5, 10, 20])
            for x in range(size)
            for y in range(size)
        }

    def get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        x, y = pos
        moves = [(0,1),(0,-1),(1,0),(-1,0)]
        results = []
        for dx, dy in moves:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.size and 0 <= ny < self.size and (nx, ny) not in self.obstacles:
                results.append((nx, ny))
        return results

def heuristic(a, b):
    return VectorCost(float(abs(a[0] - b[0]) + abs(a[1] - b[1])), 0.0)


# --- 5. FUNCIÓN GLOBAL PARA MULTIPROCESO ---

def expand_neighbor(args):
    current_g, neighbor_pos, goal_pos, danger_map = args
    heavy_computation()
    edge_cost = VectorCost(1.0, float(danger_map[neighbor_pos]))
    new_g = current_g + edge_cost
    new_h = heuristic(neighbor_pos, goal_pos)
    new_f = new_g + new_h
    return (new_f, new_g, neighbor_pos)


# --- 6. MOA* PARALELO CPU ---

class MOAStar:
    def __init__(self, grid_map: GridMap):
        self.grid = grid_map
        self.pareto_frontier = {}

    def search(self, start, goal, use_parallel=False):
        open_set = []
        start_node = Node(f_score=heuristic(start, goal), g_score=VectorCost(0,0), position=start)
        heapq.heappush(open_set, start_node)

        self.pareto_frontier = {start: [VectorCost(0,0)]}
        final_solutions = []

        executor = concurrent.futures.ProcessPoolExecutor(max_workers=4)

        try:
            while open_set:
                current = heapq.heappop(open_set)

                if current.position == goal:
                    final_costs = [n.g_score for n in final_solutions]
                    if not any(current.g_score == fc for fc in final_costs):
                        if not is_dominated_by_set(current.g_score, final_costs):
                            final_solutions = [
                                n for n in final_solutions if not dominates(current.g_score, n.g_score)
                            ]
                            final_solutions.append(current)
                    continue

                neighbors = self.grid.get_neighbors(current.position)

                valid_children = []

                if use_parallel and len(neighbors) > 1:
                    args_list = [
                        (current.g_score, n, goal, self.grid.danger_map)
                        for n in neighbors
                    ]
                    results = executor.map(expand_neighbor, args_list)
                    valid_children.extend(results)
                else:
                    for n in neighbors:
                        valid_children.append(
                            expand_neighbor((current.g_score, n, goal, self.grid.danger_map))
                        )

                for (new_f, new_g, pos) in valid_children:
                    frontier = self.pareto_frontier.setdefault(pos, [])

                    if is_dominated_by_set(new_g, frontier):
                        continue

                    filtered = remove_dominated_by_new(new_g, frontier)
                    filtered.append(new_g)
                    self.pareto_frontier[pos] = filtered

                    heapq.heappush(open_set, Node(new_f, new_g, pos, current))

        finally:
            executor.shutdown()

        return final_solutions


# --- 7. VISUALIZACIÓN ---

def reconstruct_path(node):
    path = []
    curr = node
    while curr:
        path.append(curr.position)
        curr = curr.parent
    return path[::-1]

def plot_pareto_results(grid_map, solutions, start, goal):
    size = grid_map.size
    grid_img = np.ones((size, size))

    max_d = max(grid_map.danger_map.values())
    for (x,y), d in grid_map.danger_map.items():
        grid_img[y,x] = 1 - (d / (1.2 * max_d))

    for (x,y) in grid_map.obstacles:
        grid_img[y,x] = 0.0

    plt.figure(figsize=(10,10))
    plt.imshow(grid_img, cmap='gray', origin='lower')

    solutions.sort(key=lambda s: s.g_score.c1)
    colors = plt.cm.plasma(np.linspace(0,0.9,len(solutions)))

    print(f"\n--- FRONTERA DE PARETO ({len(solutions)} soluciones) ---")
    for i,(node,color) in enumerate(zip(solutions,colors)):
        path = reconstruct_path(node)
        px,py = zip(*path)
        d,r = node.g_score.c1, node.g_score.c2
        print(f"Sol {i+1}: Dist {d:.1f}, Riesgo {r:.1f}")
        plt.plot(px,py,color=color,linewidth=2)

    plt.scatter(*start,color='lime',s=120)
    plt.scatter(*goal,color='blue',s=120)
    plt.title("Frontera de Pareto MOA* CPU Paralelo")
    plt.show()


# ---------------------------------------------------------------------
#  FUNCIÓN AUXILIAR: revisar conectividad del mapa
# ---------------------------------------------------------------------
def is_reachable(grid, start, goal):
    q = deque([start])
    visited = {start}

    while q:
        x, y = q.popleft()
        if (x, y) == goal:
            return True
        for nx, ny in grid.get_neighbors((x, y)):
            if (nx, ny) not in visited:
                visited.add((nx, ny))
                q.append((nx, ny))
    return False


# ---------------------------------------------------------------------
#  GENERAR GRID QUE SIEMPRE TENGA CAMINO
# ---------------------------------------------------------------------
def generate_connected_grid(size, obstacle_ratio):
    start, goal = (0, 0), (size - 1, size - 1)

    while True:
        # Generar obstáculos
        obstacles = set()
        target = int(size * size * obstacle_ratio)

        while len(obstacles) < target:
            obstacles.add((random.randint(0,size-1), random.randint(0,size-1)))

        obstacles.discard(start)
        obstacles.discard(goal)

        # Crear grid y verificar conectividad
        grid = GridMap(size, obstacles)

        if is_reachable(grid, start, goal):
            return grid, obstacles
        else:
            print(f"   (Reintentando {size}x{size} → mapa desconectado...)")


# ---------------------------------------------------------------------
#  BENCHMARK MULTIGRID ASEGURANDO CONECTIVIDAD
# ---------------------------------------------------------------------
def benchmark_multiple_grids(grid_sizes):
    sequential_times = []
    parallel_times = []

    for size in grid_sizes:
        print(f"\n--- GRID {size}x{size} ---")

        grid, obstacles = generate_connected_grid(size, OBSTACLE_RATIO)
        start, goal = (0, 0), (size - 1, size - 1)

        print(f"  Obstáculos: {len(obstacles)}  (Mapa conectado ✓)")

        moa = MOAStar(grid)

        # Secuencial
        t0 = time.time()
        moa.search(start, goal, use_parallel=False)
        t1 = time.time()
        seq_time = t1 - t0
        sequential_times.append(seq_time)
        print(f"  Secuencial: {seq_time:.2f}s")

        # Paralelo
        t0 = time.time()
        moa.search(start, goal, use_parallel=True)
        t1 = time.time()
        par_time = t1 - t0
        parallel_times.append(par_time)
        print(f"  Paralelo:   {par_time:.2f}s")

    return sequential_times, parallel_times



# --- 10. MAIN PROGRAM ---

if __name__ == "__main__":
    obstacles = set()
    while len(obstacles) < int(GRID_SIZE**2 * OBSTACLE_RATIO):
        obstacles.add((random.randint(0,GRID_SIZE-1), random.randint(0,GRID_SIZE-1)))

    start, goal = (0,0), (GRID_SIZE-1, GRID_SIZE-1)
    obstacles.discard(start)
    obstacles.discard(goal)

    mapper = GridMap(GRID_SIZE, obstacles)
    times = {}
    final_solutions = []

    print(f"Iniciando Benchmark MOA* (CPU paralelo real)...\n")

    for mode in ["Secuencial", "Paralelo"]:
        moa = MOAStar(mapper)
        use_parallel = (mode == "Paralelo")

        print(f" -> Modo {mode}...")
        t0 = time.perf_counter()
        solutions = moa.search(start, goal, use_parallel)
        t1 = time.perf_counter()

        elapsed = t1 - t0
        times[mode] = elapsed

        print(f"    Tiempo: {elapsed:.2f}s  | Soluciones: {len(solutions)}")

        if mode == "Paralelo":
            final_solutions = solutions

    speedup = times["Secuencial"] / times["Paralelo"]
    print(f"\nSPEEDUP REAL CPU = {speedup:.2f}x\n")

    if final_solutions:
        plot_pareto_results(mapper, final_solutions, start, goal)

    # --------- BENCHMARK MULTIGRID -----------

    grid_sizes = [20, 25, 30, 35, 40]

    seq_times, par_times = benchmark_multiple_grids(grid_sizes)

    plt.figure(figsize=(10,6))
    plt.plot(grid_sizes, seq_times, marker='o', label='Secuencial')
    plt.plot(grid_sizes, par_times, marker='o', label='Paralelo CPU')
    plt.xlabel('Tamaño del Grid (NxN)')
    plt.ylabel('Tiempo (segundos)')
    plt.title('Comparación de Tiempos MOA* Secuencial vs Paralelo CPU')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("benchmark_moa.png", dpi=200)
    plt.show()

    print("\nGráfica generada: benchmark_moa.png")