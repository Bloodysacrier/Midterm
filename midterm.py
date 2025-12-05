import heapq
import math
import random
import concurrent.futures
import time
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Set, Optional
import matplotlib.pyplot as plt
import numpy as np

# --- 1. CONFIGURACIÓN DEL ESCENARIO ---
GRID_SIZE = 30           # Tamaño del mapa (30x30)
OBSTACLE_RATIO = 0.25    # 25% de muros
RANDOM_SEED = 42         # Semilla para reproducibilidad

random.seed(RANDOM_SEED)

# --- 2. ESTRUCTURAS DE DATOS (CORE) ---

@dataclass
class VectorCost:
    """
    Costo Vectorial (c1, c2).
    No usamos order=True para evitar conflictos con la lógica custom de _lt_.
    """
    c1: float  # Distancia (Minimizar)
    c2: float  # Riesgo/Peligrosidad (Minimizar)

    def _add_(self, other: "VectorCost") -> "VectorCost":
        return VectorCost(self.c1 + other.c1, self.c2 + other.c2)

    def _repr_(self):
        return f"({self.c1:.1f}, {self.c2:.1f})"

    # Lógica de comparación para la Cola de Prioridad (Lexicográfica)
    # Prioridad: Menor Distancia -> Menor Riesgo
    def _lt_(self, other: "VectorCost") -> bool:
        if abs(self.c1 - other.c1) > 1e-6: # Comparación con tolerancia float
            return self.c1 < other.c1
        return self.c2 < other.c2

@dataclass(order=True)
class Node:
    f_score: VectorCost
    g_score: VectorCost = field(compare=False)
    position: Tuple[int, int] = field(compare=False)
    parent: Optional["Node"] = field(default=None, compare=False)

    def _hash_(self):
        return hash(self.position)

# --- 3. LÓGICA DE PARETO (TEORÍA DE OPTIMIZACIÓN) ---

def dominates(a: VectorCost, b: VectorCost) -> bool:
    """Retorna True si el vector A domina estrictamente al vector B."""
    return (a.c1 <= b.c1 and a.c2 <= b.c2) and (a.c1 < b.c1 or a.c2 < b.c2)

def is_dominated_by_set(candidate: VectorCost, pareto_set: List[VectorCost]) -> bool:
    """Verifica si el candidato ya es superado por alguna solución existente."""
    for c in pareto_set:
        if dominates(c, candidate):
            return True
    return False

def remove_dominated_by_new(new: VectorCost, pareto_set: List[VectorCost]) -> List[VectorCost]:
    """Limpia el set eliminando soluciones antiguas que ahora son peores que la nueva."""
    return [c for c in pareto_set if not dominates(new, c)]

# --- 4. MAPA Y SIMULACIÓN ---

def heavy_computation():
    """
    SIMULACIÓN DE CARGA: Justificación académica del paralelismo.
    Simula un coste computacional alto por nodo (ej. procesamiento de imagen o física).
    Sin esto, el overhead de hilos en Python haría la versión paralela más lenta.
    """
    _ = [math.sin(x) * math.cos(x) for x in range(800)]

class GridMap:
    def _init_(self, size: int, obstacles: Set[Tuple[int, int]]):
        self.size = size
        self.obstacles = obstacles
        # Mapa de peligrosidad: 1 (seguro), 5 (medio), 15 (muy peligroso)
        self.danger_map = {}
        for x in range(size):
            for y in range(size):
                self.danger_map[(x, y)] = random.choice([1, 1, 1, 5, 15])

    def get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        x, y = pos
        moves = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        results = []
        for dx, dy in moves:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.size and 0 <= ny < self.size and (nx, ny) not in self.obstacles:
                results.append((nx, ny))
        return results

    def get_move_cost(self, pos: Tuple[int, int]) -> VectorCost:
        # Costo = (1 paso de distancia, N nivel de peligro)
        return VectorCost(1.0, float(self.danger_map.get(pos, 1)))

def heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> VectorCost:
    # Heurística admisible para distancia (Manhattan), 0 para riesgo (optimista)
    h1 = float(abs(a[0] - b[0]) + abs(a[1] - b[1]))
    return VectorCost(h1, 0.0)

# --- 5. ALGORITMO MOA* (PARALELIZABLE) ---

class MOAStar:
    def _init_(self, grid_map: GridMap):
        self.grid = grid_map
        # Diccionario: Posición -> Lista de Costos de Pareto encontrados
        self.pareto_frontier: Dict[Tuple[int, int], List[VectorCost]] = {}

    def expand_neighbor(self, args):
        """Unidad de trabajo para el ThreadPoolExecutor."""
        current_g, neighbor_pos, goal_pos = args
        
        # Inyectamos carga artificial para justificar el uso de hilos
        heavy_computation()
        
        edge_cost = self.grid.get_move_cost(neighbor_pos)
        new_g = current_g + edge_cost
        new_h = heuristic(neighbor_pos, goal_pos)
        new_f = new_g + new_h
        
        return (new_f, new_g, neighbor_pos)

    def search(self, start: Tuple[int, int], goal: Tuple[int, int], use_parallel: bool = False) -> List[Node]:
        open_set: List[Node] = []
        start_h = heuristic(start, goal)
        start_node = Node(f_score=start_h, g_score=VectorCost(0.0, 0.0), position=start)
        
        heapq.heappush(open_set, start_node)
        self.pareto_frontier = {start: [VectorCost(0.0, 0.0)]}
        
        # Almacenamos soluciones finales únicas
        final_solutions: List[Node] = []

        executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

        try:
            while open_set:
                current = heapq.heappop(open_set)

                # Si llegamos a la meta
                if current.position == goal:
                    # Chequeo de dominancia contra otras soluciones finales
                    final_g_scores = [n.g_score for n in final_solutions]
                    if not is_dominated_by_set(current.g_score, final_g_scores):
                        # Poda inversa: Eliminar soluciones anteriores si la nueva es mejor
                        final_solutions = [n for n in final_solutions if not dominates(current.g_score, n.g_score)]
                        final_solutions.append(current)
                    continue

                # Poda Lazy: Si al sacar del heap ya encontramos algo mejor para este nodo
                if is_dominated_by_set(current.g_score, self.pareto_frontier.get(current.position, [])[:-1]):
                    continue

                neighbors = self.grid.get_neighbors(current.position)
                valid_children = []

                # --- LÓGICA PARALELA VS SECUENCIAL ---
                if use_parallel and len(neighbors) > 1:
                    args_list = [(current.g_score, n, goal) for n in neighbors]
                    results = executor.map(self.expand_neighbor, args_list)
                    valid_children.extend(results)
                else:
                    for n in neighbors:
                        # En secuencial también llamamos a la carga heavy para comparación justa
                        res = self.expand_neighbor((current.g_score, n, goal))
                        valid_children.append(res)

                # --- PROCESAMIENTO DE HIJOS ---
                for (new_f, new_g, n_pos) in valid_children:
                    if n_pos not in self.pareto_frontier:
                        self.pareto_frontier[n_pos] = []
                    
                    existing = self.pareto_frontier[n_pos]
                    
                    # Chequeo de Dominancia (Critical Section)
                    if is_dominated_by_set(new_g, existing):
                        continue
                    
                    # Actualizar frontera de Pareto local
                    filtered = remove_dominated_by_new(new_g, existing)
                    filtered.append(new_g)
                    self.pareto_frontier[n_pos] = filtered
                    
                    heapq.heappush(open_set, Node(f_score=new_f, g_score=new_g, position=n_pos, parent=current))

        finally:
            executor.shutdown()
            
        return final_solutions

# --- 6. VISUALIZACIÓN Y RESULTADOS ---

def reconstruct_path(node: Node) -> List[Tuple[int, int]]:
    path = []
    curr = node
    while curr:
        path.append(curr.position)
        curr = curr.parent
    return path[::-1]

def plot_pareto_results(grid_map: GridMap, solutions: List[Node], start, goal):
    size = grid_map.size
    grid = np.ones((size, size))
    
    # Visualizar peligrosidad (mapa de calor gris)
    max_danger = max(grid_map.danger_map.values())
    for (x, y), danger in grid_map.danger_map.items():
        # Más oscuro = más peligroso
        grid[y, x] = 1 - (danger / (max_danger * 1.5))
        
    for (x, y) in grid_map.obstacles:
        grid[y, x] = 0.0 # Obstáculos en negro

    plt.figure(figsize=(10, 8))
    plt.imshow(grid, cmap='gray', origin='lower')
    
    # Generar colores distintos para cada solución óptima
    colors = plt.cm.rainbow(np.linspace(0, 1, len(solutions)))
    
    print(f"\n--- ANÁLISIS DE LA FRONTERA DE PARETO ({len(solutions)} soluciones) ---")
    
    # Ordenamos soluciones por distancia para la leyenda
    solutions.sort(key=lambda x: x.g_score.c1)

    for i, (node, color) in enumerate(zip(solutions, colors)):
        path = reconstruct_path(node)
        px, py = zip(*path)
        
        dist = node.g_score.c1
        risk = node.g_score.c2
        
        plt.plot(px, py, color=color, linewidth=3, alpha=0.8, 
                 label=f'Sol {i+1}: Dist={dist:.0f}, Riesgo={risk:.0f}')
        print(f"Solución {i+1}: Distancia {dist:.1f} | Riesgo {risk:.1f} | Nodos: {len(path)}")

    plt.scatter(*start, color='lime', s=200, edgecolors='black', label='Inicio', zorder=10)
    plt.scatter(*goal, color='blue', s=200, edgecolors='white', label='Meta', zorder=10)
    
    plt.title(f"Frontera de Pareto MOA*\nGrid {GRID_SIZE}x{GRID_SIZE} - {OBSTACLE_RATIO*100}% Obstáculos")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

# --- 7. MAIN ---

if _name_ == "_main_":
    # Generación de Obstáculos
    obstacles: Set[Tuple[int, int]] = set()
    for _ in range(int(GRID_SIZE**2 * OBSTACLE_RATIO)):
        obstacles.add((random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1)))
    
    start, goal = (0, 0), (GRID_SIZE-1, GRID_SIZE-1)
    if start in obstacles: obstacles.remove(start)
    if goal in obstacles: obstacles.remove(goal)

    mapper = GridMap(GRID_SIZE, obstacles)

    # Comparación de Tiempos
    times = {}
    final_solutions = []

    print(f"Iniciando Benchmark (Grid {GRID_SIZE}x{GRID_SIZE})...")
    
    for mode in ["Secuencial", "Paralelo"]:
        moa = MOAStar(mapper)
        use_parallel = (mode == "Paralelo")
        
        print(f" -> Ejecutando modo {mode}...")
        t0 = time.perf_counter()
        solutions = moa.search(start, goal, use_parallel=use_parallel)
        t1 = time.perf_counter()
        
        elapsed = t1 - t0
        times[mode] = elapsed
        print(f"    Tiempo: {elapsed:.4f}s | Soluciones Pareto: {len(solutions)}")
        
        if mode == "Paralelo":
            final_solutions = solutions

    # Gráfico de Barras de Rendimiento
    plt.figure(figsize=(6, 4))
    bars = plt.bar(times.keys(), times.values(), color=['skyblue', 'salmon'])
    plt.ylabel("Tiempo (s)")
    plt.title("Comparación de Rendimiento (con Carga Simulada)")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Calcular Speedup
    speedup = times["Secuencial"] / times["Paralelo"]
    print(f"\nSpeedup logrado: {speedup:.2f}x")
    
    plt.text(0.5, max(times.values())/2, f"Speedup: {speedup:.2f}x", 
             ha='center', fontsize=12, fontweight='bold', color='black')
    
    plt.show()

    # Visualizar Mapa y Caminos
    if final_solutions:
        plot_pareto_results(mapper, final_solutions, start, goal)
    else:
        print("No se encontró ningún camino a la meta.")