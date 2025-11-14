import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

# ------------------------
# Carga de configuración
# ------------------------
try:
    import tomllib  # Python 3.11+
    def _load_toml(path: str) -> Dict[str, Any]:
        with open(path, "rb") as f:
            return tomllib.load(f)
except ImportError:
    import toml  # type: ignore
    def _load_toml(path: str) -> Dict[str, Any]:
        return toml.load(path)


# ------------------------
# Modelo del problema TSP
# ------------------------
@dataclass
class ProblemConfig:
    # Lista de ciudades como [[x, y], [x, y], ...]
    cities: List[List[float]]
    metric: str = "euclidean"


class TSProblem:
    """
    TSP mínimo:
    - Instancia dada por coordenadas 2D.
    - Distancias euclidianas simétricas.
    - Tour: lista de índices de ciudades, cerrando el ciclo.
    """

    def __init__(self, cfg: ProblemConfig):
        self.cities = cfg.cities
        self.metric = cfg.metric
        self.num_cities = len(self.cities)
        self.dist_matrix = self._build_distance_matrix()

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TSProblem":
        cfg = ProblemConfig(
            cities=d["cities"],
            metric=d.get("metric", "euclidean"),
        )
        return cls(cfg)

    def _distance(self, i: int, j: int) -> float:
        (x1, y1) = self.cities[i]
        (x2, y2) = self.cities[j]
        return math.hypot(x1 - x2, y1 - y2)

    def _build_distance_matrix(self) -> List[List[float]]:
        n = self.num_cities
        dist = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                d = self._distance(i, j)
                dist[i][j] = d
                dist[j][i] = d
        return dist

    def tour_length(self, tour: List[int]) -> float:
        total = 0.0
        n = self.num_cities
        for i in range(n):
            j = (i + 1) % n
            total += self.dist_matrix[tour[i]][tour[j]]
        return total

    def random_tour(self) -> List[int]:
        tour = list(range(self.num_cities))
        random.shuffle(tour)
        return tour


# ------------------------
# Configuración de ACO
# ------------------------
@dataclass
class ACOConfig:
    num_ants: int = 20
    num_iterations: int = 100
    alpha: float = 1.0        # influencia de feromona
    beta: float = 2.0         # influencia de visibilidad (1/dist)
    rho: float = 0.1          # tasa de evaporación
    q: float = 1.0            # cantidad base de depósito
    initial_pheromone: float = 0.1
    random_seed: Optional[int] = None

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ACOConfig":
        return cls(
            num_ants=d.get("num_ants", 20),
            num_iterations=d.get("num_iterations", 100),
            alpha=d.get("alpha", 1.0),
            beta=d.get("beta", 2.0),
            rho=d.get("rho", 0.1),
            q=d.get("q", 1.0),
            initial_pheromone=d.get("initial_pheromone", 0.1),
            random_seed=d.get("random_seed"),
        )


# ------------------------
# Solver ACO para TSP
# ------------------------
class ACOSolverTSP:
    """
    ACO clásico para TSP:
    - Feromona en cada arista (i, j).
    - Heurística: 1 / dist(i, j).
    - Cada hormiga construye un tour completo.
    - Actualización global con todas las hormigas.
    """

    def __init__(self, problem: TSProblem, cfg: ACOConfig):
        self.problem = problem
        self.cfg = cfg
        if cfg.random_seed is not None:
            random.seed(cfg.random_seed)

        n = self.problem.num_cities
        self.pheromone = [
            [cfg.initial_pheromone for _ in range(n)] for _ in range(n)
        ]
        self.visibility = self._build_visibility_matrix()

    def _build_visibility_matrix(self) -> List[List[float]]:
        n = self.problem.num_cities
        vis = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i == j:
                    vis[i][j] = 0.0
                else:
                    d = self.problem.dist_matrix[i][j]
                    vis[i][j] = 1.0 / d if d > 0 else 0.0
        return vis

    def _construct_tour(self) -> List[int]:
        n = self.problem.num_cities
        start = random.randint(0, n - 1)
        tour = [start]
        unvisited = set(range(n))
        unvisited.remove(start)

        alpha = self.cfg.alpha
        beta = self.cfg.beta

        current = start
        while unvisited:
            probs = []
            cities = list(unvisited)
            for j in cities:
                tau = self.pheromone[current][j] ** alpha
                eta = self.visibility[current][j] ** beta
                probs.append((j, tau * eta))

            total = sum(p for _, p in probs)
            if total <= 0:
                next_city = random.choice(cities)
            else:
                r = random.random() * total
                acc = 0.0
                next_city = cities[-1]
                for city, p in probs:
                    acc += p
                    if r <= acc:
                        next_city = city
                        break

            tour.append(next_city)
            unvisited.remove(next_city)
            current = next_city

        return tour

    def _evaporate(self):
        n = self.problem.num_cities
        rho = self.cfg.rho
        for i in range(n):
            for j in range(n):
                self.pheromone[i][j] *= (1.0 - rho)

    def _deposit(self, tours: List[List[int]]):
        n = self.problem.num_cities
        q = self.cfg.q
        for tour in tours:
            length = self.problem.tour_length(tour)
            if length <= 0:
                continue
            delta = q / length
            for i in range(n):
                j = (i + 1) % n
                a = tour[i]
                b = tour[j]
                self.pheromone[a][b] += delta
                self.pheromone[b][a] += delta

    def run(self) -> Tuple[List[int], float]:
        best_tour: Optional[List[int]] = None
        best_length = float("inf")

        for _ in range(self.cfg.num_iterations):
            tours = []
            for _ in range(self.cfg.num_ants):
                tour = self._construct_tour()
                tours.append(tour)

                length = self.problem.tour_length(tour)
                if length < best_length:
                    best_length = length
                    best_tour = tour

            self._evaporate()
            self._deposit(tours)

        assert best_tour is not None
        return best_tour, best_length


# ------------------------
# Glue: carga de TOML y main
# ------------------------
def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    if path is None:
        default_path = Path(__file__).with_suffix(".toml")
        path = str(default_path)
    return _load_toml(path)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="ACO mínimo para TSP")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Ruta al archivo de configuración TOML.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    problem_cfg = cfg["problem"]
    aco_cfg = cfg.get("aco", {})

    problem = TSProblem.from_dict(problem_cfg)
    aco_conf = ACOConfig.from_dict(aco_cfg)

    solver = ACOSolverTSP(problem, aco_conf)
    best_tour, best_length = solver.run()

    print(f"Mejor longitud de tour: {best_length:.3f}")
    print("Mejor tour (índices de ciudades):")
    print(best_tour)


if __name__ == "__main__":
    main()
