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
    cities: List[List[float]]
    metric: str = "euclidean"


class TSProblem:
    """
    TSP mínimo: mismas consideraciones del modelo ACO.
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
# Config ABC
# ------------------------
@dataclass
class ABCConfig:
    colony_size: int = 20  # número de fuentes SN
    max_cycles: int = 200
    limit: int = 50        # intentos sin mejora
    random_seed: Optional[int] = None

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ABCConfig":
        return cls(
            colony_size=d.get("colony_size", 20),
            max_cycles=d.get("max_cycles", 200),
            limit=d.get("limit", 50),
            random_seed=d.get("random_seed"),
        )


@dataclass
class FoodSource:
    tour: List[int]
    cost: float
    fitness: float
    trials: int = 0


# ------------------------
# Solver ABC para TSP
# ------------------------
class ABCSolverTSP:
    """
    ABC mínimo para TSP:

    - Cada fuente es un tour (permutación de ciudades).
    - Vecindario: intercambio (swap) de dos posiciones aleatorias.
    """

    def __init__(self, problem: TSProblem, cfg: ABCConfig):
        self.problem = problem
        self.cfg = cfg
        if cfg.random_seed is not None:
            random.seed(cfg.random_seed)
        self.food_sources: List[FoodSource] = []

    def _fitness(self, cost: float) -> float:
        return 1.0 / (1.0 + cost)

    def _init_colony(self):
        self.food_sources = []
        for _ in range(self.cfg.colony_size):
            tour = self.problem.random_tour()
            cost = self.problem.tour_length(tour)
            fit = self._fitness(cost)
            self.food_sources.append(FoodSource(tour, cost, fit, 0))

    def _mutate_tour(self, tour: List[int]) -> List[int]:
        """Vecino por swap de dos posiciones."""
        n = len(tour)
        i = random.randint(0, n - 1)
        j = random.randint(0, n - 1)
        while j == i:
            j = random.randint(0, n - 1)
        new_tour = list(tour)
        new_tour[i], new_tour[j] = new_tour[j], new_tour[i]
        return new_tour

    def _employed_bees_phase(self):
        for i in range(self.cfg.colony_size):
            current = self.food_sources[i]
            candidate_tour = self._mutate_tour(current.tour)
            candidate_cost = self.problem.tour_length(candidate_tour)

            if candidate_cost < current.cost:
                current.tour = candidate_tour
                current.cost = candidate_cost
                current.fitness = self._fitness(candidate_cost)
                current.trials = 0
            else:
                current.trials += 1

    def _onlooker_bees_phase(self):
        total_fit = sum(fs.fitness for fs in self.food_sources)
        if total_fit <= 0:
            probs = [1.0 / len(self.food_sources)] * len(self.food_sources)
        else:
            probs = [fs.fitness / total_fit for fs in self.food_sources]

        i = 0
        t = 0
        while t < self.cfg.colony_size:
            r = random.random()
            if r < probs[i]:
                t += 1
                current = self.food_sources[i]
                candidate_tour = self._mutate_tour(current.tour)
                candidate_cost = self.problem.tour_length(candidate_tour)

                if candidate_cost < current.cost:
                    current.tour = candidate_tour
                    current.cost = candidate_cost
                    current.fitness = self._fitness(candidate_cost)
                    current.trials = 0
                else:
                    current.trials += 1

            i = (i + 1) % self.cfg.colony_size

    def _scout_bees_phase(self):
        for i, fs in enumerate(self.food_sources):
            if fs.trials >= self.cfg.limit:
                new_tour = self.problem.random_tour()
                new_cost = self.problem.tour_length(new_tour)
                new_fit = self._fitness(new_cost)
                self.food_sources[i] = FoodSource(new_tour, new_cost, new_fit, 0)

    def run(self) -> Tuple[List[int], float]:
        self._init_colony()
        best_fs = min(self.food_sources, key=lambda fs: fs.cost)

        for _ in range(self.cfg.max_cycles):
            self._employed_bees_phase()
            self._onlooker_bees_phase()
            self._scout_bees_phase()

            current_best = min(self.food_sources, key=lambda fs: fs.cost)
            if current_best.cost < best_fs.cost:
                best_fs = FoodSource(
                    list(current_best.tour),
                    current_best.cost,
                    current_best.fitness,
                    current_best.trials,
                )

        return best_fs.tour, best_fs.cost


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

    parser = argparse.ArgumentParser(description="ABC mínimo para TSP")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Ruta al archivo de configuración TOML.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    problem_cfg = cfg["problem"]
    abc_cfg = cfg.get("abc", {})

    problem = TSProblem.from_dict(problem_cfg)
    abc_conf = ABCConfig.from_dict(abc_cfg)

    solver = ABCSolverTSP(problem, abc_conf)
    best_tour, best_length = solver.run()

    print(f"Mejor longitud de tour: {best_length:.3f}")
    print("Mejor tour (índices de ciudades):")
    print(best_tour)


if __name__ == "__main__":
    main()
