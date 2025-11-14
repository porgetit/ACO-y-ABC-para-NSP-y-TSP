import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Any, Dict

try:
    import tomllib  # Python 3.11+
    def _load_toml(path: str) -> Dict[str, Any]:
        with open(path, "rb") as f:
            return tomllib.load(f)
except ImportError:  # Fallback a paquete externo "toml"
    import toml  # type: ignore
    def _load_toml(path: str) -> Dict[str, Any]:
        return toml.load(path)


@dataclass
class ProblemConfig:
    num_nurses: int
    num_days: int
    num_shifts: int
    demand: List[List[int]]
    max_consecutive_days: int = 5
    penalty_uncovered: float = 10.0
    penalty_consecutive: float = 1.0


class NurseSchedulingProblem:
    """
    NSP mínimo, mismo modelo que en la versión ACO.
    """

    def __init__(self, cfg: ProblemConfig):
        self.cfg = cfg
        self.num_nurses = cfg.num_nurses
        self.num_days = cfg.num_days
        self.num_shifts = cfg.num_shifts
        self.demand = cfg.demand
        self.max_consec = cfg.max_consecutive_days
        self.penalty_uncovered = cfg.penalty_uncovered
        self.penalty_consec = cfg.penalty_consecutive
        self.off_shift = self.num_shifts
        self.vector_length = self.num_nurses * self.num_days

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "NurseSchedulingProblem":
        cfg = ProblemConfig(
            num_nurses=d["num_nurses"],
            num_days=d["num_days"],
            num_shifts=d["num_shifts"],
            demand=d["demand"],
            max_consecutive_days=d.get("max_consecutive_days", 5),
            penalty_uncovered=d.get("penalty_uncovered", 10.0),
            penalty_consecutive=d.get("penalty_consecutive", 1.0),
        )
        return cls(cfg)

    def random_vector(self) -> List[int]:
        """Vector de longitud N*D, valores en [0..num_shifts] (incluye 'off_shift')."""
        return [random.randint(0, self.num_shifts) for _ in range(self.vector_length)]

    def vector_to_schedule(self, vec: List[int]) -> List[List[int]]:
        """Convierte un vector plano en matriz [nurse][day]."""
        schedule: List[List[int]] = []
        idx = 0
        for _ in range(self.num_nurses):
            row = []
            for _ in range(self.num_days):
                row.append(vec[idx])
                idx += 1
            schedule.append(row)
        return schedule

    def cost(self, schedule: List[List[int]]) -> float:
        cost = 0.0
        # Demanda
        for d in range(self.num_days):
            for s in range(self.num_shifts):
                assigned = sum(1 for n in range(self.num_nurses)
                               if schedule[n][d] == s)
                required = self.demand[d][s]
                if assigned < required:
                    cost += (required - assigned) * self.penalty_uncovered
        # Secuencias largas
        for n in range(self.num_nurses):
            consec = 0
            for day in range(self.num_days):
                if schedule[n][day] != self.off_shift:
                    consec += 1
                    if consec > self.max_consec:
                        cost += self.penalty_consec * (consec - self.max_consec)
                else:
                    consec = 0
        return cost

    def cost_from_vector(self, vec: List[int]) -> float:
        sched = self.vector_to_schedule(vec)
        return self.cost(sched)


@dataclass
class ABCConfig:
    colony_size: int = 20   # número de fuentes de alimento (SN)
    max_cycles: int = 200   # iteraciones
    limit: int = 50         # máx. intentos sin mejora
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
    vector: List[int]
    cost: float
    fitness: float
    trials: int = 0


class ABCSolverNSP:
    """
    Implementación mínima de Artificial Bee Colony (ABC) para NSP.

    Vecindario discreto:
    - Se elige otra fuente k.
    - En una dimensión j se copia el valor de k o se asigna un turno aleatorio.
    """

    def __init__(self, problem: NurseSchedulingProblem, cfg: ABCConfig):
        self.problem = problem
        self.cfg = cfg
        if cfg.random_seed is not None:
            random.seed(cfg.random_seed)
        self.food_sources: List[FoodSource] = []

    def _fitness(self, cost: float) -> float:
        # Fitness típico para minimización
        return 1.0 / (1.0 + cost)

    def _init_colony(self):
        self.food_sources = []
        for _ in range(self.cfg.colony_size):
            vec = self.problem.random_vector()
            cost = self.problem.cost_from_vector(vec)
            fit = self._fitness(cost)
            self.food_sources.append(FoodSource(vec, cost, fit, 0))

    def _mutate_vector(self, i: int) -> List[int]:
        """Genera un vecino discreto para la fuente i."""
        phi = random.random()
        j = random.randint(0, self.problem.vector_length - 1)
        k = i
        while k == i:
            k = random.randint(0, self.cfg.colony_size - 1)

        vec_i = self.food_sources[i].vector
        vec_k = self.food_sources[k].vector
        new_vec = list(vec_i)

        if phi < 0.5:
            new_vec[j] = vec_k[j]
        else:
            new_vec[j] = random.randint(0, self.problem.num_shifts)

        return new_vec

    def _employed_bees_phase(self):
        for i in range(self.cfg.colony_size):
            current = self.food_sources[i]
            candidate_vec = self._mutate_vector(i)
            candidate_cost = self.problem.cost_from_vector(candidate_vec)
            if candidate_cost < current.cost:
                current.vector = candidate_vec
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
                candidate_vec = self._mutate_vector(i)
                candidate_cost = self.problem.cost_from_vector(candidate_vec)
                if candidate_cost < current.cost:
                    current.vector = candidate_vec
                    current.cost = candidate_cost
                    current.fitness = self._fitness(candidate_cost)
                    current.trials = 0
                else:
                    current.trials += 1
            i = (i + 1) % self.cfg.colony_size

    def _scout_bees_phase(self):
        for i, fs in enumerate(self.food_sources):
            if fs.trials >= self.cfg.limit:
                vec = self.problem.random_vector()
                cost = self.problem.cost_from_vector(vec)
                fit = self._fitness(cost)
                self.food_sources[i] = FoodSource(vec, cost, fit, 0)

    def run(self) -> Tuple[List[List[int]], float]:
        self._init_colony()
        best_fs = min(self.food_sources, key=lambda fs: fs.cost)

        for _ in range(self.cfg.max_cycles):
            self._employed_bees_phase()
            self._onlooker_bees_phase()
            self._scout_bees_phase()

            current_best = min(self.food_sources, key=lambda fs: fs.cost)
            if current_best.cost < best_fs.cost:
                best_fs = FoodSource(
                    list(current_best.vector),
                    current_best.cost,
                    current_best.fitness,
                    current_best.trials,
                )

        best_schedule = self.problem.vector_to_schedule(best_fs.vector)
        return best_schedule, best_fs.cost


def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    if path is None:
        default_path = Path(__file__).with_suffix(".toml")
        path = str(default_path)
    return _load_toml(path)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="ABC mínimo para NSP")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Ruta al archivo de configuración TOML (opcional).",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    problem_cfg = cfg["problem"]
    abc_cfg = cfg.get("abc", {})

    problem = NurseSchedulingProblem.from_dict(problem_cfg)
    abc_conf = ABCConfig.from_dict(abc_cfg)

    solver = ABCSolverNSP(problem, abc_conf)
    best_sched, best_cost = solver.run()

    print(f"Mejor costo encontrado: {best_cost:.3f}")
    print("Horario (nurse x day) con valores de turno (0..num_shifts-1, num_shifts = libre):")
    for n, row in enumerate(best_sched):
        print(f"Enfermero {n:2d}: {row}")


if __name__ == "__main__":
    main()
