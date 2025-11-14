import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Any, Dict

try:
    import tomllib  # Python 3.11+
    def _load_toml(path: str) -> Dict[str, Any]:
        with open(path, "rb") as f:
            return tomllib.load(f)
except ImportError:  # Fallback to external "toml" package
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
    NSP mínimo:
    - Cada enfermero trabaja a lo sumo un turno por día.
    - Demanda mínima de enfermeros por día y turno.
    - Se penaliza demanda no cubierta y secuencias largas de días seguidos.
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
        self.off_shift = self.num_shifts  # entero que representa "día libre"

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

    def random_schedule(self) -> List[List[int]]:
        """Matriz [nurse][day] con valores en [0..num_shifts] (incluye 'off_shift')."""
        schedule = []
        for _ in range(self.num_nurses):
            row = [random.randint(0, self.num_shifts) for _ in range(self.num_days)]
            schedule.append(row)
        return schedule

    def cost(self, schedule: List[List[int]]) -> float:
        """Función objetivo: menor es mejor."""
        cost = 0.0

        # 1) Penalizar demanda no cubierta
        for d in range(self.num_days):
            for s in range(self.num_shifts):
                assigned = sum(1 for n in range(self.num_nurses)
                               if schedule[n][d] == s)
                required = self.demand[d][s]
                if assigned < required:
                    cost += (required - assigned) * self.penalty_uncovered

        # 2) Penalizar secuencias demasiado largas de días trabajados
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


@dataclass
class ACOConfig:
    num_ants: int = 20
    num_iterations: int = 100
    alpha: float = 1.0       # importancia de feromona
    beta: float = 0.0        # (no se usa en este mínimo)
    rho: float = 0.1         # evaporación
    initial_pheromone: float = 1.0
    random_seed: Optional[int] = None

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ACOConfig":
        return cls(
            num_ants=d.get("num_ants", 20),
            num_iterations=d.get("num_iterations", 100),
            alpha=d.get("alpha", 1.0),
            beta=d.get("beta", 0.0),
            rho=d.get("rho", 0.1),
            initial_pheromone=d.get("initial_pheromone", 1.0),
            random_seed=d.get("random_seed"),
        )


class ACOSolverNSP:
    """
    ACO mínimo para NSP:
    - Feromona en cada decisión (nurse, day, shift).
    - Cada hormiga construye un horario completo.
    - Actualización global sobre la mejor solución de la iteración.
    """

    def __init__(self, problem: NurseSchedulingProblem, cfg: ACOConfig):
        self.problem = problem
        self.cfg = cfg
        if cfg.random_seed is not None:
            random.seed(cfg.random_seed)

        # Matriz 3D: [nurse][day][shift (incluye off)]
        self.pheromone = [
            [
                [cfg.initial_pheromone for _ in range(self.problem.num_shifts + 1)]
                for _ in range(self.problem.num_days)
            ]
            for _ in range(self.problem.num_nurses)
        ]

    def _select_shift(self, nurse: int, day: int) -> int:
        """Selecciona un turno para (nurse, day) usando ruleta basada en feromona."""
        tau_list = self.pheromone[nurse][day]
        alpha = self.cfg.alpha
        weights = [tau ** alpha for tau in tau_list]
        total = sum(weights)
        if total <= 0:
            return random.randint(0, self.problem.num_shifts)

        r = random.random() * total
        acc = 0.0
        for shift, w in enumerate(weights):
            acc += w
            if r <= acc:
                return shift
        return len(weights) - 1  # fallback numérico

    def _construct_solution(self) -> List[List[int]]:
        schedule = []
        for nurse in range(self.problem.num_nurses):
            row = []
            for day in range(self.problem.num_days):
                shift = self._select_shift(nurse, day)
                row.append(shift)
            schedule.append(row)
        return schedule

    def _evaporate(self):
        rho = self.cfg.rho
        for n in range(self.problem.num_nurses):
            for d in range(self.problem.num_days):
                for s in range(self.problem.num_shifts + 1):
                    self.pheromone[n][d][s] *= (1.0 - rho)

    def _deposit(self, schedule: List[List[int]], cost: float):
        """Deposita feromona en la mejor solución."""
        delta = 1.0 if cost <= 0 else 1.0 / cost
        for n in range(self.problem.num_nurses):
            for d in range(self.problem.num_days):
                s = schedule[n][d]
                self.pheromone[n][d][s] += delta

    def run(self) -> Tuple[List[List[int]], float]:
        best_sched: Optional[List[List[int]]] = None
        best_cost = float("inf")

        for _ in range(self.cfg.num_iterations):
            iteration_best_sched: Optional[List[List[int]]] = None
            iteration_best_cost = float("inf")

            for _ in range(self.cfg.num_ants):
                sched = self._construct_solution()
                c = self.problem.cost(sched)
                if c < iteration_best_cost:
                    iteration_best_cost = c
                    iteration_best_sched = sched

            if iteration_best_sched is not None and iteration_best_cost < best_cost:
                best_cost = iteration_best_cost
                best_sched = iteration_best_sched

            self._evaporate()
            if iteration_best_sched is not None:
                self._deposit(iteration_best_sched, iteration_best_cost)

        assert best_sched is not None
        return best_sched, best_cost


def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    if path is None:
        # Por defecto, archivo TOML con el mismo nombre que este script.
        default_path = Path(__file__).with_suffix(".toml")
        path = str(default_path)
    return _load_toml(path)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="ACO mínimo para NSP")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Ruta al archivo de configuración TOML (opcional).",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    problem_cfg = cfg["problem"]
    aco_cfg = cfg.get("aco", {})

    problem = NurseSchedulingProblem.from_dict(problem_cfg)
    aco_conf = ACOConfig.from_dict(aco_cfg)

    solver = ACOSolverNSP(problem, aco_conf)
    best_sched, best_cost = solver.run()

    print(f"Mejor costo encontrado: {best_cost:.3f}")
    print("Horario (nurse x day) con valores de turno (0..num_shifts-1, num_shifts = libre):")
    for n, row in enumerate(best_sched):
        print(f"Enfermero {n:2d}: {row}")


if __name__ == "__main__":
    main()
