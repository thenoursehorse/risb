# ruff: noqa: D100, D103

from typing import Literal

import numpy as np
import pytest
from fockit.algebra import OpSum, nc, zero
from numpy.typing import NDArray
from test_common import (
    bethe_h0_k,
    build_cubic_h0_k,
    build_two_orbital_complex_h0_k,
    one_band_cubic_h0_k,
    random_unitary,
    symmetrize_blocks,
)

from risb import (
    DiscardedImaginaryWarning,
    LatticeSolver,
    PhysicalMuLatticeSolver,
    helpers,
)
from risb.gauge import lambda_eigenbasis
from risb.helpers_fockit import h_int_kanamori
from risb.impurity_solvers import FockitImpuritySolver
from risb.kintegration import BandKIntegrator, WeightedKPointIntegrator
from risb.optimize import (
    DIIS,
    BianLi,
    FixedPointLoop,
    NewtonSolver,
    OnPlateau,
    OptimizerSchedule,
    Pulay,
    ScheduleCondition,
    SchedulerPhase,
    SchedulerState,
    ScipyRoot,
    SolveResult,
    after_steps,
    all_of,
    any_of,
    plateau,
    reduced_by,
    residual_below,
)
from risb.stabilization import FullEmptyPinning, LinearWeightedDensityFeedback
from risb.types import (
    BlockMatrix,
    DensityFeedback,
    FixedPointProblem,
    GaugeFix,
    GfStruct,
    ImpuritySolver,
    PinFullEmpty,
)

type Model = tuple[
    GfStruct,
    dict[str, NDArray[np.float64]],
    ImpuritySolver,
    BandKIntegrator,
    float,
    NDArray[np.float64],
    NDArray[np.float64],
]


def do_assert(
    subtests: pytest.Subtests,
    mu: float | None,
    Lambda: list[BlockMatrix],
    Z: list[BlockMatrix],
    mu_expected: float,
    Lambda_expected: NDArray[np.float64],
    Z_expected: NDArray[np.float64],
) -> None:
    n_clusters = len(Lambda)
    abs = 1e-10
    with subtests.test(msg="mu"):
        assert mu == pytest.approx(mu_expected, abs=abs)
    # Lambda and Z depend on the gauge of the quasiparticles; their eigenvalues do not
    with subtests.test(msg="Lambda"):
        for i in range(n_clusters):
            for bl in Lambda[i]:
                assert np.linalg.eigvalsh(Lambda[i][bl]) == pytest.approx(
                    np.linalg.eigvalsh(Lambda_expected), abs=abs
                )
    with subtests.test(msg="Z"):
        for i in range(n_clusters):
            for bl in Z[i]:
                assert np.linalg.eigvalsh(Z[i][bl]) == pytest.approx(
                    np.linalg.eigvalsh(Z_expected), abs=abs
                )


@pytest.fixture
def one_band() -> Model:
    n_orb = 1
    spatial_dim = 3
    nkx = 10
    beta = 40
    U = 4
    mu = U / 2  # half-filling
    n_target = 1
    gf_struct = [(bl, n_orb) for bl in ["up", "dn"]]
    h0_k = build_cubic_h0_k(gf_struct=gf_struct, nkx=nkx, spatial_dim=spatial_dim)
    h_int = U * nc("up", 0) * nc("dn", 0)
    impurity_solver = FockitImpuritySolver(h_int, gf_struct)
    # band_integrator = BandKIntegrator(beta=beta, mu=mu)
    band_integrator = BandKIntegrator(
        WeightedKPointIntegrator(), beta=beta, n_target=n_target
    )
    Lambda_expected = np.array([[2.0]])
    Z_expected = np.array([[0.437828801025]])
    return (
        gf_struct,
        h0_k,
        impurity_solver,
        band_integrator,
        mu,
        Lambda_expected,
        Z_expected,
    )


@pytest.fixture
def bilayer() -> Model:
    U = 4
    V = 0.25
    mu = U / 2.0  # half-filling
    n_target = 2
    n_orb = 2
    spatial_dim = 3
    nkx = 10
    beta = 40
    spin_names = ["up", "dn"]
    gf_struct = [(bl, n_orb) for bl in spin_names]
    h0_k = build_cubic_h0_k(gf_struct=gf_struct, nkx=nkx, spatial_dim=spatial_dim)
    for bl in h0_k:
        h0_k[bl][:, 0, 1] += V
        h0_k[bl][:, 1, 0] += V
    h_int = zero()
    for o in range(n_orb):
        h_int += U * nc("up", o) * nc("dn", o)
    impurity_solver = FockitImpuritySolver(h_int, gf_struct)
    # band_integrator = BandKIntegrator(beta=beta, mu=mu)
    band_integrator = BandKIntegrator(
        WeightedKPointIntegrator(), beta=beta, n_target=n_target
    )
    Lambda_expected = np.array([[2.0, 0.114569681915], [0.114569681915, 2.0]])
    Z_expected = np.array([[0.452846149446, 0], [0, 0.452846149446]])
    return (
        gf_struct,
        h0_k,
        impurity_solver,
        band_integrator,
        mu,
        Lambda_expected,
        Z_expected,
    )


@pytest.fixture
def kanamori() -> Model:
    coeff = 0.2
    U = 3
    J = coeff * U
    Up = U - 2 * J
    mu = 0.5 * U + 0.5 * Up + 0.5 * (Up - J)  # half-filling
    n_target = 2
    # mu = -0.81 + (0.6899-1.1099*coeff)*U + (-0.02548+0.02709*coeff-0.1606*coeff**2)*U**2 # quarter-filling DMFT result
    n_orb = 2
    spatial_dim = 3
    nkx = 10
    beta = 40
    spin_names = ["up", "dn"]
    gf_struct = [(bl, n_orb) for bl in spin_names]
    h0_k = build_cubic_h0_k(gf_struct=gf_struct, nkx=nkx, spatial_dim=spatial_dim)
    h_int = h_int_kanamori(
        spin_names=spin_names,
        n_orb=n_orb,
        U=np.array([[0, Up - J], [Up - J, 0]]),
        Uprime=np.array([[U, Up], [Up, U]]),
        J_hund=J,
    )
    impurity_solver = FockitImpuritySolver(h_int, gf_struct)
    band_integrator = BandKIntegrator(WeightedKPointIntegrator(), beta=beta, mu=mu)
    band_integrator = BandKIntegrator(
        WeightedKPointIntegrator(), beta=beta, n_target=n_target
    )
    Lambda_expected = np.array([[3.0, 0.0], [0.0, 3.0]])
    Z_expected = np.array([[0.574940323948, 0.0], [0.0, 0.574940323948]])
    return (
        gf_struct,
        h0_k,
        impurity_solver,
        band_integrator,
        mu,
        Lambda_expected,
        Z_expected,
    )


@pytest.mark.parametrize("model", ["one_band", "bilayer", "kanamori"])
def test_diis_symmetrize(
    subtests: pytest.Subtests, request: pytest.FixtureRequest, model: str
) -> None:
    (
        gf_struct,
        h0_k,
        impurity_solver,
        band_integrator,
        mu_expected,
        Lambda_expected,
        Z_expected,
    ) = request.getfixturevalue(model)
    S = LatticeSolver(
        h0_k=h0_k,
        gf_struct=[gf_struct],
        impurity_solvers=[impurity_solver],
        band_integrator=band_integrator,
        symmetrizers=[symmetrize_blocks],
    )
    for i in range(S.n_clusters):
        for bl, _ in S.gf_struct[i]:
            np.fill_diagonal(S.Lambda[i][bl], mu_expected)
    S.solve(tol=1e-12)
    mu_calculated = band_integrator.mu
    do_assert(
        subtests, mu_calculated, S.Lambda, S.Z, mu_expected, Lambda_expected, Z_expected
    )


@pytest.mark.parametrize("model", ["one_band", "bilayer", "kanamori"])
def test_diis_nosymmetrize(
    subtests: pytest.Subtests, request: pytest.FixtureRequest, model: str
) -> None:
    (
        gf_struct,
        h0_k,
        impurity_solver,
        band_integrator,
        mu_expected,
        Lambda_expected,
        Z_expected,
    ) = request.getfixturevalue(model)
    S = LatticeSolver(
        h0_k=h0_k,
        gf_struct=[gf_struct],
        impurity_solvers=[impurity_solver],
        band_integrator=band_integrator,
    )
    for i in range(S.n_clusters):
        for bl, _ in S.gf_struct[i]:
            np.fill_diagonal(S.Lambda[i][bl], mu_expected)
    S.solve(tol=1e-12)
    mu_calculated = band_integrator.mu
    do_assert(
        subtests, mu_calculated, S.Lambda, S.Z, mu_expected, Lambda_expected, Z_expected
    )


@pytest.mark.parametrize(
    ("source_error", "policy"),
    [
        ("recursion", Pulay(error_fun="root", mixing_period=None)),
        ("root", Pulay(error_fun="recursion", mixing_period=2, restart_period=4)),
        ("root", BianLi()),
        ("root", BianLi(schedule="residual")),
        ("root", BianLi(schedule="combined")),
    ],
)
def test_optimizer_history_can_be_transferred_between_residuals(
    subtests: pytest.Subtests,
    one_band: Model,
    source_error: Literal["root", "recursion"],
    policy: Pulay | BianLi,
) -> None:
    (
        gf_struct,
        h0_k,
        impurity_solver,
        band_integrator,
        mu_expected,
        Lambda_expected,
        Z_expected,
    ) = one_band
    S = LatticeSolver(
        h0_k=h0_k,
        gf_struct=[gf_struct],
        impurity_solvers=[impurity_solver],
        band_integrator=band_integrator,
        symmetrizers=[symmetrize_blocks],
    )
    for i in range(S.n_clusters):
        for bl, _ in S.gf_struct[i]:
            np.fill_diagonal(S.Lambda[i][bl], mu_expected)
    warmup_policy = Pulay(alpha=0.1, mixing_period=1, error_fun=source_error)
    warmup = DIIS(policy=warmup_policy)
    boundary = S.solve(warmup, tol=0, maxiter=4, self_consistency_tol=np.inf)
    history = warmup.get_history()
    optimizer = DIIS(history_size=2, policy=policy)
    iteration = S.iteration
    optimizer.set_history(history)
    x_next = optimizer.update_x()
    assert S.iteration == iteration

    # Changing exported samples leaves both optimizers usable.
    for vectors in (history.x, history.g_x, history.residual):
        for vector in vectors:
            vector.fill(np.nan)
    warmup_policy.alpha = 1.0
    warmup_policy.mixing_period = None
    for solver, x0 in ((warmup, boundary.x_next), (optimizer, x_next)):
        result = S.solve(solver, x0=x0, tol=1e-12)
        assert result.converged
        do_assert(
            subtests,
            band_integrator.mu,
            S.Lambda,
            S.Z,
            mu_expected,
            Lambda_expected,
            Z_expected,
        )


@pytest.mark.parametrize("model", ["one_band", "bilayer", "kanamori"])
def test_scheduled_and_standalone_optimizers_reach_the_same_solution(
    subtests: pytest.Subtests,
    request: pytest.FixtureRequest,
    model: str,
) -> None:
    (
        gf_struct,
        h0_k,
        impurity_solver,
        band_integrator,
        mu_expected,
        Lambda_expected,
        Z_expected,
    ) = request.getfixturevalue(model)
    optimizers: dict[str, NewtonSolver] = {
        "Pulay": DIIS(),
        "BianLi": DIIS(policy=BianLi()),
    }
    for history in ("carry", "clear"):
        optimizers[f"scheduled_{history}"] = OptimizerSchedule(
            [
                SchedulerPhase(
                    DIIS(policy=Pulay(alpha=0.1, mixing_period=1)),
                    until=after_steps(4),
                ),
                SchedulerPhase(DIIS(policy=BianLi()), history=history),
            ]
        )
    reference_Z: list[BlockMatrix] | None = None
    for name, optimizer in optimizers.items():
        S = LatticeSolver(
            h0_k=h0_k,
            gf_struct=[gf_struct],
            impurity_solvers=[impurity_solver],
            band_integrator=band_integrator,
            symmetrizers=[symmetrize_blocks],
        )
        for i in range(S.n_clusters):
            for bl, _ in S.gf_struct[i]:
                np.fill_diagonal(S.Lambda[i][bl], mu_expected)
        result = S.solve(optimizer, callbacks=[], tol=1e-12)
        request.node.user_properties.append((f"evaluations_{name}", result.cycles))
        assert result.converged, name
        assert S.self_consistency_error < 1e-10, name
        do_assert(
            subtests,
            band_integrator.mu,
            S.Lambda,
            S.Z,
            mu_expected,
            Lambda_expected,
            Z_expected,
        )
        if reference_Z is None:
            reference_Z = [
                {block: matrix.copy() for block, matrix in cluster.items()}
                for cluster in S.Z
            ]
        else:
            for cluster, reference in zip(S.Z, reference_Z, strict=True):
                for block, matrix in cluster.items():
                    assert np.linalg.eigvalsh(matrix) == pytest.approx(
                        np.linalg.eigvalsh(reference[block]), abs=1e-10
                    )


@pytest.mark.parametrize(
    ("history", "condition", "boundary_steps"),
    [
        ("carry", after_steps(3), 3),
        ("clear", after_steps(3), 3),
        ("carry", after_steps(4), 4),
        ("carry", residual_below(1.0, error_fun="recursion"), 1),
        ("carry", reduced_by(1.1, error_fun="recursion"), None),
        ("carry", plateau(2, factor=1e6), 3),
        (
            "carry",
            any_of(
                after_steps(4),
                all_of(after_steps(2), residual_below(1.0, error_fun="recursion")),
            ),
            2,
        ),
    ],
)
def test_scheduled_optimizers_reach_the_physical_solution(
    subtests: pytest.Subtests,
    one_band: Model,
    history: Literal["carry", "clear"],
    condition: ScheduleCondition,
    boundary_steps: int | None,
) -> None:
    (
        gf_struct,
        h0_k,
        impurity_solver,
        band_integrator,
        mu_expected,
        Lambda_expected,
        Z_expected,
    ) = one_band
    S = LatticeSolver(
        h0_k=h0_k,
        gf_struct=[gf_struct],
        impurity_solvers=[impurity_solver],
        band_integrator=band_integrator,
        symmetrizers=[symmetrize_blocks],
    )
    for i in range(S.n_clusters):
        for bl, _ in S.gf_struct[i]:
            np.fill_diagonal(S.Lambda[i][bl], mu_expected)
    schedule = OptimizerSchedule(
        [
            SchedulerPhase(
                DIIS(policy=Pulay(alpha=0.1, mixing_period=1, restart_period=4)),
                until=condition,
            ),
            SchedulerPhase(
                DIIS(history_size=2, policy=BianLi(schedule="residual")),
                history=history,
            ),
        ]
    )
    seen: list[SchedulerState] = []
    accepted: list[tuple[NDArray[np.float64], float, float]] = []

    def record(loop: FixedPointLoop, _problem: FixedPointProblem) -> None:
        state = schedule.get_state()
        seen.append(state)
        accepted.append(
            (loop.x.copy(), state.root_norms[-1], state.recursion_norms[-1])
        )
        for error_fun in ("root", "recursion"):
            if error_fun == "root":
                best = min(accepted, key=lambda sample: sample[1])
                best_error = best[1]
            else:
                best = min(accepted, key=lambda sample: sample[2])
                best_error = best[2]
            assert loop.get_best_error(error_fun) == pytest.approx(best_error)
            assert loop.get_best_x(error_fun) == pytest.approx(best[0])
        assert loop.best_x == pytest.approx(loop.get_best_x())
        assert loop.error == pytest.approx(loop.get_error())

    result = S.solve(schedule, callbacks=[record], tol=1e-12)
    assert result.history_error_funs == [state.error_fun for state in seen]
    assert result.history == pytest.approx(
        [
            state.root_norms[-1]
            if state.error_fun == "root"
            else state.recursion_norms[-1]
            for state in seen
        ]
    )
    assert result.converged
    state = schedule.get_state()
    assert state.scheduler_phase_index == 1
    assert state.n_steps == len(seen)
    assert state.n_evaluations == result.cycles
    boundary = next(state for state in seen if state.scheduler_phase_index == 1)
    assert boundary.n_phase_steps == 0
    assert boundary.n_phase_evaluations == 0
    if boundary_steps is not None:
        assert boundary.n_steps == boundary_steps
    do_assert(
        subtests,
        band_integrator.mu,
        S.Lambda,
        S.Z,
        mu_expected,
        Lambda_expected,
        Z_expected,
    )


@pytest.mark.parametrize(
    ("reset_on_entry", "monitored_error", "incoming_error", "expected_restart_steps"),
    [
        (False, "root", "root", (5,)),
        (True, "root", "root", ()),
        (False, None, "recursion", (6,)),
        (True, None, "recursion", ()),
        (False, "root", "recursion", (5,)),
    ],
)
def test_plateau_recovery_respects_scheduler_phase_entries(
    subtests: pytest.Subtests,
    one_band: Model,
    reset_on_entry: bool,
    monitored_error: Literal["root", "recursion"] | None,
    incoming_error: Literal["root", "recursion"],
    expected_restart_steps: tuple[int, ...],
) -> None:
    (
        gf_struct,
        h0_k,
        impurity_solver,
        band_integrator,
        mu_expected,
        Lambda_expected,
        Z_expected,
    ) = one_band
    S = LatticeSolver(
        h0_k=h0_k,
        gf_struct=[gf_struct],
        impurity_solvers=[impurity_solver],
        band_integrator=band_integrator,
        symmetrizers=[symmetrize_blocks],
    )
    for i in range(S.n_clusters):
        for bl, _ in S.gf_struct[i]:
            np.fill_diagonal(S.Lambda[i][bl], mu_expected)
    schedule = OptimizerSchedule(
        [
            SchedulerPhase(
                DIIS(policy=Pulay(alpha=0.1, mixing_period=1, error_fun="root")),
                until=after_steps(2),
            ),
            SchedulerPhase(
                DIIS(
                    policy=Pulay(alpha=0.1, mixing_period=1, error_fun=incoming_error)
                ),
                until=after_steps(3),
                next_phase=1,
            ),
        ]
    )
    monitor = OnPlateau(
        patience=3,
        near=0,
        factor=1e100,
        error_fun=monitored_error,
        reset_on_scheduler_phase_entry=reset_on_entry,
    )
    restart_steps: list[int] = []

    def record(loop: FixedPointLoop, _problem: FixedPointProblem) -> None:
        if monitor.restarts > len(restart_steps):
            restart_steps.append(loop.cycle)

    result = S.solve(
        schedule,
        callbacks=[monitor, record],
        tol=0,
        maxiter=9,
        self_consistency_tol=np.inf,
    )
    assert tuple(restart_steps) == expected_restart_steps
    state = schedule.get_state()
    assert state.n_steps == 9
    assert state.scheduler_phase_index == 1
    assert state.n_phase_steps == 1
    continued = S.solve(DIIS(), x0=result.x_next, tol=1e-12)
    assert continued.converged
    do_assert(
        subtests,
        band_integrator.mu,
        S.Lambda,
        S.Z,
        mu_expected,
        Lambda_expected,
        Z_expected,
    )


def test_a_callback_residual_change_rechecks_convergence(
    subtests: pytest.Subtests, one_band: Model
) -> None:
    (
        gf_struct,
        h0_k,
        impurity_solver,
        band_integrator,
        mu_expected,
        Lambda_expected,
        Z_expected,
    ) = one_band
    S = LatticeSolver(
        h0_k=h0_k,
        gf_struct=[gf_struct],
        impurity_solvers=[impurity_solver],
        band_integrator=band_integrator,
        symmetrizers=[symmetrize_blocks],
    )
    for i in range(S.n_clusters):
        for bl, _ in S.gf_struct[i]:
            np.fill_diagonal(S.Lambda[i][bl], mu_expected)

    def switch(loop: FixedPointLoop, _problem: FixedPointProblem) -> None:
        if loop.cycle == 1:
            assert loop.converged
            assert loop.get_error("root") < loop.tol < loop.get_error("recursion")
            loop.optimizer = DIIS(policy=Pulay(error_fun="recursion"))

    result = S.solve(
        DIIS(policy=Pulay(error_fun="root")),
        callbacks=[switch],
        tol=0.2,
        self_consistency_tol=np.inf,
    )
    assert result.cycles > 1
    assert result.history_error_funs[0] == "root"
    assert result.history_error_funs[1:] == ["recursion"] * (len(result.history) - 1)
    converged = S.solve(DIIS(), x0=result.x_next, tol=1e-12)
    assert converged.converged
    do_assert(
        subtests,
        band_integrator.mu,
        S.Lambda,
        S.Z,
        mu_expected,
        Lambda_expected,
        Z_expected,
    )


def test_a_branching_schedule_continues_across_evaluation_budgets(
    subtests: pytest.Subtests, one_band: Model
) -> None:
    (
        gf_struct,
        h0_k,
        impurity_solver,
        band_integrator,
        mu_expected,
        Lambda_expected,
        Z_expected,
    ) = one_band
    S = LatticeSolver(
        h0_k=h0_k,
        gf_struct=[gf_struct],
        impurity_solvers=[impurity_solver],
        band_integrator=band_integrator,
        symmetrizers=[symmetrize_blocks],
    )
    for i in range(S.n_clusters):
        for bl, _ in S.gf_struct[i]:
            np.fill_diagonal(S.Lambda[i][bl], mu_expected)
    initial_x = S.get_x().copy()

    def make_schedule() -> OptimizerSchedule:
        return OptimizerSchedule(
            [
                SchedulerPhase(
                    DIIS(policy=Pulay(alpha=0.1, mixing_period=1)),
                    until=after_steps(2),
                    next_phase=lambda state: 2 if state.n_steps >= 6 else 1,
                ),
                SchedulerPhase(
                    DIIS(
                        policy=Pulay(alpha=0.3, mixing_period=1, error_fun="recursion")
                    ),
                    until=after_steps(3, scope="solve"),
                    next_phase=0,
                    history="clear",
                ),
                SchedulerPhase(
                    DIIS(history_size=2, policy=BianLi(schedule="constant", gamma=0.01))
                ),
            ]
        )

    runs: list[
        tuple[list[NDArray[np.float64]], list[Literal["root", "recursion"]], int]
    ] = []
    for budget in (1000, 3):
        schedule = make_schedule()
        seen: list[SchedulerState] = []
        accepted_guesses: list[NDArray[np.float64]] = []
        error_funs: list[Literal["root", "recursion"]] = []

        def record(
            loop: FixedPointLoop,
            _problem: FixedPointProblem,
            *,
            observations: list[SchedulerState] = seen,
            optimizer: OptimizerSchedule = schedule,
            guesses: list[NDArray[np.float64]] = accepted_guesses,
        ) -> None:
            observations.append(optimizer.get_state())
            guesses.append(loop.x.copy())

        x_next = initial_x
        cycles = 0
        for _ in range(200):
            result = S.solve(
                schedule,
                callbacks=[record],
                maxiter=budget,
                x0=x_next,
                tol=1e-12,
                self_consistency_tol=np.inf,
            )
            cycles += result.cycles
            error_funs.extend(result.history_error_funs)
            assert result.cycles <= budget
            if result.converged:
                break
            assert result.x_next is not None
            x_next = result.x_next
        else:
            pytest.fail("The scheduled solve did not converge.")
        runs.append((accepted_guesses, error_funs, cycles))
        state = schedule.get_state()
        assert state.n_steps == len(seen)
        assert state.n_evaluations == cycles
        assert state.n_evaluations > state.n_steps
        visited: list[int] = []
        for observation in seen:
            index = observation.scheduler_phase_index
            if not visited or index != visited[-1]:
                visited.append(index)
        assert visited == [0, 1, 0, 1, 0, 2]
        do_assert(
            subtests,
            band_integrator.mu,
            S.Lambda,
            S.Z,
            mu_expected,
            Lambda_expected,
            Z_expected,
        )
        schedule.reset_active_optimizer()
        assert schedule.get_state() == state
        continued = S.solve(schedule, callbacks=[], maxiter=3, x0=result.x, tol=1e-12)
        assert continued.converged
        assert schedule.get_state().scheduler_phase_index == 2
        schedule.reset()
        reset = schedule.get_state()
        assert reset.scheduler_phase_index == 0
        assert reset.n_steps == 0
        assert reset.n_evaluations == 0

    uninterrupted_run, continued_run = runs
    assert continued_run[1:] == uninterrupted_run[1:]
    for uninterrupted_x, continued_x in zip(
        uninterrupted_run[0], continued_run[0], strict=True
    ):
        assert continued_x == pytest.approx(uninterrupted_x, abs=1e-12, rel=0)


def test_one_cycle_per_solve_from_x_next_reaches_the_solution(
    subtests: pytest.Subtests, one_band: Model
) -> None:
    (
        gf_struct,
        h0_k,
        impurity_solver,
        band_integrator,
        mu_expected,
        Lambda_expected,
        Z_expected,
    ) = one_band
    S = LatticeSolver(
        h0_k=h0_k,
        gf_struct=[gf_struct],
        impurity_solvers=[impurity_solver],
        band_integrator=band_integrator,
        symmetrizers=[symmetrize_blocks],
    )
    for i in range(S.n_clusters):
        for bl, _ in S.gf_struct[i]:
            np.fill_diagonal(S.Lambda[i][bl], mu_expected)
    # As in charge self-consistency with one RISB cycle per step
    optimizer = DIIS()
    result = S.solve(optimizer, maxiter=1, self_consistency_tol=np.inf)
    for _ in range(200):
        if result.error < 1e-12:
            break
        result = S.solve(
            optimizer, maxiter=1, self_consistency_tol=np.inf, x0=result.x_next
        )
    assert result.error < 1e-12
    do_assert(
        subtests,
        band_integrator.mu,
        S.Lambda,
        S.Z,
        mu_expected,
        Lambda_expected,
        Z_expected,
    )


@pytest.mark.parametrize(
    ("model", "root_method"),
    [
        ("one_band", "krylov"),
        ("bilayer", "krylov"),
        ("kanamori", "krylov"),
    ],
)
def test_scipy_root(
    subtests: pytest.Subtests,
    request: pytest.FixtureRequest,
    model: str,
    root_method: str,
) -> None:
    (
        gf_struct,
        h0_k,
        impurity_solver,
        band_integrator,
        mu_expected,
        Lambda_expected,
        Z_expected,
    ) = request.getfixturevalue(model)

    S = LatticeSolver(
        h0_k=h0_k,
        gf_struct=[gf_struct],
        impurity_solvers=[impurity_solver],
        band_integrator=band_integrator,
        symmetrizers=[symmetrize_blocks],
    )
    for i in range(S.n_clusters):
        for bl, _ in S.gf_struct[i]:
            np.fill_diagonal(S.Lambda[i][bl], mu_expected)
    result = S.solve(
        optimizer=ScipyRoot(
            method=root_method, options={"fatol": 1e-10, "maxiter": 100}
        )
    )
    assert result.converged
    mu_calculated = band_integrator.mu
    do_assert(
        subtests, mu_calculated, S.Lambda, S.Z, mu_expected, Lambda_expected, Z_expected
    )


# In the atomic limit the self-consistency equations need not be satisfied
@pytest.mark.filterwarnings("ignore:The RISB self-consistency:RuntimeWarning")
def test_three_orbital_kanamori_is_a_paramagnetic_mott_insulator() -> None:
    n_orb = 3
    U = 3
    J = 0.2 * U
    Up = U - 2 * J
    spin_names = ["up", "dn"]
    gf_struct = [(bl, n_orb) for bl in spin_names]
    off_diag = 1 - np.eye(n_orb)
    h_int = h_int_kanamori(
        spin_names,
        n_orb,
        U=(Up - J) * off_diag,
        Uprime=U * np.eye(n_orb) + Up * off_diag,
        J_hund=J,
    )
    S = LatticeSolver(
        h0_k=build_cubic_h0_k(gf_struct=gf_struct, nkx=10, spatial_dim=3),
        gf_struct=gf_struct,
        impurity_solvers=FockitImpuritySolver(h_int, gf_struct),
        band_integrator=BandKIntegrator(
            WeightedKPointIntegrator(), beta=40, n_target=n_orb
        ),
        symmetrizers=[symmetrize_blocks],
    )
    assert S.solve(DIIS(policy=Pulay(error_fun="recursion")), tol=1e-8).converged
    for bl in S.Z[0]:
        assert S.Z[0][bl] == pytest.approx(np.zeros((n_orb, n_orb)), abs=1e-6)
    # The ground state of the impurity is a spin multiplet, and the symmetrizer
    # averages it into the same density in each spin
    assert S.rho_c[0]["up"] == pytest.approx(S.rho_c[0]["dn"], abs=1e-10)


def two_orbital_complex_lattice(h_int: OpSum) -> tuple[LatticeSolver, BandKIntegrator]:
    gf_struct = [(bl, 2) for bl in ["up", "dn"]]
    band_integrator = BandKIntegrator(WeightedKPointIntegrator(), beta=40, n_target=2)
    S = LatticeSolver(
        h0_k=build_two_orbital_complex_h0_k(),
        gf_struct=gf_struct,
        impurity_solvers=FockitImpuritySolver(h_int, gf_struct),
        band_integrator=band_integrator,
    )
    return S, band_integrator


def test_quasiparticle_basis_of_the_starting_guess_does_not_change_the_physics() -> (
    None
):
    U, J = 3.0, 0.6
    Up = U - 2 * J
    h_int = h_int_kanamori(
        ["up", "dn"],
        2,
        U=(Up - J) * (1 - np.eye(2)),
        Uprime=U * np.eye(2) + Up * (1 - np.eye(2)),
        J_hund=J,
    )
    results = []
    for u in [np.eye(2), random_unitary(2, seed=1), random_unitary(2, seed=2)]:
        S, band_integrator = two_orbital_complex_lattice(h_int)
        for bl in ["up", "dn"]:
            S.Lambda[0][bl] = 1.5 * np.eye(2)
            S.R[0][bl] = u.conj().T
        assert S.solve(tol=1e-10).converged
        results.append((band_integrator.mu, S.Z[0]["up"]))
    for mu, weight in results[1:]:
        assert mu == pytest.approx(results[0][0], abs=1e-8)
        assert weight == pytest.approx(results[0][1], abs=1e-8)


def test_rotated_copy_of_the_non_interacting_solution_is_converged() -> None:
    S, _ = two_orbital_complex_lattice(zero())
    u = random_unitary(2, seed=3)
    for bl in ["up", "dn"]:
        S.R[0][bl] = u.conj().T
    result = S.solve(tol=1e-10)
    assert result.converged
    assert result.cycles <= 3
    for bl in ["up", "dn"]:
        assert S.Z[0][bl] == pytest.approx(np.eye(2), abs=1e-8)


@pytest.mark.parametrize("force_real", [False, True])
def test_real_rotation_of_the_non_interacting_solution_is_converged(
    force_real: bool,
) -> None:
    gf_struct = [("up", 2), ("dn", 2)]
    angle = 0.7
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    S = LatticeSolver(
        h0_k=build_cubic_h0_k(gf_struct=gf_struct),
        gf_struct=gf_struct,
        impurity_solvers=FockitImpuritySolver(zero(), gf_struct),
        band_integrator=BandKIntegrator(
            WeightedKPointIntegrator(), beta=40, n_target=2
        ),
        force_real=force_real,
    )
    for bl, _ in gf_struct:
        S.R[0][bl] = rotation.copy()
    assert S.solve(tol=1e-10, maxiter=1).converged
    for bl, _ in gf_struct:
        assert S.Z[0][bl] == pytest.approx(np.eye(2), abs=1e-8)
        assert S.rho_c[0][bl] == pytest.approx(0.5 * np.eye(2), abs=1e-8)


def test_force_real_warns_when_discarding_imaginary_parts(one_band: Model) -> None:
    gf_struct, h0_k, impurity_solver, band_integrator, *_ = one_band
    S = LatticeSolver(
        h0_k,
        gf_struct,
        impurity_solver,
        band_integrator,
        force_real=True,
    )
    S.R[0]["up"] = np.array([[1 + 1e-6j]])
    with pytest.warns(DiscardedImaginaryWarning):
        S.get_x()


@pytest.mark.parametrize(
    "pin_full_empty", [FullEmptyPinning(), None], ids=["pinned", "not pinned"]
)
def test_full_orbital_pinning_prevents_the_wrong_impurity_filling(
    pin_full_empty: PinFullEmpty | None,
) -> None:
    # Orbital 0 is a completely full band with no hopping to orbital 1, so
    # with only density interactions orbital 1 is the one-band model, shifted
    # in energy. The full orbital has no bath partner in the impurity problem
    spin_names = ["up", "dn"]
    gf_struct = [(bl, 2) for bl in spin_names]
    h_int = h_int_kanamori(
        spin_names, 2, U=2 * (1 - np.eye(2)), Uprime=2 * np.ones((2, 2)), J_hund=0
    )
    h0_k = build_cubic_h0_k(gf_struct=gf_struct, nkx=10, spatial_dim=3)
    for bl in h0_k:
        h0_k[bl] = h0_k[bl] + np.diag([-2.0, 2.0])
    S = LatticeSolver(
        h0_k=h0_k,
        gf_struct=gf_struct,
        impurity_solvers=FockitImpuritySolver(h_int, gf_struct),
        band_integrator=BandKIntegrator(
            WeightedKPointIntegrator(), beta=40, n_target=3
        ),
        symmetrizers=[symmetrize_blocks],
        pin_full_empty=pin_full_empty,
    )
    for bl, _ in gf_struct:
        S.Lambda[0][bl] = np.diag([0.0, 6.0])
    result = S.solve(tol=1e-10, maxiter=100, self_consistency_tol=np.inf)
    filling = sum(float(np.trace(rho).real) for rho in S.rho_c[0].values())
    if pin_full_empty is None:
        assert filling == pytest.approx(2.0, abs=1e-8)
        assert S.self_consistency_error > 0.1
        for bl, _ in gf_struct:
            assert np.linalg.eigvalsh(S.Z[0][bl]) == pytest.approx([0, 0], abs=1e-8)
        return

    assert filling == pytest.approx(3.0, abs=1e-8)

    gf_struct_one = [(bl, 1) for bl in spin_names]
    one_band = LatticeSolver(
        h0_k=build_cubic_h0_k(gf_struct=gf_struct_one, nkx=10, spatial_dim=3),
        gf_struct=gf_struct_one,
        impurity_solvers=FockitImpuritySolver(
            2 * nc("up", 0) * nc("dn", 0), gf_struct_one
        ),
        band_integrator=BandKIntegrator(
            WeightedKPointIntegrator(), beta=40, n_target=1
        ),
    )
    for bl, _ in gf_struct_one:
        one_band.Lambda[0][bl] = np.array([[1.0]])
    one_band.solve(tol=1e-10)

    assert result.converged
    assert S.self_consistency_error < 1e-8
    for bl, _ in gf_struct:
        assert np.linalg.eigvalsh(S.Z[0][bl]) == pytest.approx(
            [0, one_band.Z[0][bl][0, 0].real], abs=1e-6
        )


def test_decoupled_ghosts_of_a_metal_on_ligands_converge() -> None:
    # Two metal orbitals with three quasiparticles each, hybridized with two
    # uncorrelated ligand orbitals. Two ghosts per spin decouple with
    # occupation 1/2, and the update does not change their levels
    spin_names = ["up", "dn"]
    U, J = 3.0, 0.5
    e = bethe_h0_k(2000)
    eye = np.eye(2)
    metal = np.diag([0.5, -0.5]) + 0.5 * e * eye
    ligand = ((3 * U - 5 * J) / 2 - 1.5) * eye + e * eye
    hybridization = np.broadcast_to(0.5 * eye, metal.shape)
    h0_k = np.block([[metal, hybridization], [hybridization, ligand]])
    gf_struct = [(bl, 2) for bl in spin_names]
    bath_size = dict.fromkeys(spin_names, 6)
    h_int = h_int_kanamori(
        spin_names,
        2,
        U=(U - 3 * J) * (1 - eye),
        Uprime=U * eye + (U - 2 * J) * (1 - eye),
        J_hund=0,
    )
    sz = sum((nc("up", a) - nc("dn", a) for a in range(2)), zero())
    sz += sum((nc("bath_up", a) - nc("bath_dn", a) for a in range(6)), zero())
    S = PhysicalMuLatticeSolver(
        h0_k=dict.fromkeys(spin_names, h0_k),
        gf_struct=gf_struct,
        impurity_solvers=FockitImpuritySolver(
            h_int, gf_struct, diagonal_symmetries={sz: 0}, bath_size=bath_size
        ),
        band_integrator=BandKIntegrator(WeightedKPointIntegrator(), beta=100),
        projectors=[dict.fromkeys(spin_names, np.eye(4)[:2])],
        symmetrizers=[symmetrize_blocks],
        force_real=True,
        bath_size=[bath_size],
        n_target=6,
    )
    result = S.solve(tol=1e-9, maxiter=60)

    assert result.converged
    assert S.self_consistency_error < 1e-8
    for bl, _ in gf_struct:
        rho_f = np.linalg.eigvalsh(S.rho_f[0][bl])
        assert rho_f[2:4] == pytest.approx([0.5, 0.5], abs=1e-8)


@pytest.mark.parametrize(
    "density_feedback",
    [LinearWeightedDensityFeedback(), None],
    ids=["feedback", "no feedback"],
)
def test_density_feedback_does_not_change_the_solution(
    one_band: Model, density_feedback: DensityFeedback | None
) -> None:
    gf_struct, h0_k, impurity_solver, band_integrator, _, _, Z_expected = one_band
    S = LatticeSolver(
        h0_k,
        gf_struct,
        impurity_solver,
        band_integrator,
        density_feedback=density_feedback,
    )
    S.solve(tol=1e-10)
    for bl, _ in gf_struct:
        assert S.Z[0][bl] == pytest.approx(Z_expected, abs=1e-8)


def test_ghost_risb_finds_the_saturated_ferromagnet() -> None:
    # Low filling of a band with its density of states peaked at the bottom.
    # The fully polarized state has no double occupation, so its energy is
    # that of free spinless electrons, below the paramagnet
    spin_names = ["up", "dn"]
    n, U, B = 0.4, 10.0, 3
    n_k = 400
    x = (np.arange(n_k) + 0.5) / n_k
    h0_k = (-1 + 2 * x**2)[:, None, None]
    gf_struct = [(bl, 1) for bl in spin_names]
    bath_size = dict.fromkeys(spin_names, B)

    def solve(paramagnetic: bool) -> LatticeSolver:
        band_integrator = BandKIntegrator(WeightedKPointIntegrator(), beta=100)
        S = LatticeSolver(
            h0_k=dict.fromkeys(spin_names, h0_k),
            gf_struct=gf_struct,
            impurity_solvers=FockitImpuritySolver(
                U * nc("up", 0) * nc("dn", 0), gf_struct, bath_size=bath_size
            ),
            band_integrator=band_integrator,
            symmetrizers=[symmetrize_blocks] if paramagnetic else (),
            force_real=True,
            bath_size=[bath_size],
        )
        band_integrator.n_target = n + S.n_ghost_electrons
        for bl, split in [("up", -0.2), ("dn", 0.2)]:
            S.Lambda[0][bl] = S.Lambda[0][bl] + split * np.eye(B)
        S.solve(tol=1e-9)
        assert S.self_consistency_error < 1e-8
        return S

    S = solve(paramagnetic=False)
    magnetization = np.trace(S.rho_c[0]["up"] - S.rho_c[0]["dn"]).real
    assert magnetization == pytest.approx(n, abs=1e-6)
    spinless_energy = np.sort(h0_k[:, 0, 0])[: round(n * n_k)].sum() / n_k
    assert _energy(S) == pytest.approx(spinless_energy, abs=1e-3)
    assert _energy(S) < _energy(solve(paramagnetic=True)) - 1e-3


def test_ghost_risb_finds_the_neel_antiferromagnet() -> None:
    # Half filling of a bipartite lattice with a cluster on each sublattice,
    # from a start with opposite spin splittings on the two sublattices
    spin_names = ["up", "dn"]
    U, B = 4.0, 3
    e = bethe_h0_k(400)
    h0_k = np.block([[0 * e, e], [e, 0 * e]])
    gf_struct = [(bl, 1) for bl in spin_names]
    bath_size = dict.fromkeys(spin_names, B)

    def solve(paramagnetic: bool) -> LatticeSolver:
        band_integrator = BandKIntegrator(WeightedKPointIntegrator(), beta=100)
        S = LatticeSolver(
            h0_k=dict.fromkeys(spin_names, h0_k),
            gf_struct=[gf_struct] * 2,
            impurity_solvers=[
                FockitImpuritySolver(
                    U * nc("up", 0) * nc("dn", 0), gf_struct, bath_size=bath_size
                )
                for _ in range(2)
            ],
            band_integrator=band_integrator,
            projectors=[
                dict.fromkeys(spin_names, np.eye(2)[:1]),
                dict.fromkeys(spin_names, np.eye(2)[1:]),
            ],
            symmetrizers=[symmetrize_blocks] if paramagnetic else (),
            force_real=True,
            bath_size=[bath_size] * 2,
        )
        band_integrator.n_target = 2 + S.n_ghost_electrons
        for i, sign in [(0, 1), (1, -1)]:
            for bl, split in [("up", -0.1), ("dn", 0.1)]:
                S.Lambda[i][bl] = S.Lambda[i][bl] + sign * split * np.eye(B)
        S.solve(tol=1e-9)
        assert S.self_consistency_error < 1e-8
        return S

    S = solve(paramagnetic=False)
    m = [np.trace(S.rho_c[i]["up"] - S.rho_c[i]["dn"]).real for i in range(2)]
    assert m[0] > 0.9
    assert m[1] == pytest.approx(-m[0], abs=1e-6)
    assert _energy(S) < _energy(solve(paramagnetic=True)) - 1e-3


@pytest.mark.parametrize(
    "gauge_fix", [None, lambda_eigenbasis], ids=["free gauge", "lambda eigenbasis"]
)
def test_ghost_risb_from_a_perturbed_start_reaches_the_solution_of_the_default_start(
    gauge_fix: GaugeFix | None,
) -> None:
    # Two-orbital Kanamori model on the Bethe lattice at quarter filling, with
    # three quasiparticles per orbital and spin. From this start, ghost levels
    # close to empty grow from cycle to cycle unless their steps are limited
    B = 3
    n_orb = 2
    U, J = 3.0, 0.45
    spin_names = ["up", "dn"]
    gf_struct = [(bl, n_orb) for bl in spin_names]
    bath_size = dict.fromkeys(spin_names, B * n_orb)
    Up = U - 2 * J
    off_diag = 1 - np.eye(n_orb)
    h_int = h_int_kanamori(
        spin_names,
        n_orb,
        U=(Up - J) * off_diag,
        Uprime=U * np.eye(n_orb) + Up * off_diag,
        J_hund=J,
    )
    Sz = zero()
    for o in range(n_orb):
        Sz += nc("up", o) - nc("dn", o)
    for a in range(B * n_orb):
        Sz += nc("bath_up", a) - nc("bath_dn", a)

    def solve(seed: int | None) -> tuple[LatticeSolver, FockitImpuritySolver]:
        impurity_solver = FockitImpuritySolver(
            h_int, gf_struct, diagonal_symmetries={Sz: 0}, bath_size=bath_size
        )
        band_integrator = BandKIntegrator(
            WeightedKPointIntegrator(), beta=2000, n_target=1
        )
        S = LatticeSolver(
            h0_k={bl: bethe_h0_k(2000) * np.eye(n_orb) for bl in spin_names},
            gf_struct=gf_struct,
            impurity_solvers=impurity_solver,
            band_integrator=band_integrator,
            symmetrizers=[symmetrize_blocks],
            force_real=True,
            bath_size=[bath_size],
            gauge_fix=gauge_fix,
        )
        band_integrator.n_target = 1 + S.n_ghost_electrons
        x0 = S.get_x()
        if seed is not None:
            x0 = x0 + 0.05 * np.random.default_rng(seed).standard_normal(x0.shape)
        assert S.solve(tol=1e-10, maxiter=400, x0=x0).converged
        return S, impurity_solver

    S, impurity_solver = solve(None)
    perturbed, perturbed_solver = solve(4)
    assert perturbed.Z[0]["up"] == pytest.approx(S.Z[0]["up"], abs=1e-6)
    docc = nc("up", 0) * nc("dn", 0)
    assert perturbed_solver.overlap(docc) == pytest.approx(
        impurity_solver.overlap(docc), abs=1e-6
    )


def test_two_site_cell_with_a_cluster_per_site_gives_the_one_site_solution(
    subtests: pytest.Subtests,
) -> None:
    U = 4.0
    spin_names = ["up", "dn"]

    def solve(
        h0_k: NDArray[np.complex128],
        gf_struct: list[GfStruct],
        projectors: list[BlockMatrix] | None = None,
        gf_struct_mapping: list[dict[str, str]] | None = None,
    ) -> LatticeSolver:
        S = LatticeSolver(
            h0_k=dict.fromkeys(spin_names, h0_k),
            gf_struct=gf_struct,
            impurity_solvers=[
                FockitImpuritySolver(U * nc(gf[0][0], 0) * nc(gf[1][0], 0), gf)
                for gf in gf_struct
            ],
            band_integrator=BandKIntegrator(
                WeightedKPointIntegrator(), beta=40, n_target=len(gf_struct)
            ),
            projectors=projectors,
            gf_struct_mapping=gf_struct_mapping,
            check_projectors=projectors is not None,
        )
        for i in range(S.n_clusters):
            for bl in S.Lambda[i]:
                S.Lambda[i][bl] = np.array([[U / 2]])
        S.solve(tol=1e-12)
        return S

    one_site = solve(
        one_band_cubic_h0_k(6, supercell=False), [[(bl, 1) for bl in spin_names]]
    )
    projectors = [
        dict.fromkeys(spin_names, np.array([[1.0, 0.0]])),
        dict.fromkeys(spin_names, np.array([[0.0, 1.0]])),
    ]
    h0_k = one_band_cubic_h0_k(6, supercell=True)
    with subtests.test("same blocks as h0_k"):
        S = solve(h0_k, [[(bl, 1) for bl in spin_names]] * 2, projectors)
        for i in range(2):
            assert S.Z[i]["up"] == pytest.approx(one_site.Z[0]["up"], abs=1e-10)
    with subtests.test("blocks renamed with gf_struct_mapping"):
        names = [{bl: f"{bl}_{site}" for bl in spin_names} for site in ["A", "B"]]
        S = solve(
            h0_k,
            [[(names[i][bl], 1) for bl in spin_names] for i in range(2)],
            [
                {names[i][bl]: P[bl] for bl in spin_names}
                for i, P in enumerate(projectors)
            ],
            [{names[i][bl]: bl for bl in spin_names} for i in range(2)],
        )
        for i in range(2):
            assert S.Z[i][names[i]["up"]] == pytest.approx(
                one_site.Z[0]["up"], abs=1e-10
            )


def test_check_projectors_rejects_only_projectors_that_break_the_assumptions(
    subtests: pytest.Subtests,
) -> None:
    spin_names = ["up", "dn"]
    h0 = one_band_cubic_h0_k(4, supercell=True)
    h0_k = dict.fromkeys(spin_names, h0)
    n_k = h0.shape[0]
    site_a = np.array([[1.0, 0.0]])
    site_b = np.array([[0.0, 1.0]])
    # The same rotation of both sites at every k-point
    rotations = np.array([random_unitary(2, seed=k) for k in range(n_k)])

    def lattice(projectors: list[NDArray[np.inexact]], check: bool) -> None:
        gf_struct = [[(bl, 1) for bl in spin_names]] * len(projectors)
        LatticeSolver(
            h0_k=h0_k,
            gf_struct=gf_struct,
            impurity_solvers=[
                FockitImpuritySolver(nc("up", 0) * nc("dn", 0), gf) for gf in gf_struct
            ],
            band_integrator=BandKIntegrator(
                WeightedKPointIntegrator(), beta=40, n_target=2
            ),
            projectors=[dict.fromkeys(spin_names, P) for P in projectors],
            check_projectors=check,
        )

    not_orthonormal_at_one_k = np.broadcast_to(site_a, (n_k, 1, 2)).copy()
    not_orthonormal_at_one_k[3] *= 2
    bad = {
        "overlapping": [site_a, site_a],
        "not orthonormal": [2 * site_a, site_b],
        "not orthonormal at one k-point": [not_orthonormal_at_one_k, site_b],
    }
    for case, projectors in bad.items():
        with subtests.test(case):
            with pytest.raises(ValueError, match="projectors"):
                lattice(projectors, check=True)
            # Not checked by default
            lattice(projectors, check=False)
    good = {
        "an orbital in no cluster": [site_a],
        "depending on k": [site_a @ rotations, site_b @ rotations],
    }
    for case, projectors in good.items():
        with subtests.test(case):
            lattice(projectors, check=True)


def test_orbital_in_no_cluster_keeps_its_band() -> None:
    # Site 1 in no cluster must give the same as site 1 a cluster without
    # interactions, where RISB is exact
    spin_names = ["up", "dn"]
    h0_k = dict.fromkeys(spin_names, one_band_cubic_h0_k(6, supercell=True))
    site_a = dict.fromkeys(spin_names, np.array([[1.0, 0.0]]))
    site_b = dict.fromkeys(spin_names, np.array([[0.0, 1.0]]))
    h_ints = [4 * nc("up", 0) * nc("dn", 0), zero()]

    def solve(projectors: list[BlockMatrix]) -> LatticeSolver:
        gf_struct = [[(bl, 1) for bl in spin_names]] * len(projectors)
        S = LatticeSolver(
            h0_k=h0_k,
            gf_struct=gf_struct,
            impurity_solvers=[
                FockitImpuritySolver(h_ints[i], gf) for i, gf in enumerate(gf_struct)
            ],
            band_integrator=BandKIntegrator(
                WeightedKPointIntegrator(), beta=40, n_target=2
            ),
            projectors=projectors,
        )
        for i in range(S.n_clusters):
            for bl in spin_names:
                S.Lambda[i][bl] = np.array([[2.0 if i == 0 else 0.0]])
        assert S.solve(tol=1e-12).converged
        return S

    one_cluster = solve([site_a])
    two_clusters = solve([site_a, site_b])
    for bl in spin_names:
        assert one_cluster.Z[0][bl] == pytest.approx(two_clusters.Z[0][bl], abs=1e-8)
        assert two_clusters.Z[1][bl] == pytest.approx(np.eye(1), abs=1e-8)


def test_projectors_that_depend_on_k_give_the_same_solution() -> None:
    # The lattice in a basis rotated differently at every k-point, with the
    # projectors rotated with it, is the same problem
    spin_names = ["up", "dn"]
    h0 = one_band_cubic_h0_k(6, supercell=True)
    h0[:, 0, 0] *= 1.3  # Inequivalent sites
    rotations = np.array([random_unitary(2, seed=k) for k in range(h0.shape[0])])
    sites = [np.array([[1.0, 0.0]]), np.array([[0.0, 1.0]])]

    def solve(
        h0: NDArray[np.complex128], projectors: list[NDArray[np.inexact]]
    ) -> LatticeSolver:
        gf_struct = [[(bl, 1) for bl in spin_names]] * 2
        S = LatticeSolver(
            h0_k=dict.fromkeys(spin_names, h0),
            gf_struct=gf_struct,
            impurity_solvers=[
                FockitImpuritySolver(4 * nc("up", 0) * nc("dn", 0), gf)
                for gf in gf_struct
            ],
            band_integrator=BandKIntegrator(
                WeightedKPointIntegrator(), beta=40, n_target=2
            ),
            projectors=[dict.fromkeys(spin_names, P) for P in projectors],
            check_projectors=True,
        )
        for i in range(2):
            for bl in spin_names:
                S.Lambda[i][bl] = np.array([[2.0]])
        assert S.solve(tol=1e-12).converged
        return S

    fixed = solve(h0, sites)
    rotated = solve(
        rotations @ h0 @ np.swapaxes(rotations.conj(), -1, -2),
        [P @ np.swapaxes(rotations.conj(), -1, -2) for P in sites],
    )
    for i in range(2):
        for bl in spin_names:
            assert rotated.Z[i][bl] == pytest.approx(fixed.Z[i][bl], abs=1e-8)
            assert rotated.Lambda[i][bl] == pytest.approx(fixed.Lambda[i][bl], abs=1e-8)


def test_constant_double_counting_is_a_shift_of_the_local_energies() -> None:
    U, J = 3.0, 0.6
    Up = U - 2 * J
    h_int = h_int_kanamori(
        ["up", "dn"],
        2,
        U=(Up - J) * (1 - np.eye(2)),
        Uprime=U * np.eye(2) + Up * (1 - np.eye(2)),
        J_hund=J,
    )
    gf_struct = [(bl, 2) for bl in ["up", "dn"]]
    potential = np.array([[0.3, 0.1], [0.1, -0.2]])
    calls: list[tuple[int, BlockMatrix]] = []

    def double_counting(i: int, rho_c: BlockMatrix) -> BlockMatrix:
        calls.append((i, rho_c))
        return dict.fromkeys(rho_c, potential)

    def solve(
        h0_k: dict[str, NDArray[np.complex128]], with_hook: bool
    ) -> tuple[LatticeSolver, SolveResult]:
        S = LatticeSolver(
            h0_k=h0_k,
            gf_struct=gf_struct,
            impurity_solvers=FockitImpuritySolver(h_int, gf_struct),
            band_integrator=BandKIntegrator(
                WeightedKPointIntegrator(), beta=40, n_target=2
            ),
            double_counting=double_counting if with_hook else None,
        )
        for bl, _ in gf_struct:
            S.Lambda[0][bl] = 1.5 * np.eye(2)
        return S, S.solve(tol=1e-12)

    h0_k = build_two_orbital_complex_h0_k()
    hook, result = solve(h0_k, with_hook=True)
    shifted, _ = solve({bl: h - potential for bl, h in h0_k.items()}, with_hook=False)
    assert result.converged
    for bl, _ in gf_struct:
        assert hook.Z[0][bl] == pytest.approx(shifted.Z[0][bl], abs=1e-8)
        assert hook.Lambda[0][bl] == pytest.approx(shifted.Lambda[0][bl], abs=1e-8)
    # Called every cycle with the cluster index and a matrix for each block
    assert len(calls) == result.cycles
    assert all(i == 0 and set(rho) == {"up", "dn"} for i, rho in calls)


def test_ghost_risb_with_double_counting_restarts_on_its_solution() -> None:
    # Hubbard model on the Bethe lattice away from half filling, with a Hartree
    # double counting of the density of the other spin
    B, U = 3, 2.0
    gf_struct = [("up", 1), ("dn", 1)]
    bath_size = [{bl: B for bl, _ in gf_struct}]
    h_int = U * nc("up", 0) * nc("dn", 0)

    def double_counting(_i: int, rho_c: BlockMatrix) -> BlockMatrix:
        return {"up": U * rho_c["dn"], "dn": U * rho_c["up"]}

    def lattice_solver() -> LatticeSolver:
        return LatticeSolver(
            h0_k={bl: bethe_h0_k(200) for bl, _ in gf_struct},
            gf_struct=gf_struct,
            impurity_solvers=FockitImpuritySolver(
                h_int, gf_struct, bath_size=bath_size[0]
            ),
            band_integrator=BandKIntegrator(
                WeightedKPointIntegrator(), beta=200, n_target=0.8 + (B - 1)
            ),
            symmetrizers=[symmetrize_blocks],
            force_real=True,
            double_counting=double_counting,
            bath_size=bath_size,
        )

    S = lattice_solver()
    result = S.solve(tol=1e-10)
    assert S.self_consistency_error < 1e-8
    restarted = lattice_solver()
    assert restarted.solve(x0=result.x_next, rho_c0=S.rho_c, maxiter=1).error < 1e-8
    # Changing the old solver must not change the restarted one.
    for cluster in S.rho_c:
        for rho in cluster.values():
            rho.fill(0)
    assert restarted.solve(maxiter=1).error < 1e-8


def test_ghost_risb_without_interactions_is_the_non_interacting_solution() -> None:
    B = 3
    gf_struct = [("up", 1), ("dn", 1)]
    bath_size = [{bl: B for bl, _ in gf_struct}]
    h0_k = {bl: bethe_h0_k(200) for bl, _ in gf_struct}
    impurity_solver = FockitImpuritySolver(zero(), gf_struct, bath_size=bath_size[0])
    band_integrator = BandKIntegrator(
        WeightedKPointIntegrator(), beta=200, n_target=1 + (B - 1)
    )
    S = LatticeSolver(
        h0_k=h0_k,
        gf_struct=gf_struct,
        impurity_solvers=impurity_solver,
        band_integrator=band_integrator,
        symmetrizers=[symmetrize_blocks],
        force_real=True,
        bath_size=bath_size,
    )
    S.solve(tol=1e-10)
    assert S.self_consistency_error < 1e-8
    assert band_integrator.mu is not None
    quasiparticle_weight = helpers.get_z(
        S.R[0]["up"], S.Lambda[0]["up"], band_integrator.mu
    )
    assert quasiparticle_weight == pytest.approx(np.eye(1), abs=1e-8)
    assert impurity_solver.overlap(nc("up", 0) * nc("dn", 0)) == pytest.approx(
        0.25, abs=1e-8
    )
    # Kinetic energy sum_k eps_k R^dag f(H_qp) R of both spins, -4 / (3 pi) for
    # the half-filled semicircular density of states
    amplitude = np.einsum("a,kan->kn", S.R_full["up"][:, 0], S.eigenvectors_qp["up"])
    kinetic = 2 * np.sum(
        S.weighted_occupations_qp["up"]
        * np.abs(amplitude) ** 2
        * h0_k["up"][:, 0, 0, None]
    )
    assert kinetic == pytest.approx(-4 / (3 * np.pi), abs=1e-4)


@pytest.mark.parametrize(
    ("U", "energy", "double_occupancy", "Z"),
    [(2.4, -0.061, 0.054, 0.17), (3.5, -0.037, 0.0108, 0.0)],
)
def test_ghost_risb_gives_the_published_energy_of_the_hubbard_model(
    U: float, energy: float, double_occupancy: float, Z: float
) -> None:
    # Half-filled Hubbard model on the Bethe lattice with three quasiparticles
    # per spin, a metal and a Mott insulator. Values read off Figs. 3 and 4 of
    # Lee, Lanata, and Kotliar, Phys. Rev. B 107, L121104 (2023)
    B = 3
    gf_struct = [("up", 1), ("dn", 1)]
    bath_size = [{bl: B for bl, _ in gf_struct}]
    h0_k = {bl: bethe_h0_k(200) for bl, _ in gf_struct}
    h_int = U * nc("up", 0) * nc("dn", 0) - U / 2 * (nc("up", 0) + nc("dn", 0))
    Sz = nc("up", 0) - nc("dn", 0)
    for a in range(B):
        Sz += nc("bath_up", a) - nc("bath_dn", a)
    # A thermal impurity averages the two ground states of the Mott insulator
    impurity_solver = FockitImpuritySolver(
        h_int, gf_struct, diagonal_symmetries={Sz: 0}, beta=200, bath_size=bath_size[0]
    )
    band_integrator = BandKIntegrator(
        WeightedKPointIntegrator(), beta=200, n_target=1 + (B - 1)
    )
    S = LatticeSolver(
        h0_k=h0_k,
        gf_struct=gf_struct,
        impurity_solvers=impurity_solver,
        band_integrator=band_integrator,
        symmetrizers=[symmetrize_blocks],
        force_real=True,
        bath_size=bath_size,
    )
    S.solve(tol=1e-9)
    assert S.n_ghost_electrons == pytest.approx(B - 1)
    assert S.self_consistency_error < 1e-8
    assert S.rho_c[0]["up"] == pytest.approx(np.array([[0.5]]), abs=1e-8)
    # Kinetic energy sum_k eps_k R^dag f(H_qp) R of both spins
    amplitude = np.einsum("a,kan->kn", S.R_full["up"][:, 0], S.eigenvectors_qp["up"])
    kinetic = 2 * np.sum(
        S.weighted_occupations_qp["up"]
        * np.abs(amplitude) ** 2
        * h0_k["up"][:, 0, 0, None]
    )
    docc = impurity_solver.overlap(nc("up", 0) * nc("dn", 0))
    assert docc == pytest.approx(double_occupancy, abs=1e-3)
    assert kinetic + U * docc == pytest.approx(energy, abs=1e-3)
    assert band_integrator.mu is not None
    quasiparticle_weight = helpers.get_z(
        S.R[0]["up"], S.Lambda[0]["up"], band_integrator.mu
    )
    assert quasiparticle_weight == pytest.approx(np.array([[Z]]), abs=1e-2)


def test_orbitals_with_different_numbers_of_ghosts_give_their_own_solutions() -> None:
    # Two half-filled Hubbard orbitals with no hopping or interaction between
    # them, in one cluster with B = 3 for orbital 0 and B = 1 for orbital 1.
    # Each orbital must match a cluster of that orbital alone
    U = 2.4

    def solve(bath_sizes: dict[str, int]) -> dict[str, tuple[float, float]]:
        gf_struct = [(spin + o, 1) for o in bath_sizes for spin in ["up", "dn"]]
        bath_size = {bl: bath_sizes[bl[2:]] for bl, _ in gf_struct}
        h_int = zero()
        Sz = zero()
        for o, B in bath_sizes.items():
            h_int += U * nc("up" + o, 0) * nc("dn" + o, 0)
            h_int -= U / 2 * (nc("up" + o, 0) + nc("dn" + o, 0))
            Sz += nc("up" + o, 0) - nc("dn" + o, 0)
            for a in range(B):
                Sz += nc("bath_up" + o, a) - nc("bath_dn" + o, a)
        impurity_solver = FockitImpuritySolver(
            h_int, gf_struct, diagonal_symmetries={Sz: 0}, bath_size=bath_size
        )

        def paramagnetic(A: list[BlockMatrix]) -> list[BlockMatrix]:
            for o in bath_sizes:
                A[0]["up" + o] = A[0]["dn" + o] = 0.5 * (
                    A[0]["up" + o] + A[0]["dn" + o]
                )
            return A

        band_integrator = BandKIntegrator(
            WeightedKPointIntegrator(), beta=200, n_target=len(bath_sizes)
        )
        S = LatticeSolver(
            h0_k={bl: bethe_h0_k(200) for bl, _ in gf_struct},
            gf_struct=gf_struct,
            impurity_solvers=impurity_solver,
            band_integrator=band_integrator,
            symmetrizers=[paramagnetic],
            force_real=True,
            bath_size=[bath_size],
        )
        band_integrator.n_target = len(bath_sizes) + S.n_ghost_electrons
        S.solve(tol=1e-10)
        assert S.self_consistency_error < 1e-8
        assert band_integrator.mu is not None
        solution = {}
        for o in bath_sizes:
            Z = helpers.get_z(
                S.R[0]["up" + o], S.Lambda[0]["up" + o], band_integrator.mu
            )
            docc = impurity_solver.overlap(nc("up" + o, 0) * nc("dn" + o, 0))
            solution[o] = (float(Z[0, 0].real), float(np.real(docc)))
        return solution

    mixed = solve({"_0": 3, "_1": 1})
    assert mixed["_0"] == pytest.approx(solve({"_0": 3})["_0"], abs=1e-8)
    assert mixed["_1"] == pytest.approx(solve({"_1": 1})["_1"], abs=1e-8)


def test_a_cluster_with_an_odd_number_of_ghosts_is_rejected() -> None:
    # One ghost, so the impurity problem has five orbitals and no half filling
    gf_struct = [("up", 1), ("dn", 1)]
    bath_size = {"up": 2, "dn": 1}
    with pytest.raises(ValueError, match="no half-filled sector"):
        FockitImpuritySolver(zero(), gf_struct, bath_size=bath_size)
    with pytest.raises(ValueError, match="1 ghosts"):
        LatticeSolver(
            h0_k={bl: bethe_h0_k(10) for bl, _ in gf_struct},
            gf_struct=gf_struct,
            impurity_solvers=FockitImpuritySolver(zero(), gf_struct),
            band_integrator=BandKIntegrator(
                WeightedKPointIntegrator(), beta=10, n_target=1
            ),
            bath_size=[bath_size],
        )


def _energy(S: LatticeSolver) -> float:
    # Band energy of the quasiparticles plus the local energy of each impurity
    result = S.band_integration_result
    assert result is not None
    density_k = helpers.get_density_matrix_k(
        R_full=S.R_full,
        density_qp_k=S.density_qp_k,
        rho_c=S.rho_c,
        projectors=S.projectors,
        weights=result.weights,
    )
    band = helpers.get_band_energy(S.h0_kin_k, density_k, weights=result.weights)
    local = 0.0
    for solver in S.impurity_solvers:
        assert isinstance(solver, FockitImpuritySolver)
        local += float(np.real(solver.overlap(solver.h0_loc + solver.h_int)))
    return float(band) + local
