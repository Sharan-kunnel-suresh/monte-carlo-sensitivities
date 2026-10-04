import time

from src.mc_sensitivities.black_scholes import (
    black_scholes_call_delta,
)

from src.mc_sensitivities.sensitivities import (
    finite_difference_delta,
    pathwise_delta,
    ad_delta,
)


# ============================================================
# Model parameters
# ============================================================

S = 100.0
K = 100.0
r = 0.05
sigma = 0.20
T = 1.0

N_PATHS = 100_000
SEED = 42
H = 1e-4


# ============================================================
# Analytical benchmark
# ============================================================

black_scholes_delta = black_scholes_call_delta(
    S=S,
    K=K,
    r=r,
    sigma=sigma,
    T=T,
)


# ============================================================
# Finite Difference Delta
# ============================================================

start = time.perf_counter()

fd_central, fd_forward, fd_backward = finite_difference_delta(
    S=S,
    K=K,
    r=r,
    sigma=sigma,
    T=T,
    h=H,
    n_paths=N_PATHS,
    seed=SEED,
)

fd_runtime = time.perf_counter() - start


# ============================================================
# Pathwise Delta
# ============================================================

start = time.perf_counter()

pathwise = pathwise_delta(
    S=S,
    K=K,
    r=r,
    sigma=sigma,
    T=T,
    n_paths=N_PATHS,
    seed=SEED,
)

pathwise_runtime = time.perf_counter() - start


# ============================================================
# Algorithmic Differentiation Delta
# ============================================================

start = time.perf_counter()

ad = ad_delta(
    S=S,
    K=K,
    r=r,
    sigma=sigma,
    T=T,
    n_paths=N_PATHS,
    seed=SEED,
)

ad_runtime = time.perf_counter() - start


# ============================================================
# Errors
# ============================================================

fd_error = fd_central - black_scholes_delta
pathwise_error = pathwise - black_scholes_delta
ad_error = ad - black_scholes_delta


# ============================================================
# Results
# ============================================================

print()
print("=" * 70)
print("DELTA COMPARISON")
print("=" * 70)

print(f"Number of paths:        {N_PATHS:,}")
print(f"Random seed:            {SEED}")
print(f"Finite difference h:    {H}")
print()

print(f"{'Method':<25} {'Delta':>15} {'Error':>15} {'Runtime (s)':>15}")
print("-" * 70)

print(
    f"{'Black-Scholes':<25}"
    f"{black_scholes_delta:>15.8f}"
    f"{'---':>15}"
    f"{'---':>15}"
)

print(
    f"{'Finite Difference':<25}"
    f"{fd_central:>15.8f}"
    f"{fd_error:>15.8f}"
    f"{fd_runtime:>15.6f}"
)

print(
    f"{'Pathwise':<25}"
    f"{pathwise:>15.8f}"
    f"{pathwise_error:>15.8f}"
    f"{pathwise_runtime:>15.6f}"
)

print(
    f"{'Algorithmic Diff.':<25}"
    f"{ad:>15.8f}"
    f"{ad_error:>15.8f}"
    f"{ad_runtime:>15.6f}"
)

print("=" * 70)
print()

print("Forward FD Delta: ", fd_forward)
print("Backward FD Delta:", fd_backward)
print()