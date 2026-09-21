import numpy as np

from src.mc_sensitivities.sensitivities import ( pathwise_delta,ad_delta,finite_difference_delta)


S = 100
K = 100
r = 0.05
sigma = 0.20
T = 1.0


def test_pathwise_delta_is_close_to_black_scholes():
    delta = pathwise_delta(
        S=S,
        K=K,
        r=r,
        sigma=sigma,
        T=T,
        n_paths=100_000,
        seed=42,
    )

    black_scholes_delta = 0.6368306512

    assert np.isclose(
        delta,
        black_scholes_delta,
        atol=4e-3,
    )