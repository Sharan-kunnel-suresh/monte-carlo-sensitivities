import numpy as np

from src.mc_sensitivities.monte_carlo import monte_carlo_call


S = 100
K = 100
r = 0.05
sigma = 0.20
T = 1.0


def test_monte_carlo_price_is_close_to_black_scholes():

    price, standard_error = monte_carlo_call(
        S=S,
        K=K,
        r=r,
        sigma=sigma,
        T=T,
        n_paths=100_000,
        seed=42,
    )

    black_scholes_price = 10.45058357

    # The Monte Carlo estimate is random,
    # so we don't expect exact equality.
    assert abs(price - black_scholes_price) < 4 * standard_error