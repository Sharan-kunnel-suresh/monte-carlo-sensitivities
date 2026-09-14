import numpy as np
from .monte_carlo import simulate_terminal_prices_from_z


def finite_difference_delta(
    S,
    K,
    r,
    sigma,
    T,
    h=1e-4,
    n_paths=100_000,
    seed=42,
):
    """
    Estimate Delta using finite differences with common random numbers.
    """

    if S - h <= 0:
        raise ValueError("S - h must be positive for GBM simulation.")

    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(n_paths)

    # Terminal prices for S+h, S, S-h
    ST_up = simulate_terminal_prices_from_z(S + h, r, sigma, T, Z)
    ST_mid = simulate_terminal_prices_from_z(S,     r, sigma, T, Z)
    ST_down = simulate_terminal_prices_from_z(S - h, r, sigma, T, Z)

    # Payoffs
    payoff_up = np.maximum(ST_up - K, 0)
    payoff_mid = np.maximum(ST_mid - K, 0)
    payoff_down = np.maximum(ST_down - K, 0)

    discount = np.exp(-r * T)

    price_up = discount * np.mean(payoff_up)
    price_mid = discount * np.mean(payoff_mid)
    price_down = discount * np.mean(payoff_down)

    return {
        "central": (price_up - price_down) / (2 * h),
        "forward": (price_up - price_mid) / h,
        "backward": (price_mid - price_down) / h,
    }
