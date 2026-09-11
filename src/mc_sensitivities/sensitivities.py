import numpy as np

from mc_sensitivities.monte_carlo import monte_carlo_call, simulate_terminal_prices_from_z


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
    Estimate Delta using finite differences.

    Uses common random numbers so that the same Monte Carlo
    samples are used for S+h, S, and S-h.
    """

    rng = np.random.default_rng(seed)

    Z = rng.standard_normal(n_paths)

    ST_up = simulate_terminal_prices_from_z(
        S=S + h,
        r=r,
        sigma=sigma,
        T=T,
        Z=Z,
    )

    ST = simulate_terminal_prices_from_z(
        S=S,
        r=r,
        sigma=sigma,
        T=T,
        Z=Z,
    )

    ST_down = simulate_terminal_prices_from_z(
        S=S - h,
        r=r,
        sigma=sigma,
        T=T,
        Z=Z,
    )

    payoff_up = np.maximum(ST_up - K, 0)
    payoff = np.maximum(ST - K, 0)
    payoff_down = np.maximum(ST_down - K, 0)

    discount_factor = np.exp(-r * T)

    price_up = discount_factor * np.mean(payoff_up)
    price = discount_factor * np.mean(payoff)
    price_down = discount_factor * np.mean(payoff_down)

    central_delta = (
        price_up - price_down
    ) / (2 * h)

    forward_delta = (
        price_up - price
    ) / h

    backward_delta = (
        price - price_down
    ) / h

    return (
        central_delta,
        forward_delta,
        backward_delta,
    )