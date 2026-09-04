import numpy as np


def monte_carlo_call( 
    S,
    K,
    r,
    sigma,
    T,
    n_paths,
    seed=None,
):
    """
    Price a European call option using Monte Carlo simulation.
    """

    ST = simulate_terminal_prices(
        S=S,
        r=r,
        sigma=sigma,
        T=T,
        n_paths=n_paths,
        seed=seed,
    )

    payoff = np.maximum(ST - K, 0)

    discount_factor = np.exp(-r * T)

    price = discount_factor * np.mean(payoff)

    standard_error = (
        discount_factor
        * np.std(payoff, ddof=1)
        / np.sqrt(n_paths)
    )

    return price, standard_error



def simulate_terminal_prices(S, r, sigma, T, n_paths, seed=None):
    """
    Simulate terminal stock prices using geometric Brownian motion.
    """
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(n_paths)
    drift = (r - 0.5 * sigma**2) * T
    diffusion = sigma * np.sqrt(T) * Z
    ST = S * np.exp(drift + diffusion)
    return ST