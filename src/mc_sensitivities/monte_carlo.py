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

    Parameters
    ----------
    S : float
        Initial stock price.
    K : float
        Strike price.
    r : float
        Risk-free interest rate.
    sigma : float
        Volatility.
    T : float
        Time to maturity.
    n_paths : int
        Number of Monte Carlo paths.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    price : float
        Monte Carlo estimate of the option price.
    standard_error : float
        Standard error of the Monte Carlo estimate.
    """

    rng = np.random.default_rng(seed)

    Z = rng.standard_normal(n_paths)

    drift = (r - 0.5 * sigma**2) * T
    diffusion = sigma * np.sqrt(T) * Z

    ST = S * np.exp(drift + diffusion)

    payoff = np.maximum(ST - K, 0)

    discount_factor = np.exp(-r * T)

    price = discount_factor * np.mean(payoff)

    standard_error = (
        discount_factor
        * np.std(payoff, ddof=1)
        / np.sqrt(n_paths)
    )

    return price, standard_error