import numpy as np


def simulate_terminal_prices(
    S,
    r,
    sigma,
    T,
    n_paths,
    seed=None,
):
    """
    Simulate terminal stock prices under geometric Brownian motion.

    Parameters
    ----------
    S : float
        Initial stock price.
    r : float
        Continuously compounded risk-free interest rate.
    sigma : float
        Volatility.
    T : float
        Time to maturity.
    n_paths : int
        Number of Monte Carlo simulations.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    numpy.ndarray
        Simulated terminal stock prices.
    """
    rng = np.random.default_rng(seed)

    Z = rng.standard_normal(n_paths)

    drift = (r - 0.5 * sigma**2) * T
    diffusion = sigma * np.sqrt(T) * Z

    ST = S * np.exp(drift + diffusion)

    return ST


def simulate_terminal_prices_from_z(
    S,
    r,
    sigma,
    T,
    Z,
):
    """
    Simulate terminal stock prices using externally supplied
    standard-normal random variables.

    This allows the same random numbers to be reused across
    different parameter values, which is useful for common
    random-number finite-difference estimators.

    Parameters
    ----------
    S : float
        Initial stock price.
    r : float
        Continuously compounded risk-free interest rate.
    sigma : float
        Volatility.
    T : float
        Time to maturity.
    Z : numpy.ndarray
        Standard-normal random variables.

    Returns
    -------
    numpy.ndarray
        Simulated terminal stock prices.
    """
    drift = (r - 0.5 * sigma**2) * T
    diffusion = sigma * np.sqrt(T) * Z

    ST = S * np.exp(drift + diffusion)

    return ST


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
        Continuously compounded risk-free interest rate.
    sigma : float
        Volatility.
    T : float
        Time to maturity.
    n_paths : int
        Number of Monte Carlo simulations.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    tuple
        Monte Carlo option price and standard error.
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