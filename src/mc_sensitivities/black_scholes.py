import numpy as np
from scipy.stats import norm


def _d1_d2(S, K, r, sigma, T):
    d1 = (
        np.log(S / K)
        + (r + 0.5 * sigma**2) * T
    ) / (sigma * np.sqrt(T))

    d2 = d1 - sigma * np.sqrt(T)

    return d1, d2

def black_scholes_call(S, K, r, sigma, T):
    """
    Black-Scholes price of a European call option.
    """

    d1, d2 = _d1_d2(S, K, r, sigma, T)

    return (
        S * norm.cdf(d1)
        - K * np.exp(-r * T) * norm.cdf(d2)
    )


def black_scholes_put(S, K, r, sigma, T):
    """
    Black-Scholes price of a European put option.
    """

    d1, d2 = _d1_d2(S, K, r, sigma, T)

    return (
        K * np.exp(-r * T) * norm.cdf(-d2)
        - S * norm.cdf(-d1)
    )


def black_scholes_call_delta(S, K, r, sigma, T):
    """
    Black-Scholes delta of a European call option.
    """

    d1, _ = _d1_d2(S, K, r, sigma, T)

    return norm.cdf(d1)

def black_scholes_call_gamma(S, K, r, sigma, T):
    """
    Black-Scholes gamma of a European call option.
    """

    d1, _ = _d1_d2(S, K, r, sigma, T)

    return norm.pdf(d1) / (S * sigma * np.sqrt(T))


def black_scholes_call_vega(S, K, r, sigma, T):
    """
    Black-Scholes vega of a European call option.

    Returns the derivative with respect to sigma,
    not vega per 1 percentage-point change.
    """

    d1, _ = _d1_d2(S, K, r, sigma, T)

    return S * norm.pdf(d1) * np.sqrt(T)


def black_scholes_call_rho(S, K, r, sigma, T):
    """
    Black-Scholes rho of a European call option.
    """

    _, d2 = _d1_d2(S, K, r, sigma, T)

    return K * T * np.exp(-r * T) * norm.cdf(d2)


def black_scholes_call_theta(S, K, r, sigma, T):
    """
    Black-Scholes theta as dV/dT,
    where T is time remaining to maturity.
    """

    d1, d2 = _d1_d2(S, K, r, sigma, T)

    first_term = (
        -S * norm.pdf(d1) * sigma
        / (2 * np.sqrt(T))
    )

    second_term = (
        -r * K * np.exp(-r * T) * norm.cdf(d2)
    )

    return first_term + second_term