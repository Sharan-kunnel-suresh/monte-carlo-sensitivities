import numpy as np
import jax
import jax.numpy as jnp

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


def pathwise_delta(S,K,r,sigma,T,n_paths=100_000,seed=42,):
    """
    Estimate call Delta using the pathwise method.
    """

    rng = np.random.default_rng(seed)

    Z = rng.standard_normal(n_paths)

    ST = simulate_terminal_prices_from_z(
        S,
        r,
        sigma,
        T,
        Z,
    )

    # Indicator: 1 if the option finishes in-the-money, 0 otherwise
    indicator = (ST > K).astype(float)

    # Pathwise derivative:
    # dH/dST * dST/dS
    pathwise_delta_values = indicator * (ST / S)

    discount = np.exp(-r * T)

    delta_estimate = (
        discount
        * np.mean(pathwise_delta_values)
    )

    return delta_estimate

def monte_carlo_price_jax(S,K,r,sigma,T,Z,):
    """
    Monte Carlo European call price using JAX.

    Z is supplied externally so that the random numbers
    remain fixed during differentiation.
    """

    drift = (r - 0.5 * sigma**2) * T
    diffusion = sigma * jnp.sqrt(T) * Z

    ST = S * jnp.exp(drift + diffusion)

    payoff = jnp.maximum(ST - K, 0.0)

    discount = jnp.exp(-r * T)

    price = discount * jnp.mean(payoff)

    return price