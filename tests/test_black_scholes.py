import numpy as np

from src.mc_sensitivities.black_scholes import (
    black_scholes_call,
    black_scholes_put,
    black_scholes_call_delta,
    black_scholes_call_gamma,
    black_scholes_call_vega,
    black_scholes_call_rho,
    black_scholes_call_theta,
)


S = 100
K = 100
r = 0.05
sigma = 0.20
T = 1.0


def test_call_price():
    result = black_scholes_call(S, K, r, sigma, T)

    assert np.isclose(result, 10.45058357, atol=1e-6)


def test_put_price():
    result = black_scholes_put(S, K, r, sigma, T)

    assert np.isclose(result, 5.57352602, atol=1e-6)


def test_call_delta():
    result = black_scholes_call_delta(S, K, r, sigma, T)

    assert np.isclose(result, 0.63683065, atol=1e-6)


def test_call_gamma():
    result = black_scholes_call_gamma(S, K, r, sigma, T)

    assert np.isclose(result, 0.01876202, atol=1e-6)


def test_call_vega():
    result = black_scholes_call_vega(S, K, r, sigma, T)

    assert np.isclose(result, 37.52403148, atol=1e-6)


def test_call_rho():
    result = black_scholes_call_rho(S, K, r, sigma, T)

    assert np.isclose(result, 53.23248155, atol=1e-6)


def test_call_theta():
    result = black_scholes_call_theta(S, K, r, sigma, T)

    assert np.isclose(result, -6.41402755, atol=1e-6)

def test_put_call_parity():
    call = black_scholes_call(S, K, r, sigma, T)
    put = black_scholes_put(S, K, r, sigma, T)

    lhs = call - put
    rhs = S - K * np.exp(-r * T)

    assert np.isclose(lhs, rhs, atol=1e-6)
