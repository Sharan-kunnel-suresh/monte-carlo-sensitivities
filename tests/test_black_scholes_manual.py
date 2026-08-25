from src.mc_sensitivities.black_scholes import (
    black_scholes_call,
    black_scholes_call_theta,
    black_scholes_put,
    black_scholes_call_delta,
    black_scholes_call_gamma,
    black_scholes_call_vega,
    black_scholes_call_rho
)

S = 100
K = 100
r = 0.05
sigma = 0.20
T = 1.0

call = black_scholes_call(S, K, r, sigma, T)
put = black_scholes_put(S, K, r, sigma, T)
delta = black_scholes_call_delta(S, K, r, sigma, T)
theta = black_scholes_call_theta(S, K, r, sigma, T)
gamma=black_scholes_call_gamma(S,K,r,sigma,T)
vega=black_scholes_call_vega(S,K,r,sigma,T)
rho=black_scholes_call_rho(S,K,r,sigma,T)

print("Call:", call)
print("Put:", put)
print("Call Delta:", delta)
print("Call Theta:", theta)
print("Call Gamma:", gamma)
print("Call Vega:", vega)
print("Call Rho:", rho)