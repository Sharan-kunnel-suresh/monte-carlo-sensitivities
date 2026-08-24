import numpy as np
#1. SET PARAMETERS
s0 = 100  # initial stock price
K = 100   # strike price
r=0.05  # risk-free interest rate
sigma = 0.2  # volatility
T= 1.0
N=10000000
black_scholes_price = 10.4506
black_scholes_delta = 0.6368

#2 generating norm 10000 random numbers
Z= np.random.randn(N)

# 3. Calculate ST for every path (Geometric Brownian Motion)
drift = (r-0.5*sigma**2)*T
diffusion = sigma*np.sqrt(T)*Z
ST= s0*np.exp(drift + diffusion)

#calculate call payoff
payoff = np.maximum(ST-K,0)

#5 discounted_payoff = np.exp(-r*T)*payoff
price_MC = np.exp(-r*T)*np.mean(payoff)

# standard error 
std_error = np.exp(-r*T)*np.std(payoff,ddof=1)/np.sqrt(N)

#pathwise delta
#derivative of payoff wrt s0 is ST/S0 
pathwise_derivative = np.where(ST > K, ST / s0, 0)
delta_MC = np.exp(-r * T) * np.mean(pathwise_derivative)

price_error = price_MC - black_scholes_price
delta_error = delta_MC - black_scholes_delta
# 8. Print everything
print("Monte Carlo Call Price:", price_MC)
print("Black-Scholes Call Price:", black_scholes_price)
print("Price Error:", price_error, "\n","\n")
print("Standard Error:", std_error, "\n","\n")
print("Monte Carlo Pathwise Delta:", delta_MC)
print("Black-Scholes Delta:", black_scholes_delta)
print("Delta Error:", delta_error)