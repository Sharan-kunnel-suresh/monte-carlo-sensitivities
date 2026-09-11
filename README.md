# Monte Carlo Sensitivities Using Algorithmic Differentiation

<p align="center">
  <strong>Monte Carlo Pricing & Sensitivity Analysis for Derivative Securities</strong>
</p>

<p align="center">
  A quantitative finance project investigating algorithmic differentiation
  for estimating option sensitivities within Monte Carlo simulations.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/NumPy-Numerical%20Computing-blue" alt="NumPy">
  <img src="https://img.shields.io/badge/SciPy-Scientific%20Computing-orange" alt="SciPy">
  <img src="https://img.shields.io/badge/JAX-Automatic%20Differentiation-purple" alt="JAX">
  <img src="https://img.shields.io/badge/Pytest-Testing-green" alt="Pytest">
  <img src="https://img.shields.io/badge/Domain-Quantitative%20Finance-darkgreen" alt="Quantitative Finance">
  <img src="https://img.shields.io/badge/License-MIT-yellow" alt="License">
</p>

---

## Overview

This project investigates how **algorithmic differentiation (AD)** can be used to estimate the sensitivities, or **Greeks**, of financial derivatives within a Monte Carlo pricing framework.

The project starts from the mathematical foundations of the Black-Scholes model and progressively develops a Monte Carlo pricing framework before comparing different approaches to sensitivity estimation.

The main methods investigated are:

- Analytical differentiation
- Finite-difference methods
- Pathwise differentiation
- Likelihood-ratio methods
- Algorithmic differentiation

The objective is not simply to implement these techniques, but to study their:

- **Accuracy**
- **Convergence**
- **Numerical stability**
- **Computational cost**
- **Sensitivity to Monte Carlo sampling**
- **Behavior for non-smooth payoffs**

The project combines mathematical derivations with reproducible computational experiments.

---

# Research Question

> **How effectively can algorithmic differentiation estimate option sensitivities within a Monte Carlo framework compared with conventional sensitivity-estimation methods?**

The investigation focuses on the relationship between mathematical differentiation, stochastic simulation, and numerical computation.

The main questions are:

1. How accurately can Monte Carlo simulation reproduce analytical option prices?
2. How does Monte Carlo sampling error affect sensitivity estimates?
3. How does finite-difference step size affect Greek estimation?
4. How does pathwise differentiation compare with finite differences?
5. Can algorithmic differentiation provide accurate sensitivities without requiring repeated pricing evaluations?
6. How does computational cost scale as the number of sensitivities increases?
7. What limitations arise when differentiating non-smooth financial payoffs?

---

# Objectives

## Primary Objectives

- Implement analytical Black-Scholes pricing.
- Implement analytical Black-Scholes Greeks.
- Develop a Monte Carlo pricing engine for European options.
- Study Monte Carlo convergence and statistical error.
- Implement finite-difference sensitivity estimators.
- Implement pathwise sensitivity estimation.
- Implement algorithmic differentiation using JAX.
- Investigate first- and second-order sensitivities.
- Compare numerical estimates against analytical benchmarks.
- Measure computational performance.

## Secondary Objectives

- Investigate variance-reduction techniques.
- Study the effect of simulation size on sensitivity estimates.
- Analyze numerical stability.
- Visualize convergence and estimation error.
- Document the mathematical foundations behind each method.
- Build a reproducible quantitative-finance research framework.

---

# Mathematical Foundation

## 1. Risk-Neutral Pricing

Under the risk-neutral measure, the value of a European derivative is given by

```math
V_0
=
e^{-rT}
\mathbb{E}^{\mathbb{Q}}
\left[
H(S_T)
\right]
```

where:

- $V_0$ is the current derivative value
- $r$ is the continuously compounded risk-free interest rate
- $T$ is the time to maturity
- $S_T$ is the underlying asset price at maturity
- $H(S_T)$ is the derivative payoff
- $\mathbb{Q}$ is the risk-neutral probability measure

For a European call option:

```math
H(S_T)
=
\max(S_T-K,0)
```

where $K$ is the strike price.

Monte Carlo simulation estimates the expectation by generating a large number of possible terminal asset prices.

---

# 2. Geometric Brownian Motion

The underlying asset is initially modeled using the Black-Scholes stochastic differential equation:

\[
dS_t = rS_t\,dt + \sigma S_t\,dW_t
\]

where:

- $S_t$ is the asset price at time $t$
- $r$ is the risk-free interest rate
- $\sigma$ is the volatility
- $W_t$ is a standard Brownian motion

For constant $r$ and $\sigma$, the exact terminal solution is:

```math
S_T
=
S_0
\exp
\left[
\left(
r-\frac{1}{2}\sigma^2
\right)T
+
\sigma\sqrt{T}Z
\right]
```

where

```math
Z \sim N(0,1)
```

This exact solution allows terminal prices to be simulated directly without discretizing the stochastic differential equation.

---

# 3. Monte Carlo Pricing

Suppose $N$ independent terminal prices are simulated:

```math
S_T^{(1)},
S_T^{(2)},
\ldots,
S_T^{(N)}
```

For each simulation, the corresponding payoff is:

```math
H_i
=
H\left(S_T^{(i)}\right)
```

The Monte Carlo estimator of the option price is:

```math
\hat{V}_N
=
e^{-rT}
\frac{1}{N}
\sum_{i=1}^{N}
H\left(S_T^{(i)}\right)
```

### Convergence

By the Law of Large Numbers:

```math
\hat{V}_N
\rightarrow
V_0
\qquad
\text{as } N\rightarrow\infty
```

However, Monte Carlo convergence is relatively slow.

The standard error of the estimator is:

```math
SE(\hat{V}_N)
=
\frac{e^{-rT}\sigma_H}{\sqrt{N}}
```

where $\sigma_H$ is the standard deviation of the simulated payoff.

Therefore:

```math
SE(\hat{V}_N)
\propto
\frac{1}{\sqrt{N}}
```

This means that increasing the number of simulations by a factor of $100$ reduces the standard error by approximately a factor of $10$.

This fundamental property of Monte Carlo simulation is investigated experimentally in this project.

---

# 4. Black-Scholes Analytical Benchmark

The Monte Carlo estimates are compared against the analytical Black-Scholes solution.

For a European call option:

```math
C
=
S_0N(d_1)
-
Ke^{-rT}N(d_2)
```

where

```math
d_1
=
\frac{
\ln(S_0/K)
+
\left(r+\frac{1}{2}\sigma^2\right)T
}{
\sigma\sqrt{T}
}
```

and

```math
d_2
=
d_1-\sigma\sqrt{T}
```

Here, $N(\cdot)$ denotes the cumulative distribution function of the standard normal distribution.

The analytical Black-Scholes solution acts as a **benchmark** for evaluating Monte Carlo prices and sensitivity estimates.

---

# 5. Option Sensitivities

The project focuses on the principal option Greeks.

## Delta

Delta measures sensitivity to the underlying asset price:

```math
\Delta
=
\frac{\partial V}{\partial S_0}
```

For a European call under Black-Scholes:

```math
\Delta
=
N(d_1)
```

---

## Gamma

Gamma measures the curvature of the option value with respect to the underlying:

```math
\Gamma
=
\frac{\partial^2 V}{\partial S_0^2}
```

For a European call:

```math
\Gamma
=
\frac{\phi(d_1)}
{S_0\sigma\sqrt{T}}
```

where $\phi(\cdot)$ is the standard normal probability density function.

---

## Vega

Vega measures sensitivity to volatility:

```math
\nu
=
\frac{\partial V}{\partial \sigma}
```

For a European call:

```math
\nu
=
S_0\phi(d_1)\sqrt{T}
```

The implementation uses vega as the derivative with respect to $\sigma$ itself, rather than per one-percentage-point volatility change.

---

## Rho

Rho measures sensitivity to the risk-free interest rate:

```math
\rho
=
\frac{\partial V}{\partial r}
```

For a European call:

```math
\rho
=
KT e^{-rT}N(d_2)
```

---

## Theta

In this project, Theta is defined as the derivative with respect to **time remaining to maturity**:

```math
\Theta
=
\frac{\partial V}{\partial T}
```

This convention differs from the common market convention where Theta is often reported as the change in option value due to the passage of calendar time.

---

# 6. Sensitivity Estimation

A central objective of this project is to investigate how derivatives of a Monte Carlo pricing estimator can be calculated.

## Finite Differences

For Delta, a central finite-difference approximation is:

```math
\Delta
\approx
\frac{
V(S_0+h)-V(S_0-h)
}{
2h
}
```

where $h$ is a small perturbation of the initial asset price.

Finite differences are straightforward to implement, but the choice of $h$ introduces a trade-off between different sources of numerical error.

If $h$ is too large, the approximation suffers from **truncation error**.

If $h$ is too small, **floating-point error and Monte Carlo sampling noise** can dominate the estimate.

The project therefore investigates how the estimated sensitivity changes as $h$ varies.

---

# 7. Pathwise Differentiation

For suitable payoff functions, differentiation can be moved through the expectation:

```math
\frac{\partial V}{\partial S_0}
=
e^{-rT}
\mathbb{E}
\left[
\frac{\partial H(S_T)}{\partial S_T}
\frac{\partial S_T}{\partial S_0}
\right]
```

From the GBM solution:

```math
S_T
=
S_0e^X
```

where $X$ does not depend on $S_0$.

Therefore:

```math
\frac{\partial S_T}{\partial S_0}
=
\frac{S_T}{S_0}
```

For a European call, away from the strike:

```math
\frac{\partial H(S_T)}{\partial S_T}
=
\mathbf{1}_{\{S_T>K\}}
```

Consequently, the pathwise Delta estimator is:

```math
\Delta
=
e^{-rT}
\mathbb{E}
\left[
\mathbf{1}_{\{S_T>K\}}
\frac{S_T}{S_0}
\right]
```

and its Monte Carlo estimator is:

```math
\hat{\Delta}_N
=
e^{-rT}
\frac{1}{N}
\sum_{i=1}^{N}
\mathbf{1}_{\{S_T^{(i)}>K\}}
\frac{S_T^{(i)}}{S_0}
```

This provides an important benchmark for the algorithmic-differentiation implementation.

---

# 8. Algorithmic Differentiation

**Algorithmic differentiation (AD)** computes derivatives by systematically applying the chain rule to the elementary operations that make up a computational program.

The Monte Carlo pricing calculation can be represented as a computational graph:

```text
                  S₀
                   │
                   ▼
          ┌─────────────────┐
          │   GBM Formula   │
          └────────┬────────┘
                   │
                   ▼
                  S_T
                   │
                   ▼
          ┌─────────────────┐
          │     Payoff      │
          │  max(S_T - K,0) │
          └────────┬────────┘
                   │
                   ▼
          Discounted Payoff
                   │
                   ▼
          Monte Carlo Mean
                  │
                  ▼
               Price
```
