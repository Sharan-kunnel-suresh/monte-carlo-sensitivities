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
  <img src="https://img.shields.io/badge/JAX-Automatic%20Differentiation-orange" alt="JAX">
  <img src="https://img.shields.io/badge/Domain-Quantitative%20Finance-green" alt="Quant Finance">
  <img src="https://img.shields.io/badge/Method-Monte%20Carlo-purple" alt="Monte Carlo">
  <img src="https://img.shields.io/badge/License-MIT-yellow" alt="License">
</p>

---

## Overview

This project investigates how **algorithmic differentiation (AD)** can be used to estimate the sensitivities, or **Greeks**, of financial derivatives within a Monte Carlo pricing framework.

The project starts from the mathematical foundations of the Black-Scholes model and progressively develops a Monte Carlo pricing framework before comparing different approaches to sensitivity estimation.

The methods investigated include:

- Analytical differentiation
- Finite-difference methods
- Pathwise differentiation
- Likelihood-ratio methods
- Algorithmic differentiation

The goal is not simply to implement these methods, but to understand and experimentally compare their:

- **Accuracy**
- **Convergence**
- **Numerical stability**
- **Computational cost**
- **Sensitivity to Monte Carlo sampling**
- **Limitations for non-smooth payoffs**

The final result will be a reproducible framework for studying how different techniques estimate option sensitivities under Monte Carlo simulation.

---

## Research Question

> **How effectively can algorithmic differentiation estimate option sensitivities within a Monte Carlo framework compared with conventional sensitivity-estimation methods?**

The project investigates this question through mathematical derivations and computational experiments.

In particular, the experiments examine:

- Monte Carlo pricing accuracy
- Greek estimation accuracy
- Standard error and convergence
- Finite-difference step-size sensitivity
- Numerical stability
- Computational performance
- Higher-order derivatives
- Non-smooth payoff functions
- Variance-reduction techniques

---

# Objectives

## Primary Objectives

- Implement analytical Black-Scholes pricing and Greeks.
- Develop a Monte Carlo pricing engine for European options.
- Investigate Monte Carlo convergence and statistical error.
- Implement finite-difference sensitivity estimators.
- Implement pathwise sensitivity estimation.
- Implement algorithmic differentiation using JAX.
- Investigate first- and second-order sensitivities.
- Compare Monte Carlo estimates with analytical Black-Scholes benchmarks.
- Measure computational performance across different methods.

## Secondary Objectives

- Investigate variance-reduction techniques.
- Study the effect of simulation size on sensitivity estimates.
- Analyze numerical error and stability.
- Produce reproducible experiments and visualizations.
- Document the mathematical derivations behind each implementation.

---

# Mathematical Foundation

## 1. Risk-Neutral Pricing

Under the risk-neutral measure, the value of a European derivative can be written as

$$
V_0
=
e^{-rT}
\mathbb{E}^{\mathbb{Q}}
\left[
H(S_T)
\right]
$$

where:

- $V_0$ is the current derivative value
- $r$ is the continuously compounded risk-free interest rate
- $T$ is the time to maturity
- $S_T$ is the underlying asset price at maturity
- $H(S_T)$ is the derivative payoff
- $\mathbb{Q}$ is the risk-neutral probability measure

For a European call option:

$$
H(S_T)
=
\max(S_T-K,0)
$$

where $K$ is the strike price.

Monte Carlo simulation estimates the expectation by generating many possible realizations of the terminal asset price.

---

# 2. Geometric Brownian Motion

The underlying asset is modeled using the Black-Scholes stochastic differential equation:

$$
dS_t
=
rS_t\,dt
+
\sigma S_t\,dW_t
$$

where:

- $S_t$ is the asset price at time $t$
- $r$ is the risk-free interest rate
- $\sigma$ is the volatility
- $W_t$ is a standard Brownian motion

For constant $r$ and $\sigma$, the exact solution at maturity is:

$$
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
$$

where

$$
Z\sim N(0,1).
$$

This closed-form representation allows the terminal asset price to be simulated directly without discretizing the stochastic differential equation.

---

# 3. Monte Carlo Pricing

Suppose we generate $N$ independent terminal prices:

$$
S_T^{(1)},
S_T^{(2)},
\ldots,
S_T^{(N)}.
$$

For each simulation, the corresponding payoff is:

$$
H_i
=
H\left(S_T^{(i)}\right).
$$

The Monte Carlo estimator of the option price is:

$$
\hat{V}_N
=
e^{-rT}
\frac{1}{N}
\sum_{i=1}^{N}
H_i.
$$

Equivalently,

$$
\hat{V}_N
=
e^{-rT}
\frac{1}{N}
\sum_{i=1}^{N}
H\left(S_T^{(i)}\right).
$$

### Convergence

By the Law of Large Numbers,

$$
\hat{V}_N
\rightarrow
V_0
\qquad
\text{as }N\rightarrow\infty.
$$

However, Monte Carlo convergence is relatively slow.

The standard error behaves approximately as:

$$
SE(\hat{V}_N)
=
\frac{e^{-rT}\sigma_H}{\sqrt{N}}
$$

where $\sigma_H$ is the standard deviation of the simulated payoff.

Therefore,

$$
SE(\hat{V}_N)
\propto
\frac{1}{\sqrt{N}}.
$$

This means that increasing the number of simulations by a factor of $100$ reduces the standard error by approximately a factor of $10$.

This relationship is one of the fundamental characteristics of Monte Carlo methods and will be investigated experimentally in this project.

---

# 4. Black-Scholes Analytical Benchmark

The Monte Carlo estimates are compared against the analytical Black-Scholes solution.

For a European call option:

$$
C
=
S_0N(d_1)
-
Ke^{-rT}N(d_2)
$$

where

$$
d_1
=
\frac{
\ln(S_0/K)
+
\left(r+\frac{1}{2}\sigma^2\right)T
}{
\sigma\sqrt{T}
}
$$

and

$$
d_2
=
d_1-\sigma\sqrt{T}.
$$

Here, $N(\cdot)$ denotes the cumulative distribution function of the standard normal distribution.

The analytical solution provides a **ground truth** against which the Monte Carlo and sensitivity estimators can be evaluated.

---

# 5. Option Sensitivities

The project focuses on the main option Greeks.

## Delta

Delta measures the sensitivity of the option value to the underlying asset price:

$$
\Delta
=
\frac{\partial V}{\partial S_0}.
$$

For a Black-Scholes European call:

$$
\Delta
=
N(d_1).
$$

---

## Gamma

Gamma measures the curvature of the option value with respect to the underlying:

$$
\Gamma
=
\frac{\partial^2 V}{\partial S_0^2}.
$$

For a European call:

$$
\Gamma
=
\frac{\phi(d_1)}
{S_0\sigma\sqrt{T}}
$$

where $\phi(\cdot)$ is the standard normal probability density function.

---

## Vega

Vega measures sensitivity to volatility:

$$
\nu
=
\frac{\partial V}{\partial \sigma}.
$$

For a European call:

$$
\nu
=
S_0\phi(d_1)\sqrt{T}.
$$

---

## Rho

Rho measures sensitivity to the risk-free interest rate:

$$
\rho
=
\frac{\partial V}{\partial r}.
$$

For a European call:

$$
\rho
=
KT e^{-rT}N(d_2).
$$

---

## Theta

In this project, Theta is defined as the derivative with respect to **time remaining to maturity**:

$$
\Theta
=
\frac{\partial V}{\partial T}.
$$

This convention should be distinguished from the common market convention where Theta is often expressed as the change in option value with respect to the passage of calendar time.

---

# 6. Sensitivity Estimation Methods

A central objective of the project is to investigate how derivatives of a Monte Carlo pricing estimator can be calculated.

## Finite Differences

For Delta, a central finite-difference approximation is:

$$
\Delta
\approx
\frac{
V(S_0+h)-V(S_0-h)
}{
2h
}
$$

where $h$ is a small perturbation of the initial asset price.

Finite differences are simple to implement, but the choice of $h$ introduces a numerical trade-off.

If $h$ is too large, the approximation suffers from **truncation error**.

If $h$ is too small, **floating-point error and Monte Carlo noise** can dominate the result.

Therefore, the project will investigate how the estimated Delta changes as $h$ varies.

---

## Pathwise Differentiation

If differentiation can be moved through the expectation, Delta can be expressed as:

$$
\frac{\partial V}{\partial S_0}
=
e^{-rT}
\mathbb{E}
\left[
\frac{\partial H(S_T)}{\partial S_T}
\frac{\partial S_T}{\partial S_0}
\right].
$$

From the GBM solution,

$$
S_T
=
S_0e^X
$$

where $X$ does not depend on $S_0$.

Therefore:

$$
\frac{\partial S_T}{\partial S_0}
=
\frac{S_T}{S_0}.
$$

For a European call, away from the strike:

$$
\frac{\partial H(S_T)}{\partial S_T}
=
\mathbf{1}_{\{S_T>K\}}.
$$

Consequently, the pathwise Delta estimator becomes:

$$
\Delta
=
e^{-rT}
\mathbb{E}
\left[
\mathbf{1}_{\{S_T>K\}}
\frac{S_T}{S_0}
\right].
$$

The corresponding Monte Carlo estimator is:

$$
\hat{\Delta}_N
=
e^{-rT}
\frac{1}{N}
\sum_{i=1}^{N}
\mathbf{1}_{\{S_T^{(i)}>K\}}
\frac{S_T^{(i)}}{S_0}.
$$

This provides an important benchmark for the automatic-differentiation implementation.

---

# 7. Algorithmic Differentiation

Algorithmic differentiation, or **automatic differentiation (AD)**, computes derivatives by applying the chain rule to the elementary operations that form a computational program.

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
                   │
                   ▼
              ∂Price/∂S₀
