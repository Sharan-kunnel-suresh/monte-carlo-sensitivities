# Monte Carlo Sensitivities Using Algorithmic Differentiation

<p align="center">
  <strong>Monte Carlo Pricing & Sensitivity Analysis for Derivative Securities</strong>
</p>

<p align="center">
  A quantitative finance project investigating the use of algorithmic differentiation
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

This project investigates how **algorithmic differentiation (AD)** can be used to calculate the sensitivities, or **Greeks**, of financial derivatives within a Monte Carlo pricing framework.

The project begins with the classical Black-Scholes model and progressively develops a Monte Carlo pricing engine before comparing several approaches to sensitivity estimation:

- Finite-difference methods
- Pathwise differentiation
- Likelihood-ratio methods
- Algorithmic differentiation

The objective is not only to implement these methods, but to study their **mathematical foundations, numerical accuracy, convergence behavior, computational performance, and limitations**.

The final implementation will provide a reproducible framework for evaluating how different sensitivity-estimation techniques behave under Monte Carlo simulation.

---

## Research Question

> **How effectively can algorithmic differentiation estimate Monte Carlo sensitivities compared with traditional numerical differentiation methods?**

The project investigates this question through both mathematical analysis and computational experiments.

In particular, we examine:

- Accuracy of estimated Greeks
- Monte Carlo convergence
- Standard error
- Numerical stability
- Computational cost
- Finite-difference step-size sensitivity
- Higher-order sensitivities
- Effects of payoff discontinuities
- Variance-reduction techniques

---

## Objectives

### Primary Objectives

- Implement analytical Black-Scholes pricing and Greeks.
- Implement a Monte Carlo pricing engine for European options.
- Implement finite-difference sensitivity estimators.
- Implement pathwise sensitivity estimation.
- Implement algorithmic differentiation using JAX.
- Investigate first- and second-order derivatives.
- Compare numerical results against analytical benchmarks.
- Measure computational performance and convergence.

### Secondary Objectives

- Investigate Monte Carlo variance reduction.
- Study the effect of sample size on sensitivity estimates.
- Analyze numerical error and stability.
- Produce reproducible experiments and visualizations.
- Document the mathematical derivations behind the implementations.

---

# Mathematical Foundation

## 1. Risk-Neutral Pricing

Under the risk-neutral measure, the value of a European derivative can be represented as

\[
V_0 = e^{-rT}\mathbb{E}^{\mathbb{Q}}[H(S_T)]
\]

where:

- \(V_0\) is the current derivative value
- \(r\) is the continuously compounded risk-free rate
- \(T\) is the time to maturity
- \(S_T\) is the underlying price at maturity
- \(H(S_T)\) is the derivative payoff
- \(\mathbb{Q}\) is the risk-neutral probability measure

For a European call option:

\[
H(S_T) = \max(S_T-K,0)
\]

where \(K\) is the strike price.

Monte Carlo methods estimate the expectation numerically by generating many realizations of \(S_T\).

---

# 2. Geometric Brownian Motion

The underlying asset is initially modeled using the Black-Scholes stochastic differential equation:

\[
dS_t = rS_t\,dt + \sigma S_t\,dW_t
\]

where:

- \(S_t\) is the asset price
- \(r\) is the risk-free rate
- \(\sigma\) is the volatility
- \(W_t\) is a Brownian motion

The exact terminal solution is:

\[
S_T =
S_0
\exp
\left[
\left(r-\frac{1}{2}\sigma^2\right)T
+
\sigma\sqrt{T}Z
\right]
\]

with

\[
Z\sim N(0,1).
\]

This representation allows the terminal asset price to be simulated directly.

---

# 3. Monte Carlo Pricing

Given \(N\) simulated terminal prices:

\[
S_T^{(1)},S_T^{(2)},\ldots,S_T^{(N)}
\]

the Monte Carlo estimator is:

\[
\hat V_N =
e^{-rT}
\frac{1}{N}
\sum_{i=1}^{N}
H(S_T^{(i)}).
\]

As the number of simulations increases:

\[
\hat V_N \rightarrow V_0
\]

under the law of large numbers.

The Monte Carlo standard error behaves approximately as:

\[
SE(\hat V_N)
=
\frac{\sigma_H}{\sqrt{N}}
\]

where \(\sigma_H\) is the standard deviation of the discounted payoff.

---

# 4. Option Sensitivities

The project focuses on the main option Greeks.

### Delta

\[
\Delta =
\frac{\partial V}{\partial S_0}
\]

Measures sensitivity to the underlying asset price.

### Gamma

\[
\Gamma =
\frac{\partial^2 V}{\partial S_0^2}
\]

Measures the curvature of the option value with respect to the underlying.

### Vega

\[
\nu =
\frac{\partial V}{\partial \sigma}
\]

Measures sensitivity to volatility.

### Rho

\[
\rho =
\frac{\partial V}{\partial r}
\]

Measures sensitivity to the interest rate.

### Theta

\[
\Theta =
\frac{\partial V}{\partial T}
\]

Measures sensitivity to time to maturity.

---

# 5. Sensitivity Estimation

A central objective of this project is to investigate how derivatives of a Monte Carlo estimator can be calculated.

## Finite Differences

For Delta, a central finite-difference approximation is:

\[
\Delta
\approx
\frac{
V(S_0+h)-V(S_0-h)
}{
2h
}
\]

where \(h\) is a small perturbation.

Although straightforward, finite differences introduce a trade-off between:

- truncation error
- floating-point error
- Monte Carlo sampling noise

The project will investigate this behavior experimentally.

---

## Pathwise Differentiation

If the derivative can be moved through the expectation:

\[
\frac{\partial V}{\partial S_0}
=
e^{-rT}
\mathbb{E}
\left[
\frac{\partial H(S_T)}{\partial S_T}
\frac{\partial S_T}{\partial S_0}
\right].
\]

For the GBM terminal price:

\[
S_T=S_0e^X
\]

and therefore:

\[
\frac{\partial S_T}{\partial S_0}
=
\frac{S_T}{S_0}.
\]

For a European call, away from the payoff discontinuity:

\[
\frac{\partial H}{\partial S_T}
=
\mathbf{1}_{\{S_T>K\}}.
\]

Thus:

\[
\Delta =
e^{-rT}
\mathbb{E}
\left[
\mathbf{1}_{\{S_T>K\}}
\frac{S_T}{S_0}
\right].
\]

This provides a useful benchmark for the automatic-differentiation implementation.

---

# Algorithmic Differentiation

## Computational Graph

The Monte Carlo calculation can be viewed as a computational graph:

```text
                 S₀
                  │
                  ▼
          ┌───────────────┐
          │   GBM Model   │
          └───────┬───────┘
                  │
                  ▼
                 S_T
                  │
                  ▼
          ┌───────────────┐
          │    Payoff     │
          │ max(S_T-K,0)  │
          └───────┬───────┘
                  │
                  ▼
          Discounted Payoff
                  │
                  ▼
          Monte Carlo Mean
                  │
                  ▼
               Price
