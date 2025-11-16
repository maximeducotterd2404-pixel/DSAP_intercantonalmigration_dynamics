# Research Proposal: Migration Dynamics in Swiss Cantons# Research Proposal: Migration Dynamics in Swiss Cantons



## Overview## Overview



This research focuses on two complementary methods to analyze how local housing and labor market conditions influence inter-cantonal migration over time.This research focuses on two complementary methods to analyze how local housing and labor market conditions influence inter-cantonal migration over time.



**Dataset:** Panel data covering 2014–2024 for all Swiss cantons (~260 observations, ~30 variables)**Dataset:** Panel data covering 2014–2024 for all Swiss cantons (~260 observations, ~30 variables)’ll focus on two complementary methods, using a panel dataset covering ten years (2014–2024) for all Swiss cantons, so around 260 observations and around 30 variables.

The main objective is to understand how local housing and labor market conditions influence inter-cantonal migration over time. The dependent variable will be the yearly change in the net migration rate, explained by variables such as average rent, income growth, unemployment change, mortgage rate variation, housing construction per capita, and the share of the population aged 65 and over.

---I plan to compare three approaches:

Ordinary Least Squares (OLS) regression as a baseline to interpret the direction and magnitude of effects,

## Research ObjectivesRidge regression with L2 regularization to handle multicollinearity between correlated variables like rent and income,

Random Forest to capture possible nonlinearities or interaction effects that linear models may miss.

The main objective is to understand how local housing and labor market conditions influence inter-cantonal migration over time. Model performance will be compared through time-based validation (training on 2014–2021, testing on 2022–2024) using R² and RMSE. The goal is to evaluate both predictive accuracy and the robustness of estimated effects. If the results suggest strong uncertainty or heterogeneous responses, I might add a short Monte Carlo “what-if” simulation to explore how migration could react to hypothetical changes in mortgage rates or income growth.

As a complementary part, I’ll also use k-means clustering on standardized long-run averages of key variables (migration, rent, income, unemployment, construction) to identify cantons with similar migration and economic profiles. This could help reveal patterns across linguistic or urban–rural regions.

**Dependent Variable:** Yearly change in the net migration rateAll data come from the Swiss Federal Statistical Office (BFS/OFS), covering cantonal migration, income, unemployment, housing, and mortgage statistics.

**Key Explanatory Variables:**
- Average rent
- Income growth
- Unemployment change
- Mortgage rate variation
- Housing construction per capita
- Share of the population aged 65 and over

---

## Methodological Approach

### 1. **Regression Analysis** (Comparative Study)

Three complementary approaches will be compared:

#### a) Ordinary Least Squares (OLS)
- Baseline model for interpreting direction and magnitude of effects

#### b) Ridge Regression (L2 Regularization)
- Handles multicollinearity between correlated variables (e.g., rent and income)

#### c) Random Forest
- Captures nonlinearities and interaction effects that linear models may miss

### 2. **Model Validation**

- **Time-based validation:** Training period 2014–2021, Testing period 2022–2024
- **Performance metrics:** R² and RMSE
- **Focus:** Evaluate both predictive accuracy and robustness of estimated effects

### 3. **Sensitivity Analysis** (Optional)

Monte Carlo "what-if" simulations to explore how migration could react to hypothetical changes in:
- Mortgage rates
- Income growth

---

## Complementary Analysis: Clustering

**K-means Clustering** on standardized long-run averages of key variables:
- Migration
- Rent
- Income
- Unemployment
- Construction

**Objective:** Identify cantons with similar migration and economic profiles, revealing patterns across:
- Linguistic regions
- Urban–rural dimensions

---

## Data Sources

All data come from the **Swiss Federal Statistical Office (BFS/OFS)**, including:
- Cantonal migration statistics
- Income data
- Unemployment figures
- Housing market data
- Mortgage statistics
