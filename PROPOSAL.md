## Project Proposal

I’ll focus on two complementary methods, using a panel dataset covering ten years (2014–2024) for all Swiss cantons, so around 260 observations and around 30 variables.  
The main objective is to understand how local housing and labor market conditions influence inter-cantonal migration over time. The dependent variable will be the yearly change in the net migration rate, explained by variables such as:

- average rent  
- income growth  
- unemployment change  
- mortgage rate variation  
- housing construction per capita  
- share of the population aged 65 and over  

I plan to compare three approaches:

1. **Ordinary Least Squares (OLS)** regression as a baseline to interpret the direction and magnitude of effects,  
2. **Ridge regression (L2)** to handle multicollinearity between correlated variables like rent and income,  
3. **Random Forest** to capture possible nonlinearities or interaction effects that linear models may miss.  

Model performance will be evaluated using **time-based validation** (training: 2014–2021, testing: 2022–2024) with **R²** and **RMSE**.  
The goal is to compare predictive accuracy and the robustness of estimated effects. If the results suggest strong uncertainty or heterogeneous responses, I may add a **short Monte Carlo “what-if” simulation** to explore how migration could react to hypothetical changes in mortgage rates or income growth.

As a complementary part, I’ll also use **k-means clustering** on standardized long-run averages of key variables (migration, rent, income, unemployment, construction) to identify cantons with similar migration and economic profiles. This may reveal patterns across linguistic or urban–rural regions.

All data come from the **Swiss Federal Statistical Office (BFS/OFS)**, covering cantonal migration, income, unemployment, housing, and mortgage statistics.

---

## Project Topic

**“Inter-cantonal migration dynamics and their sensitivity to housing and mortgage market conditions in Switzerland.”**

The idea is to study how differences in rent levels, income growth, unemployment, and mortgage rates across cantons affect population movements over time. This topic combines economic, demographic, and housing dimensions and allows for rigorous quantitative analysis using real Swiss data (2014–2024).
