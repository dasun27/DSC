---
title: "Housing Affordability"
date: 2020-05-28
classes: wide
excerpt: "This paper is about analyzing housing affordability in US"
---

### Problem Statement
Housing affordability is one of the biggest problems an average American has to solve during his lifetime. In fact, for most people, it accounts for the biggest recurring payment made every month whether it’s a rented property or owned. The Housing Affordability Data System (HADS) is a set of housing-unit level datasets that measures the affordability of housing units and the housing cost burdens of households, relative to area median incomes, poverty level incomes, and Fair Market Rents. This project aims to model the amount of housing cost burden, a family would be willing to take based on their household income, age of the head of household, monthly expenses and many other factors.

### Conclusion
Random Forest models clearly had a higher accuracy level over the KNN model. One area for improvement would be the pre-processing stage. I tried centering and scaling but there are many other transformation methods that can be tested. Another area for improvement would be the variable selection. Although increasing the number of independent variables didn’t produce better results, we can still try different combinations of variables, or even a derived variable if that helps improve the accuracy of the model. Overall, the Random Forest model with 3 independent variables and pre-processing enabled, produced the best results.

### Important Files
Full report - [Housing Affordability](https://github.com/dasun27/DSC/blob/master/files/Housing%20Affordability.pdf)  
Dataset - [USA Federal Government](https://www.huduser.gov/portal/datasets/hads/hads.html)

