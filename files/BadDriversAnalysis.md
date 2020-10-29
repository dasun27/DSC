---
title: "Bad Drivers Analysis"
date: 2020-02-22
tags: [Python, data science, Machine Learning, Analysis, Driving]
header:
  image: "/images/cars.jpg"
  caption: "Photo credit: Disney Movie Cars"
classes: wide
excerpt: "This project analyzes a dataset from Dear Mona, Which State Has The Worst Drivers? Article"
---

Read more about my full analysis at [Bad Drivers Analysis](https://nbviewer.jupyter.org/github/thanhnguyenduong/DSC530_Bad_Drivers_Analysis/blob/master/DSC%20530%20Final%20Project.ipynb)

### Dataset
The dataset used in my analysis is from Dear Mona, Which State Has The Worst Drivers? article, and was obtained through 
[Kaggle](https://www.kaggle.com/fivethirtyeight/fivethirtyeight-bad-drivers-dataset).

The dataset includes these variables:

Variable | Source
---|---------
State | N/A
Number of drivers involved in fatal collisions per billion miles | National Highway Traffic Safety Administration, 2012
Percentage Of Drivers Involved In Fatal Collisions Who Were Speeding | National Highway Traffic Safety Administration, 2009
Percentage Of Drivers Involved In Fatal Collisions Who Were Alcohol-Impaired | National Highway Traffic Safety Administration, 2012
Percentage Of Drivers Involved In Fatal Collisions Who Were Not Distracted	 | National Highway Traffic Safety Administration, 2012
Percentage Of Drivers Involved In Fatal Collisions Who Had Not Been Involved In Any Previous Accidents | National Highway Traffic Safety Administration, 2012
Car Insurance Premiums ($) | National Association of Insurance Commissioners, 2011
Losses incurred by insurance companies for collisions per insured driver ($) | National Association of Insurance Commissioners, 2010

### Problem Statement and Hypothesis
I wanted to find out if percentages of driver involved in fatal collisions who were alcohol-impaired may have influence on the percentages of driver involved in fatal collisions who were speeding.

### Summary of my analysis 
The outcome of my EDA was that there was no significant relationship between these two variables since the p-value was 0.491 which is > 0.05, it failed to reject the null hypothesis. The percentages of driver involve in fatal collisions who were speeding will can increase or decrease in the event of with or without alcohol-impairment. However, as for correlation from scatter plot result, there is a what seems like a very weak positive correlation between these two variables, but it is to weak and there are many other confounding factors to determine any significance or infer causation between these two variables. 

#### What do you feel was missed during the analysis?  
There are other variables in this dataset that I thought it was irrelevant to my hypothesis, so I did not include it in the data analysis such as insurance premiums, state     and losses incurred by insurance companies for collisions per insured driver variable. With the addition of those variables, it may show some causation from any unforeseen     correlation. Aside from that, I feel like I did not fully analyze my dataset by individual State since the percentage between these variables may vary between different         States. For some States, the percentage between these variables may correlate while others may not.

### Any variables I felt could have helped in my analysis?
The variable State could have helped in my analysis if I included, because there may be variations in percentages among States. Some may have high percentages, and some may have lower percentages which can skew the datasets. As a result, it may create a false negative error in my analysis. By including State variable, I feel like the analysis would be a lot better and I could have gotten a full picture on how percentages of my variables differ among different States.

### Assumptions I made that I felt incorrect
I assumed that other factors like NotDistracted, and NoPrevAccidents could have play a role in influence the number of fatal collisions per billion miles. In addition, I assumed people will have a higher chance of speeding with alcohol impairment, but it turns out there were no correlations, or I did not see the correlation as much as I expected. Also, there may be some outliers left behind even after I cleaned the data. 
