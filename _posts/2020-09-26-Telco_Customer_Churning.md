---
title: "Telco Customer Churning"
date: 2020-09-26
header:
  overlay_image: "/images/customerchurn.png"
  overlay_filter: 0.5
classes: wide
excerpt: "This paper is about analyzing a dataset of customer churning in the Telecom industry"
---

### Problem Statement
Customer churning is a major problem to many leading industries and Telecom is no exception. Due to the finite number of customers in a given area, it is important that Telecom companies retain their customers as it is really hard to win back those who churn away. Some studies have shown that it is way more profitable to retain customers than going after new customers. For this reason, many organizations now have a retention team specialized for retaining customers. Specifically, in the field of telecommunications, there are many attributes that would cause a customer to move out. The purpose of this analysis is to study some of those variables and figure out which ones have the most impact. 

### Method
For this analysis I am using a dataset taken from Kaggle. Unfortunately, it doesn’t state which service provider it belongs to. However, this dataset consists of 7000+ records which is not huge, but it is still enough for us to perform some analysis. There are 21 variables in the dataset and most of them have binary or categorical values. There is a ‘Churn’ feature which is what we are trying to predict. I did some exploratory data analysis first followed by building models using three different methods to predict the churning probability. Below is a step by step guide to the methods I used to conduct this analysis.  

•	Checking for null values and missing values – Most of the ML algorithms will give you errors if there are non-numeric values and even those that take categorical values will not tolerate missing or null values. In my dataset there were no missing or null values.  

```
data = pd.read_csv('datasets_13996_18858_WA_Fn-UseC_-Telco-Customer-Churn.csv')
data.dtypes
data.isnull().sum()
```

•	Data Cleanup – There were a few features that needed a little clean up. For example, some features had a value 'No internet service' that gave an identical meaning to ‘No’ in that given context. I replaced those instances with ‘No’. I also replaced the 1s and 0s in the ‘SeniorCitizen’ column with ‘Yes’ & ‘No’ to match the other binary features in the dataset.  

```
# Converting 'SeniorCitizen' column to Yes/No to make it categorical
data["SeniorCitizen"] = data["SeniorCitizen"].replace({1:"Yes",0:"No"})
data["SeniorCitizen"] = data["SeniorCitizen"].astype(object)

# Changing 'No internet service' value to 'No' in service columns as they mean the same
services = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']

for x in services:
    data[x]  = data[x].replace({'No internet service' : 'No'})
```

•	Studying individual variables – I studied some of the features using histograms to get an idea about the distribution of those individual variables. I split the dataset into two categories based on the ‘Churn’ variable. Then I compared individual features both in the large dataset and in the ‘Churn’ dataset. Below are some of the features that had prominent differences.  

```
fig, axes = plt.subplots(nrows = 1, ncols = 2)
binlist = [5, 10, 15, 20, 25 , 30, 35, 40, 45, 50, 55, 60, 65, 70]
churned = data[data["Churn"] == "Yes"]

axes[0].hist(data["tenure"], bins = binlist) # all records
axes[1].hist(churned["tenure"], bins = binlist) # churned customers
```

![PNG](/images/p1.png)  
![PNG](images/p2.png)

•	Next, I wanted to check if having additional services (apart from Internet & Phone) have an impact on churning. For this, I created a new feature which was a count of how many additional services customers have. Based on that feature, customers with more than four additional services had very little tendency to churn. However, this is somewhat obvious as you would hesitate to make changes that involve a lot of work for you.  

```
svc = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']

for x,row in data.iterrows():
    if row['InternetService'] == "No":
        data.at[x,'num_svc'] = 0
    else:
        n = 0
        for y in svc:
            if row[y] == "Yes":
                n += 1
        data.at[x,'num_svc'] = n
```

•	I used a correlation matrix to visualize the impact each variable has on the other. According to that, ‘tenure’, ‘contract’, and ‘payment method’ seemed to have a big impact on the ‘Churn’ while other features also had some impact. I used ‘plotly’ package to create the correlation matrix.  

```
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go

correlation = datanum.corr()
#tick labels
matrix_cols = correlation.columns.tolist()
#convert to array
corr_array  = np.array(correlation)

#Plotting
trace = go.Heatmap(z = corr_array,
                   x = matrix_cols,
                   y = matrix_cols,
                   colorscale = "Viridis",
                   colorbar   = dict(title = "Pearson Correlation coefficient",
                                     titleside = "right"
                                    ) ,
                  )

layout = go.Layout(dict(title = "Correlation Matrix for variables",
                        autosize = False,
                        height  = 720,
                        width   = 800,
                        margin  = dict(r = 0 ,l = 210,
                                       t = 25,b = 210,
                                      ),
                        yaxis   = dict(tickfont = dict(size = 9)),
                        xaxis   = dict(tickfont = dict(size = 9))
                       )
                  )

data = [trace]
fig = go.Figure(data=data,layout=layout)
py.iplot(fig)
```

![PNG](https://github.com/dasun27/DSC/tree/master/images/p3.png)

• Finally, I started working on my ML models. I started with Logistic regression as the dependent variable is binary. I used the classes from SciKit Learn module.  

```
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
from sklearn import metrics

feature_cols = ['tenure', 'Contract', 'PaymentMethod', 'MonthlyCharges', 'InternetService', 'Dependents', 'SeniorCitizen']
X = datanum[feature_cols]
y = datanum['Churn']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)

logreg = LogisticRegression()
logreg.fit(X_train,y_train)
y_pred=logreg.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))
```

• Next, I tried a KNN classifier. I tried a few different K values but k=5 seemed to be the optimal.  

```
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
```

• Last, I tried a Random Forest classifier.  

```
from sklearn.ensemble import RandomForestClassifier
clf=RandomForestClassifier(n_estimators=100)
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
```

### Conclusion
Customer churning is a major problem to many leading industries and Telecom is no exception. Due to the finite number of customers in a given area, it is important that Telcos retain their customers as it is really hard to win back those who churn away. Purpose of this analysis is to build a model to predict customer churning based on this dataset. However, this should only be used to gain some insight and as a steppingstone to a much larger production scale prediction model, as such a system should include many more factors than what is used in this analysis.

### Important Files
Full report - [Telco Customer Churning](https://github.com/dasun27/DSC/blob/master/files/Project_1_Report_Dasun_Wellawalage.pdf)  
Dataset - [Kaggle](https://www.kaggle.com/blastchar/telco-customer-churn)

### References
•	Datacamp (Aug 2018). KNN Classification using Scikit-learn. Retrieved Sep 20, 2020, from https://www.datacamp.com/community/tutorials/k-nearest-neighbor-classification-scikit-learn

Detailed explanation of using KKN classifier with SciKit Learn

•	Datacamp (Dec 2019). Understanding Logistic Regression in Python. Retrieved Sep 20, 2020, from https://www.datacamp.com/community/tutorials/understanding-logistic-regression-python

Detailed explanation of using Logistic Regression with SciKit Learn

•	Datacamp (May 2018). Understanding Random Forest Classifiers in Python. Retrieved Sep 20, 2020, from https://www.datacamp.com/community/tutorials/random-forests-classifier-python

Detailed explanation of using Random Forest Classifier with SciKit Learn

•	Medium (Oct 2019). Analysis of Telco Customer Churn Dataset. Retrieved Sep 20, 2020, from https://medium.com/@kmacver/analysis-of-telco-customer-churn-dataset-926ff04d2295

An analysis on customer churning in the Telecom industry

•	Rstudio (2020). Telecom Churn Analysis. Retrieved Sep 20, 2020, from http://rstudio-pubs-static.s3.amazonaws.com/443094_bc2c15d74e7e4b7b96d8fc95f3162b08.html

A detailed analysis on customer churning using Rstudio

•	Statista (Jun 2020). Customer churn rate in the United States in 2018, by industry. Retrieved Sep 20, 2020, from https://www.statista.com/statistics/816735/customer-churn-rate-by-industry-us/  

This shows the customer churn rate in US in 2018 for different industries. You can see how the Telecom industry does compared to others.  

•	NYU (2017). Churn in the Telecom Industry – Identifying customers likely to churn and how to retain them. Retrieved Sep 20, 2020, from https://wp.nyu.edu/adityakapoor/2017/02/17/churn-in-the-telecom-industry-identifying-customers-likely-to-churn-and-how-to-retain-them/  

This describes the current churning problem within the Telecom industry and how to address that using a data driven approach.  

•	McKinsey & Company (Dec 2017). Reducing churn in telecom through advanced analytics. Retrieved Sep 20, 2020, from https://www.mckinsey.com/industries/technology-media-and-telecommunications/our-insights/reducing-churn-in-telecom-through-advanced-analytics#  

A comprehensive, analytics-driven approach to base management which can help telecom companies reduce churn by as much as 15%.  

•	Medium (Nov 2019). Telco Customer Churn Prediction. Retrieved Sep 20, 2020, from https://towardsdatascience.com/telco-customer-churn-prediction-72f5cbfb8964  

This is a detailed article on building a machine learning model for Churn prediction.  

•	Medium (Nov 2018). Cutting the Cord: Predicting Customer Churn for a Telecom Company. Retrieved Sep 20, 2020, from https://towardsdatascience.com/cutting-the-cord-predicting-customer-churn-for-a-telecom-company-268e65f177a5  

An analysis on customer churning in the Telecom industry using a Kaggle dataset.  

•	Database Marketing Institute (Sep 2020). Churn reduction in the telecom industry. Retrieved Sep 20, 2020, from http://www.dbmarketing.com/telecom/churnreduction.html  

This is a marketing approach to customer churning in the Telecom industry.  

•	Rstudio (2020). Telecom Churn Analysis. Retrieved Sep 20, 2020, from http://rstudio-pubs-static.s3.amazonaws.com/277278_427ca6a7ce7c4eb688506efc7a6c2435.html  

A detailed analysis on customer churning using Rstudio  

•	Techsee (2019). Reasons for Customer Churn in the Telecom Industry: 2019 Survey Results. Retrieved Sep 20, 2020, from https://techsee.me/resources/surveys/2019-telecom-churn-survey/  

A survey conducted in 2019 on the reasons for customer churning in the Telecom industry.  

•	Rutgers (2020). Telecom customer churn prediction. Retrieved Sep 20, 2020, from https://rucore.libraries.rutgers.edu/rutgers-lib/62514/  

A research done by Rutgers university on Telecom customer churn prediction

