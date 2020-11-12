---
title: "Twitter US Airline Sentiment Analysis (In Progress)"
date: 2020-11-08
header:
  overlay_image: "/images/twitter.png"
  overlay_filter: 0.5
classes: wide
excerpt: "This paper is about analyzing a dataset of Tweets on US Airlines"
---

### Problem Statement
User feedback is quite important in any business and specially, in online businesses. Simply because other users read these feedbacks before deciding to use your product or service. Social media platforms like Twitter play a major role in advertising today. Thus, it is quite important to make sure the overall sentiment for your products and services remain positive on these platforms. Passengers had a negative sentiment about most US airlines for quite some time. However, due to some recent events related to passenger safety and customer service, this negative sentiment has increased. I am trying to build a model to predict the user sentiment based on their Twitter comments, which can then be used to analyze find ways to improve the service. 

### Method
Based on the dataset there are a few methods I can use to analyze this. There is a specific column that has already identified each sentiment as positive, negative, or neutral and there is confidence rating associated too. Initially I was planning to do some regression analysis based on other features like user’s location, time zone etc. But most of those features had a lot of null values. Hence, I decided to base my analysis on text field alone at which point it became a text analysis project. 

![PNG](../images/m3.png)

• Loading Data – This was easy as there was only one data file which was a CSV.  

```
data = pd.read_csv('Tweets.csv')
data.head()
```

### Conclusion
In the recent past we have seen a lot of negative sentiment towards the US airline service due to some controversial incidents. Not just in US, but overall, in the entire world, most people are dissatisfied with the airline service. My goal in this analysis was to build a model that can accurately predict the overall sentiment of a user tweet based on the raw text. I used both classifiers and deep learning models to analyze and I got the best results using a Random forest classifier and a Linear SVM classifier on their own. Both the models had better accuracy levels with negative sentiments as a vast majority of comments were negative. We can probably improve the accuracy with a much larger dataset, so we have enough positive and neutral records to improve on. Further, we can do more cleaning on the raw text, so we extract only the most meaningful words. One should be able to use these models in a real-world environment as well.

### Important Files
Full report - [Twitter US Airline Sentiment Analysis](https://github.com/dasun27/DSC/blob/master/files/Project_3_Report_Dasun_Wellawalage.pdf)  
Dataset - [Kaggle](https://www.kaggle.com/crowdflower/twitter-airline-sentiment)

### References
•	https://www.geeksforgeeks.org/twitter-sentiment-analysis-using-python/  

This article covers the sentiment analysis of any topic by parsing the tweets fetched from Twitter using Python.  

•	https://towardsdatascience.com/twitter-sentiment-analysis-classification-using-nltk-python-fa912578614c  

This is a Twitter sentiment analysis attempt using natural language processing techniques.  

•	https://www.pluralsight.com/guides/building-a-twitter-sentiment-analysis-in-python  

Another Twitter sentiment analysis using Python based machine learning libraries.  

•	https://medium.com/@francesca_lim/twitter-u-s-airline-sentiment-analysis-using-keras-and-rnns-1956f42294ef  

This is a Twitter U.S. Airline Sentiment Analysis using Keras and RNNs.  

•	https://www.datasciencecentral.com/profiles/blogs/sentiment-analysis-of-airline-tweets  

This is a detailed guide on sentiment analysis for airline tweets.  

•	https://ipullrank.com/step-step-twitter-sentiment-analysis-visualizing-united-airlines-pr-crisis/  

This post goes into great lengths to explain not only the sentiment analysis process but also how to create an application to collect user sentiments.  

•	https://www.kaggle.com/parthsharma5795/comprehensive-twitter-airline-sentiment-analysis  

This is a notebook analyzing the same dataset found on Kaggle. He uses a classification approach for his analysis.  

•	https://www.kaggle.com/anjanatiha/sentiment-analysis-with-lstm-cnn  

This notebook tries to perform sentiment analysis using LSTM & CNN.  

•	 https://www.kaggle.com/mrisdal/exploring-audience-text-length  

Another effort to use audience & tweet length to perform sentiment analysis.  

•	https://www.kaggle.com/langkilde/linear-svm-classification-of-sentiment-in-tweets  

This notebook uses Linear SVM classification for sentiment analysis.  

