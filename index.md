# ML Grad Project Midterm Report
##### Saurav Desai, Aarun Srinivas, Kanav Garg, Krish Dholakia, Rahul Hardikar


### Introduction/Background

Finances are an incredibly important part of everyday life. One of the most important verticals within finance is credit and lending. The ability to accurately determine whether the credit-worthiness of an individual and the likelihood that an individual will return the capital, is key to building sustainable financial institutions over the long run. Beyond just financial institutions, it is critical that we - as people - are able to accurately model our own credit, to help prevent us from being overextended, and taking on larger debt than we might reasonably be able to pay off. Thus, the aim of our project will be to determine whether someone will default or not on their loan.

### Problem Definition

For this class, we will be building a model to make predictions on whether a loan will default or not. For this analysis, we will be analyzing over 75 signals to determine whether an individual will default or not on their loans. Noise will be a major factor, and denoising the data to get the appropriate signal/noise ratio will be crucial to our success in this project. In particular, we will be analyzing data that involves details such as - the loan amount provided, the annual income of an individual, the interest rate, etc.

### Methods

There are several methods worth investigating. They are mentioned below, however, these are subject to change as we evolve our approaches to produce more accurate results. 

1. Linear/Logistic Regression
2. Random Forest
3. SVC's or Decision Trees
4. K-means Clustering
5. PCA

### Data Cleaning

Our dataset consists of more than 5-10 million datapoints, which would require a lot of resources to analyze and process, so, we decided to only consider the first 500,000 entries for exploratory data analysis and preliminary statistical learning. 

The dataset consists of over 150 columns, out of which multiple columns seems to have incomplete data. First, we eliminate all columns with more than 10% missing data entries, since these wouldn’t be helpful in finding correlations. We also eliminate all loans that have a “current” status since these do not have an outcome yet. 

By applying some basic domain knowledge, we were able to eliminate more irrelevant columns such as loan id, urls, employment titles, etc. and duplicate columns (funded_amnt, funded_amnt_inv), as these would not provide any predictive value to our model. We then create a heatmap of our dataset to visualize any missing data. 

[Insert heatmap 1 here] 

Next, we simply drop any remaining data entries that have a null value, which completes our dataset. 

[insert heatmap 2 here]

We still have ~70 features remaining, which is still very large. During our exploratory data analysis stage, we will attempt to find features that have higher correlations with loan outcomes to further eliminate unnecessary features. We would also attempt to use PCA to reduce our dimensionality.

Since one of our additional objectives is to use our model to predict the outcomes of rejected loan applications, we could select the 8 features that are available to us in the rejected dataset. 

Overall, since a financial institution usually doesn’t have too many features available on a client, it makes sense for us to trim down on our features significantly and weigh features such as fico scores, loan grades, and interest rates more since these are known to be relevant and available in most loan applications. 

#### Data Encoding

Since a few of our columns are strings, we must encode this into numeric values so a classification model understands the data. For the loan status column, we chose to define all good loans as 1 and bad loans as 0. So, all loan statuses that are negative (Charged off, Late on payments), are replaced with 0. For all other columns, we run a simple loop where we assign a random integer value to every unique value in a particular column. 


### Potential Results

It is our goal that through the above listed methods, we will be able to understand macro-trends and generalized patterns behind loan eligibility and determine future probability of people meeting their loan requirements. We aim to accurately predict the likelihood that people will return capital over a period of time, and visualize these trends over time. We anticipate our different approaches will yield a diverging set of predictions, and we will use this information along with our split of data into train/dev/test to evaluate the best approach. 

### Discussion

Since the ability for a person to return capital is dependent on a multitude of factors, we have the dual challenge of a) ensuring we cover as many potential signals as possible b) we weight these signals appropriately. Our ability to weigh these signals will partially depend on the size and nature of our training data. If our data is too sparse, we might overweight certain features, harming the accuracy of our loan prediction model. However, a good loan prediction model can have a powerful impact in the world of finance - potentially allowing millions of people who might otherwise have been rejected to gain access to appropriate financial resources. We can further leverage the fact that as the field of AI advances, loan prediction models will become increasingly accurate and unbiased. 


### Dataset
- [Lending Club Dataset from Kaggle](https://www.kaggle.com/wordsforthewise/lending-club)

### References

1. https://medium.com/devcareers/loan-prediction-using-selected-machine-learning-algorithms-1bdc00717631
2. https://towardsdatascience.com/predict-loan-eligibility-using-machine-learning-models-7a14ef904057
3. https://www.jstor.org/stable/1991095






