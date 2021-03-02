# ML Grad Project Proposal


### Introduction/Background

Finances are an incredibly important part of everyday life. One of the most important verticals within finance is credit and lending. The ability to accurately determine whether the credit-worthiness of an individual and the likelihood that an individual will return the capital, is key to building sustainable financial institutions over the long run. Beyond just financial institutions, it is critical that we - as people - are able to accurately model our own credit, to help prevent us from being overextended, and taking on larger debt than we might reasonably be able to pay off. Thus, the aim of our project will be to determine whether someone will default or not on their loan.

### Problem Definition

For this class, we will be building a model to make predictions on whether a loan will default or not. For this analysis, we will be analyzing over 75 signals to determine whether an individual will default or not on their loans. Noise will be a major factor, and denoising the data to get the appropriate signal/noise ratio will be crucial to our success in this project. In particular, we will be analyzing data that involves details such as - the loan amount provided, the annual income of an individual, the interest rate, etc.

### Methods

There are several methods worth investigating. A few of them are mentioned below, however, these are subject to change as we explore our dataset further and evolve our approaches to produce more accurate results. 

1. Linear Regression
2. Hierarchical Classification
3. Random Forest
4. SVC's or Decision Trees
5. K-means Clustering
6. PCA

### Potential Results

It is our goal that through the above listed methods, we will be able to understand macro-trends and generalized patterns behind loan eligibility and determine future probability of people meeting their loan requirements. We aim to accurately predict the likelihood that people will return capital over a period of time, and visualize these trends over time. We anticipate our different approaches will yield a diverging set of predictions, and we will use this information along with our split of data into train/dev/test to evaluate the best approach. 

### Discussion

Since the ability for a person to return capital is dependent on a multitude of factors, we have the dual challenge of a) ensuring we cover as many potential signals as possible b) we weight these signals appropriately. Our ability to weigh these signals will partially depend on the size and nature of our training data. If our data is too sparse, we might overweight certain features, harming the accuracy of our loan prediction model. However, a good loan prediction model can have a powerful impact in the world of finance - potentially allowing millions of people who might otherwise have been rejected to gain access to appropriate financial resources. We can further leverage the fact that as the field of AI advances, loan prediction models will become increasingly accurate and unbiased. 


### Potential Dataset
- [Dataset from Kaggle](https://www.kaggle.com/wordsforthewise/lending-club)

### References

1. https://medium.com/devcareers/loan-prediction-using-selected-machine-learning-algorithms-1bdc00717631
2. https://towardsdatascience.com/predict-loan-eligibility-using-machine-learning-models-7a14ef904057
3. https://www.jstor.org/stable/1991095






