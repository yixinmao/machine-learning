---
title: "Example algorithms of machine learning"
teaching: 10
exercises: 0
questions:
- "What are some widely used machine learning algorithms?"
- "What category do they belong to?"
objectives:
- "Gain an conceptual understanding of some example machine learning algorithms"
keypoints:
- "Machine learning algorithms"
---

## Linear Regression

#### (Supervised learning, regression)

- Yes, algorithms as simple as linear regression can also be considered as machine learning!

- Simplest regression algorithm with numerous applications

- A good first try for regression problems (it's good idea to start with simple models to understand the data better before building complicated ones!)

![alt text](https://machinelearningblogcom.files.wordpress.com/2018/01/bildschirmfoto-2018-01-24-um-14-32-02.png)

<i>Illustration of linear regression (image source: https://machinelearning-blog.com/2018/01/24/linear-regression).</i>


## Kernel Smoother

#### (Supervised learning, regression)

- Instead of fitting global data, kernel smoothers only use those input data points close to the new data point <i>x<sub>0</sub></i> and fit a simple model on these "local" input data points. Weights can be assigned to these local data points.

- Requires little or no training; just need to wait for the target data point <i>x<sub>0</sub></i>, fit a local model and predict on <i>x<sub>0</sub></i>.

![alt text](../assets/img/local_smoother_ESL.png)

<i>(image source: Hastie et al. (2016): The Elements of Statistical Learning, Second Edition, Chapter 6.</i>


## Logistic Regression

#### (Supervised learning, classification)

- Don't be fooled by its name - logistic regression is the most popular classification algorithm!

- In a binary-outcome case, logistic regression predicts the probability of the outcome being 1 versus 0.

![alt text](http://uc-r.github.io/public/images/analytics/logistic_regression/plot1-1.png)

<i>Comparison of linear regression fit and logistic regression fit for a binary outcome data set. In this example, the x-axis is the input variable, credit card balance, and the y-axis is whether or not defaulting happened (image source: http://uc-r.github.io/logistic_regression).</i>


## Decision Tree and Random Forest

#### (Supervised learning, classification or regression)

- A decision tree is similar as a decision making flowchart

- A decision tree can be *learned* from data by greedily choosing the input variable and finding the spliting point that best classifies the training data, repeating this process for each branch. 

- The result of a learned decision tree is intuitive to interpret.

- A **random forest** is a large collective of trees.

    - It aggregates a large number of decision trees trained on a random subset of input data.

    - Can potentially result in better performance.

![alt text](https://imgs.xkcd.com/comics/solar_panels.png "flowchart, xkcd: Solar Panels")

<i>Illustration of a decision flowchart (image source: https://imgs.xkcd.com/comics/solar_panels.png).</i>


## Neural Network

#### (Supervised or unsupervised learning, classification or regression)

- Neural Network is framework for many algorithms, vaguely inspired by the structure of animal brains.

- ***FILL IN MORE & IMAGES***






