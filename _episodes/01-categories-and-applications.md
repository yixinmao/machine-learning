---
title: "Major categories of machine learning"
teaching: 15
exercises: 0
questions:
- "What are the major categories of machine learning?"
- "What are some example applications for each?"
objectives:
- "Understand the major categories of machine and what kind of problems they can each solve."
keypoints:
- "machine learning categories, supervised learning, unsupervised learning, regression, classification"
---

# Major Categories of Machine Learning and Example Applications

![alt text](https://www.mathworks.com/content/mathworks/www/en/discovery/machine-learning/jcr:content/mainParsys3/discoverysubsection_1965078453/mainParsys/image_2128876021_cop.adapt.full.high.svg/1551847794310.svg)

<i> Major categories of machine learning (image source: MathWorks). </i>

## 1. Supervised Learning

Supervised learning learns a prediction model from past input data and observed outcomes (called **training data**), such that the prediction can take new input data and estimate outcome.

- **Sub-categories of supervised learning**:

    - **Regression**: outcomes are continuous values

        - Example applications: ***FILL IN***

        - Example applications in water resources research: ***FILL IN***

    - **Classification**: outcomes are catogorical values

        - Example applications: ***FILL IN; use Shay's examples***

        - Example applications in water resources research: ***FILL IN; use Shay's example on land classification***


## 2. Unsupervised Learning

Unsupervised learning groups, interprets and find patterns in input data without outcome data.

- **Unsupervised learning example 1: Clustering**

    - Clustering methods cluster a dataset into multiple groups.

    - Example applications: ***FILL IN***

    - Example applications in water resources research: ***FIND SOME AND FILL IN***

![alt text](https://pro.arcgis.com/en/pro-app/tool-reference/spatial-statistics/GUID-A06A412D-2F4F-4D35-8FFF-1F4B3B3A8F16-web.png)
<i> Illustration of clustering (image source: ArcGIS). </i>

- **Unsupervised learning example 2: Principal Component Analysis (PCA)**

    - PCA is a statistical procedure that transforms a set of input data of possibly correlated variables into a set of values of linearly uncorrelated variables called **principal components**. After the transformation, the first principle component accounts for as much of the variability in the data as possible, and the second principle component accounts for the largest variability in the remaining dimensions of the data that are orthogonal to (i.e., have no linear correlation with) the first principle component, and so on so forth (source: [Wikipedia](https://en.wikipedia.org/wiki/Principal_component_analysis)).

    - Example applications: ***FILL IN***

    - PCA is widely used in atmospheric sciences, often referred to as **Empirical Orthogonal Function (EOF)** analysis in the field. For example, EOF can be applied on a temporal-spatial climate dataset to find the most dominant spatial and temporal patterns. ***FILL IN MORE DETAILS AND REFERENCES***

![alt text](https://upload.wikimedia.org/wikipedia/commons/thumb/f/f5/GaussianScatterPCA.svg/1280px-GaussianScatterPCA.svg.png)

<i>For a input dataset with two variables, the PCA algorithm finds the first principal component that accounts for the largest data variability (the longer vector in the figure) and the second principal component that is orthogonal to the first one (the shorter vector in the figure) (image source: https://en.wikipedia.org/wiki/Principal_component_analysis)</i>.

# More References for Machine Learning Applications in Water Resources

# More Resources
https://www.mathworks.com/discovery/machine-learning.html


## [“What does the machine (i.e. the statistical model) actually learn?”](https://towardsdatascience.com/linear-regression-using-gradient-descent-in-10-lines-of-code-642f995339c0)

1. We have a set of data (x) and a set of target values (y)
1. We want to iteratively determine, or learn, the function that relates x to y
1. We want to automatically iterate to learn this function by using statistics (minimize a cost or loss function)

This will vary, but in simple terms the model learns a function f such that f(X) maps to y, where X are the features (or independent variables) and y is the target (dependent variable).  

The most basic example is simple linear regression. 

<i>y(x) = m x + b</i>

<i>m = slope</i>

<i>b = intercept (bias)</i>

By tweaking m and b, we can create a line that will best describe the relationship between x and y. How do we know we’re close? We use a cost (loss) function. A high cost value means it’s expensive — our approximation is far from describing the real relationship. On the other hand, a low cost value means it’s cheap — our approximation is close to describing the relationship.

<i> Basic Idea </i>
1. we need m,b
1. calculate a new y through iteration of possible m,b values
1. assess how close you are (use a cost function)
1. the process of iteration on m,b is done through gradient descent (subtract partial derivatives w.r.t. m, b to get the cost function error to a minimum)
1. gradient gives us direction to minimize cost function
1. learning rate tells us how big a step to take in that direction

For ML applications, there are many cost functions. For linear regression, we can use MSE.

We can't brute force our way through iterating on values for m & b. We can use something called Gradient Descent, which is the act of applying partial derivatives, with respect to both m and b, to the cost function to point us to the lowest point. 

A derivative of zero means you are at either a local minima or maxima. Which means that the closer we get to zero, the better. When we reach close to zero with our derivatives, we also get the lowest value for our cost function.

![alt-text](../assets/img/graddescent.png "Logo Title Text 1")


```
# adapted from https://github.com/mattnedrich/GradientDescentExample
from numpy import *
import matplotlib.pyplot as plt

# y = mx + b
# m is slope, b is y-intercept

def compute_mean_error_for_line_given_points(b, m, points):
    totalError = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        totalError += (y - (m * x + b)) ** 2
    return totalError / float(len(points))

def step_gradient(b_current, m_current, points, learningRate):
    b_gradient = 0
    m_gradient = 0
    N = float(len(points))
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        b_gradient += -(2/N) * (y - ((m_current * x) + b_current))
        m_gradient += -(2/N) * x * (y - ((m_current * x) + b_current))
    new_b = b_current - (learningRate * b_gradient)
    new_m = m_current - (learningRate * m_gradient)
    return [new_b, new_m]

def gradient_descent_runner(points, starting_b, starting_m, learning_rate, num_iterations):
    b = starting_b
    m = starting_m
    for i in range(num_iterations):
        b, m = step_gradient(b, m, array(points), learning_rate)
        plt.plot(points[:,0],m*points[:,0]+b,'r')
        plt.pause(1)
        plt.show()    

    return [b, m]

def run():
	plt.figure()
	plt.ion()
	points = genfromtxt("../code/data.csv", delimiter=",")
	learning_rate = 0.0001
	initial_b = 0 # initial y-intercept guess
	initial_m = 0 # initial slope guess
	num_iterations = 100
	plt.axis([20, 80, 0, 150])
	plt.plot(points[:,0],points[:,1],'ob')
	plt.show()    
	print ("Starting gradient descent at b = {0}, m = {1}, error = {2}".format(initial_b, initial_m, compute_mean_error_for_line_given_points(initial_b, initial_m, points)))
	print ("Running...")
	[b, m] = gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations)
	print ("After {0} iterations b = {1}, m = {2}, error = {3}".format(num_iterations, b, m, compute_mean_error_for_line_given_points(b, m, points)))

if __name__ == '__main__':
    run()
```

![alt-text](../assets/img/gradient_descent_example.gif "Logo Title Text 1")

## ML as AI

![alt-text](../assets/img/ml_ai.png "Logo Title Text 1")

<i> Machine learning is a component of artificial intelligence (AI) (a broader subject). There are many subject areas where ML may be applied, e.g. sound, language, and vision (essentially, traits we can identify with as a human). We can differentiate between machine learning and classical techniques. </i>

## Computer Vision Applications


![alt-text](../assets/img/cat.png "Logo Title Text 1")

<i> Images as RGB digital counts. </i>

![alt-text](../assets/img/ml_image_type.png "Logo Title Text 1")

<i> Types of computer vision ML aplications. </i>

![alt-text](../assets/img/segmentation.png "Logo Title Text 1")

<i> Semantic segmentation is the act of labelling each individual pixel. </i>

## Supervised vs. Unsupervised

![alt-text](../assets/img/supervised_unsuper.png "Logo Title Text 1")

<i> Supervised classification vs. unsupervised. </i>


## ML Ethics

[Articles about bias/politics/etc](https://medium.com/@eirinimalliaraki/toward-ethical-transparent-and-fair-ai-ml-a-critical-reading-list-d950e70a70ea)

## ML Scalability

1. Making sense of pedabytes of imagery in a consistent way
1. Human feedback loop (QA/QC)

![alt-text](../assets/img/cloud_scale.png "Logo Title Text 1")


## GIS is Ideal for ML

Example Use Cases:

1. Land classification (vegetation monitoring, growth, decline, change)
1. Impervious surface
1. Change detection/anomaly
1. Geosptatial attribute trending (census, twitter)
1. Agriculture
1. Road networks
1. Object identification & tracking (ships, cars)
1. Imagery mosaicing, stitching, pre-processing
1. Resolution enhancement
1. 3D modeling & Digital Elevation/Surface Mapping
1. Coastal vegetation monitoring
1. Kriging

![alt-text](../assets/img/lc.png "Logo Title Text 1")

<i> Pedabytes of imagery may be processed repeatable, and quickly in the cloud, using ML techniques. </i>

![alt-text](../assets/img/littoral.png "Logo Title Text 1")

<i> FL seagrass ground truth. </i>

![alt-text](../assets/img/littoral_pred.png "Logo Title Text 1")

<i> FL seagrass ML prediction mapping. </i>

![alt-text](../assets/img/impervious.png "Logo Title Text 1")

<i> Impervious surface mapping (RGB left, boolean prediction map right). </i>

![alt-text](../assets/img/houston_flood.png "Logo Title Text 1")
![alt-text](../assets/img/houston_flood_fcn.png "Logo Title Text 1")

<i> Hurricane Harvey Houston flood mapping </i>

![alt-text](../assets/img/fire_ai.png "Logo Title Text 1")

<i> Santa Rosa fire mapping </i>
