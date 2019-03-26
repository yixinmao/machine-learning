---
title: "What is machine learning?"
teaching: 10
exercises: 0
questions:
- "What is machine learning? How is it different from traditional algorithms?"
objectives:
- "Get a basic understanding of machine learning."
keypoints:
- "machine learning, traditional algorithms"
---

# What is Machine Learning (ML)?

**Short answer: letting the computer learn rules from data.**

In a typical scenario, we have a **training set** of data observed in the past:

- A set of input data (e.g., satellite image color, measurement time of year, location)
- Outcome measurements (e.g., landcover type)

Using this data, we build a **prediction model**, or **learner**, that is able to predict the outcome for new input data.

Learning itself is the act of gradually improving performance on a task without being explicitly programmed. This process mimics human neurological functions (e.g., teaching a child to speak): **repetition with more data results in a better learner**.

![alt text](../assets/img/machine_learning_illustration.png)

<i>Figure: Illustration of machine learning. </i>

# Major Categories of Machine Learning and Example Applications

![alt text](https://www.mathworks.com/content/mathworks/www/en/discovery/machine-learning/jcr:content/mainParsys3/discoverysubsection_1965078453/mainParsys/image_2128876021_cop.adapt.full.high.svg/1551847794310.svg)

<i>Figure: Major categories of machine learning (image source: MathWorks). </i>

## 1. Supervised Learning

Supervised learning learns a prediction model from past **input data** and **observed outcomes**, such that the prediction can take new input data and estimate outcome.

- **Sub-categories of supervised learning**:

    - **Regression**: outcomes are *continuous* values

        - Example applications:

            - Predict housing price given house size, location, number of rooms and year

            - Predict streamflow in the next week given mean annual streamflow, streamflow in the past week, rainfall in the past week and day of year

    - **Classification**: outcomes are *categorical* values

        - Example applications:

            - Classify handwritten digits

            - Land classification

![alt text](https://www.wolfram.com/mathematica/new-in-10/enhanced-image-processing/HTMLImages.en/handwritten-digits-classification/smallthumb_10.gif)

<i>Figure: Handwritten digits classification (image source: Wolfram).</i>

![alt text](https://www.researchgate.net/profile/Jong_Min_Yeom/publication/272737692/figure/fig3/AS:667813249290245@1536230494379/Land-cover-classification-with-RapidEye-satellite-dataleft-original-imagery-with-true.ppm)

<i>Figure: Land cover classification with RapidEye satellite data(left: original imagery with true color composition, right: classification result) (image source: Kim and Yeom (2012): Effect of the urban land cover types on the surface temperature: case study of Ilsan New City), Korean Journal of Remote Sensing, 28(2), 203~214).</i>

