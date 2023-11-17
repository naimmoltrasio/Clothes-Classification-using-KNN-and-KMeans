# Clothing Labelling using Machine Learning
Image labeling problem that I solved as a project in the Artificial Intelligence subject at the university.
## Table of contents
* [General information](#general-info)
* [Technologies](#technologies)
* [Algorithms Implemented](#algorithms-implemented)
* [Setup](#setup)
* [Example](#example)

### General information
This projects aims to properly classify different types of clothes within their colour or their type. Furthermore, we will we able to calculate the accuracy each algorithm we use to solve this labelling problem has.
### Technologies
Project is created using:
 * Python 3.9
 * matplotlib
 * numpy 
### Algorithms Implemented
We will be using K-means algorithm for retrieving the most predominant colors in the different images.
 * K-means is an unsupervised learning algorithm used for clustering data points into groups or clusters.
 * How it works? The algorithm partitions the data into k clusters based on the similarity of data points. It assigns each data point to the cluster whose centroid (center) is closest to that point. The centroids are updated iteratively until convergence, and the process continues until the clusters are well-defined.

On the other hand we use KNN algorithm to retrieve the different types of clothes.

 - KNN is a supervised learning algorithm used for classification and regression tasks. It can be used for both categorical and numerical data.
 - How it works? Given a new, unlabeled data point, KNN classifies or predicts its label based on the majority class or average of the k-nearest data points in the feature space. The "k" in KNN represents the number of nearest neighbors that influence the prediction. The distance metric (usually Euclidean distance) is used to determine the proximity of data points.

### Setup 
To run this project, install the required packages from the requirements.txt file.
```
pip install -r requirements.txt
```
Once you run the project, a menu in the console will appear with different options and its respective descriptions.
### Example
In this example we will visualize the k-means results, so we choose option number 1 in the console.
We will get the first 5 images of our dataset (this number can be changed) as it follows:
![k-means](/results/k-means.png)
