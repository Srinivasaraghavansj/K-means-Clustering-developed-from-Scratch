"""
Author: Srinivasaraghavan Seshadhri, MSc Artificial Intelligence student, Cork Institute of Technology, R00195470
This is Practical Machine Learning Assignment 1 Part 2A
"""

"""
The objective is to build a K means clustering algorithm from scratch according to the given specifications
in the assignment.

There isn't much to change into sci-kit learn's I/O format, hence it is going to be in the specified format.

This program has been highly optimized to be as fast as possible, by implementing the fastest functions like
numpy are array based operations, usage of loops as been minimized as much as possible, use of 'map()' function
has been implemented instead of loops for faster repeated operations.
"""

#Importing required Libraries, as minimal as possible
import numpy as np
from numpy import random
from time import time
from matplotlib import pyplot as plt

#Setting Random seed
np.random.seed(195470)

#get_ipython().run_line_magic('matplotlib', 'notebook')


#Class defined for K Means Clustering
class KM_Clustering():

    #The init function initializes reqd variables. It is done in such a way so that required data can be obtained later
    def __init__(self):
        self.cost = None
        self.centroids = None
        self.classes = None

    #Specified function calculates the Euclidean distance between a data and all given datas
    def calculate_distances(self,features,test_point):
        return np.sqrt(np.sum(np.square(features-test_point), axis=1))
    
    #Specified function generates and returns 'k' centroids from the given data
    def generate_centroids(self,features, k=3):
        return features[random.randint(0,features.shape[0],k)]
    
    #Specified function assigns each data feature into a centroid class(index of centorids, eg: 1)
    def assign_centroids(self,features,centroids):
        return np.argmin(np.array(list(map(self.calculate_distances,centroids,[features]*centroids.shape[0]))),axis = 0)
    
    #Specified function computes and returns new centroids according to the data in each centroid(s)
    def move_centroids(self,features,classes,centroids):
        return np.array([np.mean(features[classes==k],axis=0) if np.any(classes==k) else centroids[k] for k in range(centroids.shape[0])])
    
    #Specified function calculates and returns the Distortion cost
    def  calculate_cost(self,features,classes,centroids):
        return np.mean(np.square(features-centroids[classes]))


    #Specified function to implement a random restart KMeans algorithm and return the best solution found over the max_restarts
    def restart_KMeans(self,features,k=3,max_iter=10,max_restarts=10):
        cost = 99999999
        centroid = None
        for i in range(max_restarts):
            cen = self.generate_centroids(data,k)
            centroid = cen
            for j in range(max_iter):
                class_ = self.assign_centroids(data,cen)
                cen = self.move_centroids(data,class_,cen)
                centroid = cen
                cost_ = self.calculate_cost(data,class_,cen)
                if cost_<cost:
                    cost = cost_
                    centroid = cen
                self.centroids = centroid
                self.classes,self.cost = self.assign_centroids(data,centroid),cost
            return self.classes,self.cost



#The following reads data from the given CSV file and assigns them in appropriate variable
data = np.genfromtxt('data_clus.csv', dtype = 'float', delimiter = ',')

#Instanciating the class
km = KM_Clustering()

#Lists to store 'k' values and Distortion cost respectively
X = []
y = []

#Try different k values in KMeans algorithm and store respective costs
for i in range(1,11):
    X.append(i)
    y.append(km.restart_KMeans(data,i)[1])

#Uncomment the following to print Cost, centroids and labels
'''
print("\n\nDistortion Cost:", km.cost, end='\n\n')
print("Centroids:\n",km.centroids,end='\n\n\n')
print("Labels:\n",km.classes)
'''

#Plot the elbow graph
plt.plot(X,y)
plt.ylabel('Cost')
plt.xlabel("'K' value")
plt.show()





