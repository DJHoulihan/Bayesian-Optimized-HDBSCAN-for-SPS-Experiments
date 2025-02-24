# MLinPhysicsFinalProject: Clustering Algorithm for Reaction Identification with the Super-Enge Split-Pole Spectrograph at the John D. Fox Laboratory
The final project code, data, and documents for the Machine Learning in Physics course. The goal is to create an ML algorithm that can classify different particle groups from a PID plot created during nuclear experiments. If successful, it would be cool to try and make it also create a gate around the group of interest so that reaction can be gated upon for future analysis.

There are two forms of this algorithm: one constructed with python Classes and one with just functions. Each form is put in their respective folders. Outside of these directories, there is the roottotxt.py file which holds a function to take the PID data from a standard SE-SPS EventBuilder ROOT file and put it into a text file. 

## Functions
The entire algorithm is run through the function 'PIDClassifier', which is located in the .py file of the same name. PIDClassifier calls functions from the other .py files in the repository. 'PIDClassifierNotebook.ipynb' contains the tests of the algorithm and the results of it being run. 

## Classes
I do not have too much experience with classes, so I wanted to make the algorithm into different class structures. This could help with updating and implementing in other programs or GUI's. The imports.py function was removed and instead I just imported the required libraries in the .py files themselves. 

PIDClassifier.py - The full algorithm is ran with the run() function within the PIDClassifier class in the PIDClassifier.py file. 

clusterplots.py - The plotting functions are placed within the ClusterPlots class in the clusterplots.py file. 

density_aware_resample.py - The density_aware_resample.py file holds the resampling function

get_particle_class.py - Holds the GetParticleClass class. This is where the classify_particle function lies that calculates the relative positions of each cluster and assigns particle classes to each cluster.

![Project Logo](images/logo.png)
