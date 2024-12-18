import numpy as np
import pandas as pd


class GetParticleClass:
    
    def __init__(self,centroids):
        
        self.centroids = centroids
        assert isinstance(self.centroids, pd.DataFrame), "centroids must be a pandas DataFrame"
        self.centroids['Particle Label'] = None
        self.Xrel_positions = np.zeros((len(self.centroids), len(self.centroids)))
        self.Yrel_positions = np.zeros((len(self.centroids), len(self.centroids)))
    
    def classify_particle(self):
        
        Xpos = self.centroids['Xpos']
        Ypos = self.centroids['Ypos']
        
        for i in range(len(self.centroids)):
            for j in range(len(self.centroids)):
                # build matrices that hold the relative distances to each group
                self.Xrel_positions[i, j] = Xpos[j] - Xpos[i]  # X-axis difference
                self.Yrel_positions[i, j] = Ypos[j] - Ypos[i]  # Y-axis difference 
        
                # Set conditions that need to be met to be classified as a proton
                Proton_conditions = (
                    (sum(self.Xrel_positions[i,:] < 0) >= len(self.centroids)-1 and # Protons are the lowest-right (highest x-position, lowest y-position)
                    Ypos[i] < 1100 and
                    sum(self.Yrel_positions[i,:] > 0) >= len(self.centroids)-1) or
        
                    sum(self.Xrel_positions[i,:] == -np.max(self.Xrel_positions)) == 1 # It is usually the furthest group, others are bunched up
                )
        
                # Set conditions that need to be met to be classified as an alpha particle
                Alpha_conditions = ( 
                    sum(self.Yrel_positions[i,:] < 0) == len(self.centroids)-1 and # Highest point and usually in the corner
                    Xpos[i] < 800 and 
                    Ypos[i] > 1000
                    
                )
                # Set conditions that need to be met to be classified as a triton
                Triton_conditions = (
                    (self.centroids['Particle Label'][j] == "Alphas" and # It is below alphas if alphas are defined
                    self.Yrel_positions[i,j] > 0 and            
                    self.centroids['Particle Label'][j] == "Deuterons" and  # It is above deuterons if deuterons are defined
                    self.Xrel_positions[i,j] > 0 and self.Yrel_positions[i,j] < 0) or
        
                    (len(self.centroids) == 4 and  # If there are 4 clusters, then it is the 2nd highest in the y
                    sum(self.Yrel_positions[i,:] < 0) >= 2 and
                    sum(self.Yrel_positions[i,:] > 0) >= 1 and
                    sum(self.Xrel_positions[i,:] > 0) >= len(self.centroids)-1) or # it is usually the left-most group (need to confirm)
        
                    # If there are 3 groups and the proton group is not included
                    (len(self.centroids) == 3 and
                    sum(self.Yrel_positions[i,:] < 0) >= 1 and
                    sum(self.Yrel_positions[i,:] > 0) >= 1 and
                    sum(self.Xrel_positions[i,:] > 0) >= len(self.centroids)-1)        
                )
                
                # Set conditions that need to be met to be classified as a deuteron
                Deuteron_conditions = (
                    
                    (self.centroids['Particle Label'][j] == "Alphas" and # It is below alphas if they are defined
                    self.Yrel_positions[i,j] > 0 and            
                    self.centroids['Particle Label'][j] == "Tritons" and  # It is below and to the right of tritons if they are defined
                    self.Xrel_positions[i,j] < 0 and
                    self.Yrel_positions[i,j] > 0 and
                    self.centroids['Particle Label'][j] == "Protons" and  # It is above and to the left of protons if they are defined
                    self.Xrel_positions[i,j] > 0 and
                    self.Yrel_positions[i,j] < 0) or
        
                    # If there are three groups and tritons or alphas are missing
                    (len(self.centroids) == 3 and
                    sum(self.Yrel_positions[i,:] < 0) >= 1 and # it becomes the middle group
                    sum(self.Yrel_positions[i,:] > 0) >= 1) or
                    
                    # If there are three groups and the protons are missing
                    (len(self.centroids) == 3 and
                    sum(self.Yrel_positions[i,:] < 0) >= 2 and
                    sum(self.Xrel_positions[i,:] < 0) >= 2) or
        
                    # If there are four groups, it is the 2nd farthest to the right
                    (len(self.centroids) == 4 and  
                    sum(self.Xrel_positions[i,:] < 0) >= 2 and
                    sum(self.Xrel_positions[i,:] > 0) >= 1 and
                    sum(self.Yrel_positions[i,:] > 0) >= 2 and
                    sum(self.Yrel_positions[i,:] < 0) >= 1) 
                    
                        
                    )
        
                if Proton_conditions:
                    self.centroids.loc[i, 'Particle Label'] = "Protons"
                elif Alpha_conditions:
                    self.centroids.loc[i, 'Particle Label'] = "Alphas"
                elif Triton_conditions:
                    self.centroids.loc[i, 'Particle Label'] = "Tritons"
                elif Deuteron_conditions:
                    self.centroids.loc[i, 'Particle Label'] = "Deuterons"

        return self.centroids


   
'''
def get_particle_class(centroids):
    
    # initialize a new column to put labels in later
    centroids['Particle Label'] = None
    
    # initialize arrays for for loop
    Xrel_positions = np.zeros((len(centroids), len(centroids)))
    Yrel_positions = np.zeros((len(centroids), len(centroids)))
    Xpos = centroids['Xpos']
    Ypos = centroids['Ypos']
    
    for i in range(len(centroids)):
        for j in range(len(centroids)):
            # build matrices that hold the relative distances to each group
            Xrel_positions[i, j] = Xpos[j] - Xpos[i]  # X-axis difference
            Yrel_positions[i, j] = Ypos[j] - Ypos[i]  # Y-axis difference 
    
            # Set conditions that need to be met to be classified as a proton
            Proton_conditions = (
                (sum(Xrel_positions[i,:] < 0) >= len(centroids)-1 and # Protons are the lowest-right (highest x-position, lowest y-position)
                Ypos[i] < 1100 and
                sum(Yrel_positions[i,:] > 0) >= len(centroids)-1) or
    
                sum(Xrel_positions[i,:] == -np.max(Xrel_positions)) == 1 # It is usually the furthest group, others are bunched up
            )
    
            # Set conditions that need to be met to be classified as an alpha particle
            Alpha_conditions = ( 
                sum(Yrel_positions[i,:] < 0) == len(centroids)-1 and # Highest point and usually in the corner
                Xpos[i] < 800 and 
                Ypos[i] > 1000
                
            )
            # Set conditions that need to be met to be classified as a triton
            Triton_conditions = (
                (centroids['Particle Label'][j] == "Alphas" and # It is below alphas if alphas are defined
                Yrel_positions[i,j] > 0 and            
                centroids['Particle Label'][j] == "Deuterons" and  # It is above deuterons if deuterons are defined
                Xrel_positions[i,j] > 0 and Yrel_positions[i,j] < 0) or
    
                (len(centroids) == 4 and  # If there are 4 clusters, then it is the 2nd highest in the y
                sum(Yrel_positions[i,:] < 0) >= 2 and
                sum(Yrel_positions[i,:] > 0) >= 1 and
                sum(Xrel_positions[i,:] > 0) >= len(centroids)-1) or # it is usually the left-most group (need to confirm)
    
                # If there are 3 groups and the proton group is not included
                (len(centroids) == 3 and
                sum(Yrel_positions[i,:] < 0) >= 1 and
                sum(Yrel_positions[i,:] > 0) >= 1 and
                sum(Xrel_positions[i,:] > 0) >= len(centroids)-1)        
            )
            
            # Set conditions that need to be met to be classified as a deuteron
            Deuteron_conditions = (
                
                (centroids['Particle Label'][j] == "Alphas" and # It is below alphas if they are defined
                Yrel_positions[i,j] > 0 and            
                centroids['Particle Label'][j] == "Tritons" and  # It is below and to the right of tritons if they are defined
                Xrel_positions[i,j] < 0 and
                Yrel_positions[i,j] > 0 and
                centroids['Particle Label'][j] == "Protons" and  # It is above and to the left of protons if they are defined
                Xrel_positions[i,j] > 0 and
                Yrel_positions[i,j] < 0) or
    
                # If there are three groups and tritons or alphas are missing
                (len(centroids) == 3 and
                sum(Yrel_positions[i,:] < 0) >= 1 and # it becomes the middle group
                sum(Yrel_positions[i,:] > 0) >= 1) or
                
                # If there are three groups and the protons are missing
                (len(centroids) == 3 and
                sum(Yrel_positions[i,:] < 0) >= 2 and
                sum(Xrel_positions[i,:] < 0) >= 2) or
    
                # If there are four groups, it is the 2nd farthest to the right
                (len(centroids) == 4 and  
                sum(Xrel_positions[i,:] < 0) >= 2 and
                sum(Xrel_positions[i,:] > 0) >= 1 and
                sum(Yrel_positions[i,:] > 0) >= 2 and
                sum(Yrel_positions[i,:] < 0) >= 1) 
                
            )
            
            if Proton_conditions:
                centroids.loc[i,'Particle Label'] = "Protons"
            
            if Alpha_conditions:
                centroids.loc[i,'Particle Label'] = "Alphas"
    
            if Triton_conditions:
                centroids.loc[i,'Particle Label'] = "Tritons"
    
            if Deuteron_conditions:
                centroids.loc[i,'Particle Label'] = "Deuterons"
    
            # If no particles are detected, label group as 'Not a Particle'. 
            # Can be a good backup in case the clustering doesn't work well.
    # if Proton_conditions and Alpha_conditions and Triton_conditions and Deuteron_conditions:
    #     None
    # else:
    #     centroids.loc[i,'Particle Label'] = "Not a particle group"
                
    return centroids
    
# particles = get_particle_class(centroids)
# print("Relative Positions (i, j, [dy, dx]):\n", Yrel_positions)
# print(particles)
'''