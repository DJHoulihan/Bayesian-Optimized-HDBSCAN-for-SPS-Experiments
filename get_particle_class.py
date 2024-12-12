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
            Yrel_positions[i, j] = Y
            pos[j] - Ypos[i]  # Y-axis difference 
    
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
