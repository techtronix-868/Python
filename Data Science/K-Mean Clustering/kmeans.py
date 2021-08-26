from cluster import *
from point import *

def kmeans(pointdata, clusterdata) :
    #Fill in
    
    #1. Make list of points using makePointList and pointdata
    pointsList = makePointList(pointdata)


    #2. Make list of clusters using createClusters and clusterdata

    clusterList = createClusters((clusterdata))

    #3. As long as points keep moving:
    check = True
    i = 0
    while(check):
        check = False
        for j in pointsList:
            # Or statement for check and move to cluster
            check = check | j.moveToCluster(j.closest(clusterList))
        for k in clusterList:
            k.updateCenter()
        i = i + 1
    
        #A. Move every point to its closest cluster (use Point.closest and
        #   Point.moveToCluster)
        #   Hint: keep track here whether any point changed clusters by
        #         seeing if any moveToCluster call returns "True"
        
        #B. Update the centers for each cluster (use Cluster.updateCenter)
            
    #4. Return the list of clusters, with the centers in their final positions
    return clusterList
    
    
    
if __name__ == '__main__' :
    data = np.array([[0.5, 2.5], [0.3, 4.5], [-0.5, 3], [0, 1.2], [10, -5], [11, -4.5], [8, -3]], dtype=float)
    centers = np.array([[0, 0], [1, 1]], dtype=float)
    
    clusters = kmeans(data, centers)
    for c in clusters :
        c.printAllPoints()
