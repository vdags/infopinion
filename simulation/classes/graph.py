import numpy as np
from node import Node
from sklearn.metrics import pairwise_distances
class Graph:
    def __init__(self,nb_nodes,opinions_0,infos_0,update_opinion,update_info,generate_msg,metric='euclidean'):
        """
        nb_nodes: int, number of nodes of the simulation
        opinions: list(np.array(n,1)), list of opinion vector for nodes at time 0
        infos: list(list(np.array(m,1))), list of list of pieces of information available for each node at time 0
        update_info: func(list(np.array(m,1)),list(np.array(n,1)),np.array(n,1))--> np.array(n,1) , update info vector based on previous info and opinion states and recieved msg
        update_opinion: func(list(np.array(m,1)),list(np.array(n,1)))--> np.array(m,1) , update opinion vector based on previous info and opinion states and recieved msg
        generate_msg: func(list(np.array(m,1)),list(np.array(n,1)))--> list(Node,np.array(n,1)) , genrate a list of couple of Node and msg based on info and opinion history
        """
        dist_mat = pairwise_distances(np.concatenate(opinions_0,axis=1).transpose(),metric=metric)        

        self.nodes=[ Node(i,
                          opinions_0[i],
                            infos_0[i],
                            update_opinion,
                            update_info,
                            generate_msg,
                            dist_mat) for i in range(0,nb_nodes) ]
        self.size=nb_nodes
        self.dist_mat=np.zeros((0,np.shape(dist_mat)[0],np.shape(dist_mat)[0])) #TODO test
        self.dist_mat=np.append(self.dist_mat,np.array([dist_mat]),axis=0)



    def get_node(self,id):
        return self.nodes[id]
    
    def get_dist_matrix(self,t=-1,metric="euclidean"):
        """
        Return the distance matrix of the opinion's vectors at timestep t. 
        If no timestep is given, return matrix for latest timestep. 
        Acceptable metrics are those from sklearn.metrics.pairwise_distances.
        It is used as a distance between nodes.
        """
        opinions=np.array([i.opinions[t][:,0] for i in self.nodes])
        # graph contains always at least initial vectors, so that -1 index is always valid.
        distmat = pairwise_distances(opinions,metric=metric)
        return distmat
        
    def update_dist_matrix(self,t=-1,metric="euclidean"):
        dist_mat=self.get_dist_matrix(t,metric)
        self.dist_mat=np.append(self.dist_mat,np.array([dist_mat]),axis=0) #TODO test
        for i in self.nodes:
            i.set_dist_matrix(dist_mat)

    def get_ndi(self,t=-1):
        """
        Return Network Disagreement Index (NDI) for each coordinate of opinion's vectors of the node part of the graph at a given timestep t.

        t: int, timestep on which to calculate NDI

        Return:
        np.array((self.nodes[0].get_opinion().shape[0],1))
        """
        ndi=np.zeros((self.nodes[0].get_opinion().shape[0],1))
        for i in range(0,self.size):
            for j in range(0,self.size):
                if i!=j and self.dist_mat[t,i,j]>10**(-30):
                    tmp = ((self.dist_mat[t,i,j])**(-1) ) *(self.nodes[i].get_opinion(t)-self.nodes[j].get_opinion(t))**2 # MODIFIED alternative for non conservation of distance matrix (for calculus efficiency)
                    if tmp.mean()== np.float64(np.inf):
                        print("ZERO!!!")
                    #tmp = ((self.dist_mat[t,i,j])**(-1) ) *(self.nodes[i].get_opinion(t)-self.nodes[j].get_opinion(t))**2 # MODIFIED alternative for non conservation of distance matrix (for calculus efficiency)
                    tmp[tmp<10**(-10)]=0
                    ndi+= tmp
        return ndi
        
    def get_gdi(self,t=-1): #TODO test
        """
        Return Global Disagreement Index (GDI) for each coordinate of opinion's vectors of the node part of the graph at a given timestep t.

        t: int, timestep on which to calculate GDI

        Return:
        np.array((self.nodes[0].get_opinion().shape[0],1))
        """
        gdi=np.zeros((self.nodes[0].get_opinion().shape[0],1))
        for i in range(0,self.size):
            for j in range(0,i):
                gdi+= (self.nodes[i].get_opinion(t)-self.nodes[j].get_opinion(t))**2
        return gdi