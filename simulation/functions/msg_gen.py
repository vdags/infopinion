import numpy as np
import random as rd
from sklearn.metrics import pairwise_distances
from math import atan,pi

"""
This file content all message generating functions.
A node i can send at each timestep a finite number of messages to a subset of nodes.

Each msg generator has to comply with:
    
    (list(np.array(n,1)),list(np.array(m,1)))--> list(Node,np.array(m,1)) 

where n and m are resp. encoding size of opinions and information. 
    Note that msg are pieces of informations, as such they are of the same type as internal information vectors, therefore of size m.
       
i.e. this is interpreted as (list(opinion) , list(info)) -->  list(Node,msg) 

where list(opinion)  is the list of all opinion vectors of a given node trough time.
      list(info) is the list of all opinion vectors of a given node trough time.

In other words, it genrates a list of couple of a Node and a msg based on info and opinion history of the aforementioned Node.
"""
def mapords_np(L):
    """
    Return the ordinal matrix of L.
    """
    s = np.argsort(L)
    o = -np.ones(s.size, dtype=np.int32)
    o[s] = np.arange(s.size)
    return o

def dummy(x,y):
    return [(1,np.zeros((5,1)))]

def one_info_random(node,nb_nodes,link_proba):
    """
    Randomly assign a message  from node i to node j. 
    This message is a 0 vector except one randomly choosen coordinate which is the corresponding opinion value of i.
    
    //!\\ opnions and info must be the same size. //!\\ 

    link_proba: float in [0,1], proba for a link between two nodes to exist.
    """
    list_op=node.get_opinion()
    list_info=node.get_info()
    if len(list_op)!=len(list_info[0]):
      print("Error: opnions and info must be the same size.")
      return None
    msg_list=[]
    for i in range(0,nb_nodes):
        if rd.random()<=link_proba:
            reciever=i # node sending a msg to itself is not excluded.
            info_choosen=rd.randint(0,len(list_op)-1)
            msg_list.append((reciever,np.zeros((len(list_op),1))))
            msg_list[-1][1][info_choosen]=list_op[info_choosen]
    return msg_list

def one_info_dist_global(node,nb_nodes,link_ratio):
    #list_op,list_info,node_obj,distance_matrix,
    """
    Assign a node i to a node j based on their pairwise distance. 
    The consequently probability distribution is (2/pi)*arctan(1/d) where d is the distance from i to j. 
    The subject of conversation (msg coord) is randomly assigned.

    //!\\ opnions and info must be the same size. //!\\ 

    link_ratio: float between 0 and 1 defining the proportion of links in the graph nodes to be set among the possible to set.
    metric: str, metric used as pairwise distance. Has to be supported by sklearn.metrics.pairwise_distances
    """
    list_op=node.get_opinion()
    list_info=node.get_info()
    distance_matrix=node.get_dist_matrix()
    if len(list_op)!=len(list_info[0]):
        print("Error: opnions and info must be the same size.")
        return None    
    msg_list=[]  
    sender=node.id
    for i in range(0,nb_nodes):
        reciever=i# node sending a msg to itself is not excluded.
        if distance_matrix[sender,reciever]==0 or 2*atan(distance_matrix[sender,reciever])/pi <= link_ratio: 
        #If distmat[sender,reciever]==0, it is considered as prolongation by continuity of the probability law used here, i.e. the probability equals 1.
            info_choosen=rd.randint(0,len(list_op)-1)
            #print(np.zeros((len(list_op[0]),1)))
            msg_list.append((reciever,np.zeros((len(list_op),1))))
            #print(msg_list[i],list_op[i])
            msg_list[-1][1][info_choosen]=list_op[info_choosen]
    return msg_list
