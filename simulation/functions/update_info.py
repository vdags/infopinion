"""
This file contains several alternative functions for updating information list of vector through time.

For information updating functions:

    list(np.array(n,1)), list(list(np.array(m,1))),np.array(n,1))--> list(np.array(m,1))

i.e. return info vector's list based on all previous opinion and info states and and last recieved msg. 
Older msg are not considered as input since they are already taken into account for previous opinion state update.


"""
from random import shuffle
from copy import deepcopy

def info_dummy(x,y,z):
    return y[-1]

def latest_info_arbitrary(node,n=7):
    """
    This function keeps n last msg as all pieces of information. 
    If there is more than 7 messages recieved by a node at a given timestep, 7 of them are choosen arbitrary according to msgs input order.
    
    node: <Node> object
    n: int, nb of pieces of information to be conserved by a node. 
        Default value is 7
    
    For other args, see top of update.py
    """
    infos=node.get_info()
    msgs=node.get_recieved()
    #print(infos,"\n",msgs)
    memory=[]
    memory+=deepcopy(infos)
    memory+=[i[1] for i in msgs]
    if len(memory)<=n:
        return memory
    else:
        return memory[-7:]
    
def latest_info_random(node,n=7):
    """
    This function keeps n last msg as all pieces of information.
    If there is more than 7 messages recieved by a node at a given timestep, 7 of them are randomly selected.
    
    n: int, nb of pieces of information to be conserved by a node. 
        Default value is 7
    
    For other args, see top of update.py
    """
    infos=node.get_info()
    msgs=node.get_recieved()
    memory=[]
    msgs_cleared=deepcopy([i[1] for i in msgs])
    #print(msgs_cleared)
    memory+=deepcopy(infos)
    memory+=msgs_cleared
    if len(memory)<=n:
        return memory
    elif len(msgs)>n:
        memory=deepcopy(msgs_cleared)
        shuffle(memory)
        return memory[-7:]
    else:
        return memory[-7:]