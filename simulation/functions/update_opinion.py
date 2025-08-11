"""
This file contains several alternative functions for updating opinion vector through time.

For opinion updating functions:

 list(np.array(n,1)), list(list(np.array(m,1))),np.array(n,1))--> np.array(n,1) , 
 
i.e. update opinion vector based on all previous opinion and info states and and last recieved msg.
Older msg are not considered as input since they are already taken into account for previous opinion state update.

"""

from copy import deepcopy
import numpy as np

def opinion_dummy(x,y):
    return  x[-1]

def coord_avg(node,coef=0.05):
    """
    Return an average of previous opinion position and current pices of information currently in info.
    The ponderation coefficient is a cst for all values in info. Due to "1-hot"-like encoding, 0 values 
    are considered as non values, therefore not considered for averaging.
    
    coef: float, coefficient of ponderation of the average for info elements. The vector from opnion is considered with a coefficient of 1
    """
    coef=float(coef)
    opinions=node.get_opinion()
    info=node.get_info_all()
    new_opinion=deepcopy(opinions[:])
    for i in range(0,len(new_opinion)):
        c=0
        for j in range(0,len(info[-1])):
            if info[-1][j][i][0]!=0.: 
                new_opinion[i]=new_opinion[i] + coef*info[-1][j][i]
                c+=1
        new_opinion[i]=new_opinion[i]/(1+c*coef)
    return new_opinion

def biased_assimilation(node,coef,bias):
    """
    Return a biased average of previous opinion position and current pices of information currently in info.
    This formula is the adaptation of Dandekar et al. (2013) biased assimilation to this context. 
    Graph is considered as constant weighted to coef.
    
    The ponderation coefficient is a cst for all values in info. Due to "1-hot"-like encoding, 0 values are 
    considered as non values, therefore not considered for averaging.

    coef: float, coefficient of ponderation of the average for info elements. The vector from opnion is considered with a coefficient of 1
    bias: float in [0,1], Dandekar et al. bias coefficient.
    """
    coef=float(coef)
    bias=float(bias)
    opinions=node.get_opinion()
    info=node.get_info()
    new_opinion=deepcopy(opinions[:])
    for i in range(0,len(new_opinion)):
        c=0
        val_info=[]
        for j in range(0,len(info)):
            #print(info[j][i])
            if info[j][i][0]!=0.: 
                val_info.append(info[j][i][0])
                c+=1
        D=coef*c
        S=float(sum(coef*np.array(val_info)))
        new_opinion[i]=(1*new_opinion[i] + ((new_opinion[i])**bias)*S)/(
            1+((new_opinion[i])**bias)*S + ((1-new_opinion[i])**bias)*(D-S))
    return new_opinion