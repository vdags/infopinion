import numpy as np

"""
The functions below are several alternative for the initialisation of information 
and opinion states of the newtwork.

    (N,n,m) -> ( list(np.array(n,1)), list( list(np.array(m,1)) ) )

where N is the number of nodes,
    n and m are resp. encoding size of opinions and information.

I.e. the output tuple can be interpreted by:

    (list(opinion_0) , list(inf_0))

opinion_0: np.array(n,1), opinion vector of a given node at time 0
info_0: list(np.array(m,1)), list of pieces of information available at time 0 for a given node
"""

def dummy(N,n=10,m=5,o0=1,i0=0):
    return [[np.full((n,1),o0) for i in range(N)],[[np.full((m,1),i0)] for i in range(N)]]


def independent_cst_normal_distribution(N,n=10,m=5):
    """
    Return cst vectors which cst value follows a normal distribution centered on 0.5 with sdt of 0.15.

    N: number of nodes
    n: opinion's vector size
    m: info's vector size
    """
    #cst_info=np.random.normal(loc=0,scale=0.3,size=(1,N))
    #cst_opinion=np.random.normal(loc=0,scale=0.3,size=(1,N))
    return ( list(map(lambda x : np.full((n,1),x),np.random.normal(loc=0.5,scale=0.15,size=(1,N)).tolist()[0])),
             list(map(lambda x : [np.full((m,1),x)],np.random.normal(loc=0.5,scale=0.15,size=(1,N)).tolist()[0]))
    )
   
             
def independent_normal_distribution(N,n=10,m=5):
    """
    Return cst vectors which cst value follows a quasi-normal distribution centered on 0.5 with sdt of 0.15.
    See quasi_gaussian_segment_generator doc for more detail.

    N: number of nodes
    n: opinion's vector size
    m: info's vector size
    """
    return ( list(map(lambda x : quasi_gaussian_segment_generator((n,1),0,1), range(0,N))),
             list(map(lambda x : [quasi_gaussian_segment_generator((m,1),0,1)],range(0,N)))
    )

def independent_uniform_distribution(N,n=10,m=5):
    """
    Return cst vectors which cst value follows a uniform distribution between 0 and 1.

    N: number of nodes
    n: opinion's vector size
    m: info's vector size
    """
    return ( list(map(lambda x :  np.random.uniform(low=0,high=1,size=(n,1)), range(0,N))),
             list(map(lambda x : [np.random.uniform(low=0,high=1,size=(m,1))],range(0,N)))
    )

def quasi_gaussian_segment_generator(size,low=0,high=1):
    """
    Return vectors which coordinates are following a normal distribution between 'low' and 'high' 
    boudaries with a mean at (low+high)/2 and std of (high-low)/6 such that 
    99.7% values generated are gaussian. For values not complying with boudaries,
    they are respectively set to 'low' and 'high' if they are lower or higher to the defined boudaries.
    
    requires low<high

    size: (int,int), tuple of the dimension of the vector
    low:lower bound
    high: higher bound

    Returns: np.array(size)
    """
    vec = np.random.normal(loc=(low+high)/2,scale=(high-low)/6,size=size)
    f = np.vectorize(lambda x: low if x<low else (x if x<high else high))
    vec=f(vec)
    return vec


#def bivariated_and_centered(N,n=10,m=5,oc0=1,ic0=0,)