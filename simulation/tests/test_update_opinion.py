import numpy as np
import unittest
import sys
import os
os.chdir(os.path.abspath(os.path.dirname(__file__)))
sys.path.append("..//functions")
from update_opinion import *
sys.path.append("../classes")
from simulation import Simulation
from node import Node


class TestStateInit(unittest.TestCase):
    def test_avg(self):
        vector_size=10
        opinion=np.ones((vector_size,1))
        info_t0=[np.ones((vector_size,1))]*5
        info_update=lambda x : x.get_info()
        opinion_update=coord_avg
        msg_generator=lambda x : [(1,np.zeros((vector_size,1)))]
        dist_matrix=np.array([[1,2],[4,1]])
        node=Node(1,opinion,info_t0,opinion_update,info_update,msg_generator,dist_matrix)    
        new_opinion = node.update_opinion()
        self.assertEqual(new_opinion.all(),np.ones((vector_size,1)).all())

    def test_avg_in_context(self):
        nb_nodes=100
        vector_size=10
        func_init=lambda N,n,m : [[np.ones((n,1)) for i in range(N)],[[np.zeros((m,1))] for i in range(N)]]
        info_update=lambda x : x.get_info()
        opinion_update=coord_avg
        msg_generator=lambda x : [(1,np.zeros((5,1)))]        
        simulation=Simulation(nb_nodes,func_init,opinion_update,info_update,msg_generator,vector_size,vector_size)
        for i in range(0,5):
            simulation.update()
    
    def test_biased_assimilation(self):
        vector_size=10
        opinion=np.ones((vector_size,1))
        info_t0=[np.ones((vector_size,1))]*5
        info_update=lambda x : x.get_info()
        opinion_update=lambda x :biased_assimilation(x,0.05,0.5)
        msg_generator=lambda x : [(1,np.zeros((vector_size,1)))]
        dist_matrix=np.array([[1,2],[4,1]])
        node=Node(1,opinion,info_t0,opinion_update,info_update,msg_generator,dist_matrix)    
        new_opinion = node.update_opinion()
        self.assertEqual(new_opinion.all(),np.ones((vector_size,1)).all())
    
    def test_biased_assimilation_in_context(self):
        nb_nodes=100
        vector_size=10
        func_init=lambda N,n,m : [[np.ones((n,1)) for i in range(N)],[[np.zeros((m,1))] for i in range(N)]]
        info_update=lambda x : x.get_info()
        opinion_update=lambda x :biased_assimilation(x,0.05,0.5)
        msg_generator=lambda x : [(1,np.zeros((5,1)))]        
        simulation=Simulation(nb_nodes,func_init,opinion_update,info_update,msg_generator,vector_size,vector_size)
        for i in range(0,5):
            simulation.update()

    

if __name__=='__main__':
    unittest.main()
        