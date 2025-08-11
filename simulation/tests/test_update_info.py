import numpy as np
import unittest
import sys
import os
os.chdir(os.path.abspath(os.path.dirname(__file__)))
sys.path.append("..//functions")
from update_info import *
sys.path.append("../classes")
from simulation import Simulation
from node import Node

class TestUpdateInfo(unittest.TestCase):
    def test_latest_info_arbitrary_0(self):
        vector_size=10
        opinion=[np.ones((vector_size,1))]
        info_t0=[np.ones((vector_size,1))]*3
        info_update=latest_info_arbitrary
        opinion_update=lambda x : x.get_opinion()
        msg_generator=lambda x : [(1,np.zeros((vector_size,1)))]
        dist_matrix=np.array([[1,2],[4,1]])
        node=Node(1,opinion,info_t0,opinion_update,info_update,msg_generator,dist_matrix)    
        recieved_msg=[(0,np.ones((vector_size,1))),(0,np.ones((vector_size,1)))]
        node.recieved=[recieved_msg]
        info_t1 = node.update_info()
        self.assertEqual(len(info_t1),5)
    
    def test_latest_info_arbitrary_1(self):
        vector_size=10
        opinion=[np.ones((vector_size,1))]
        info_t0=[np.ones((vector_size,1))]*6
        info_update=latest_info_arbitrary
        opinion_update=lambda x : x.get_opinion()
        msg_generator=lambda x : [(1,np.zeros((vector_size,1)))]
        dist_matrix=np.array([[1,2],[4,1]])
        node=Node(1,opinion,info_t0,opinion_update,info_update,msg_generator,dist_matrix)    
        recieved_msg=[(0,np.ones((vector_size,1))),(0,np.ones((vector_size,1)))]
        node.recieved=[recieved_msg]
        info_t1 = node.update_info()
        self.assertEqual(len(info_t1),7)
        
    
    def test_latest_info_arbitrary_2(self):
        vector_size=10
        opinion=[np.ones((vector_size,1))]
        info_t0=[np.ones((vector_size,1))]*7
        info_update=latest_info_arbitrary
        opinion_update=lambda x : x.get_opinion()
        msg_generator=lambda x : [(1,np.zeros((vector_size,1)))]
        dist_matrix=np.array([[1,2],[4,1]])
        node=Node(1,opinion,info_t0,opinion_update,info_update,msg_generator,dist_matrix)    
        recieved_msg=[(0,np.ones((vector_size,1))),(0,np.ones((vector_size,1)))]
        node.recieved=[recieved_msg]
        info_t1 = node.update_info()
        self.assertEqual(len(info_t1),7)
    
    def test_latest_info_random_0(self):
        vector_size=10
        opinion=[np.ones((vector_size,1))]
        info_t0=[np.ones((vector_size,1))]*3
        info_update=latest_info_random
        opinion_update=lambda x : x.get_opinion()
        msg_generator=lambda x : [(1,np.zeros((vector_size,1)))]
        dist_matrix=np.array([[1,2],[4,1]])
        node=Node(1,opinion,info_t0,opinion_update,info_update,msg_generator,dist_matrix)    
        recieved_msg=[(0,np.ones((vector_size,1))),(0,np.ones((vector_size,1)))]
        node.recieved=[recieved_msg]
        info_t1 = node.update_info()
        self.assertEqual(len(info_t1),5)
        
        self.assertEqual(info_t1[1].shape[0],vector_size)
        self.assertEqual(info_t1[1].shape[1],1)
    
    def test_latest_info_random_1(self):
        vector_size=10
        opinion=[np.ones((vector_size,1))]
        info_t0=[np.ones((vector_size,1))]*6
        info_update=latest_info_random
        opinion_update=lambda x : x.get_opinion()
        msg_generator=lambda x : [(1,np.zeros((vector_size,1)))]
        dist_matrix=np.array([[1,2],[4,1]])
        node=Node(1,opinion,info_t0,opinion_update,info_update,msg_generator,dist_matrix)    
        recieved_msg=[(0,np.ones((vector_size,1))),(0,np.ones((vector_size,1)))]
        node.recieved=[recieved_msg]
        info_t1 = node.update_info()
        self.assertEqual(len(info_t1),7)
    
    def test_latest_info_random_2(self):
        vector_size=10
        opinion=[np.ones((vector_size,1))]
        info_t0=[np.ones((vector_size,1))]*7
        info_update=latest_info_random
        opinion_update=lambda x : x.get_opinion()
        msg_generator=lambda x : [(1,np.zeros((vector_size,1)))]
        dist_matrix=np.array([[1,2],[4,1]])
        node=Node(1,opinion,info_t0,opinion_update,info_update,msg_generator,dist_matrix)    
        recieved_msg=[(0,np.ones((vector_size,1))),(0,np.ones((vector_size,1)))]
        node.recieved=[recieved_msg]
        info_t1 = node.update_info()
        self.assertEqual(len(info_t1),7)
    
    def test_latest_info_arbitrary_in_context(self):
        nb_nodes=100
        vector_size=10
        func_init=lambda N,n,m : [[np.ones((n,1)) for i in range(N)],[[np.zeros((m,1))] for i in range(N)]]
        info_update=latest_info_arbitrary
        opinion_update=lambda x : x.get_opinion()
        msg_generator=lambda x : [(1,np.zeros((5,1)))]        
        simulation=Simulation(nb_nodes,func_init,opinion_update,info_update,msg_generator,vector_size,vector_size)
        for i in range(0,5):
            simulation.update()
    
    def test_latest_info_random_in_context(self):
        nb_nodes=100
        vector_size=10
        func_init=lambda N,n,m : [[np.ones((n,1)) for i in range(N)],[[np.zeros((m,1))] for i in range(N)]]
        info_update=latest_info_random
        opinion_update=lambda x : x.get_opinion()
        msg_generator=lambda x : [(1,np.zeros((5,1)))]
        simulation=Simulation(nb_nodes,func_init,opinion_update,info_update,msg_generator,vector_size,vector_size)
        for i in range(0,5):
            simulation.update()

if __name__=='__main__':
    unittest.main()
        