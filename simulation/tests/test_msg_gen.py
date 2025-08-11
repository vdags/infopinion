import numpy as np
import unittest
import sys
import os
os.chdir(os.path.abspath(os.path.dirname(__file__)))
sys.path.append("..//functions")
from msg_gen import *
sys.path.append("..//classes")
from simulation import Simulation
from node import Node

class TestMsgGen(unittest.TestCase):
    def test_one_info_random_0(self):
        nb_nodes=100
        vector_size=10
        proba=1
        opinion=np.ones((vector_size,1))
        info_t0=[np.ones((vector_size,1))]*7
        info_update=lambda x : x.get_info()
        opinion_update=lambda x : x.get_opinion()
        msg_generator=lambda x : one_info_random(x,nb_nodes,proba)
        dist_matrix=np.array([[1,2],[4,1]])
        node=Node(1,opinion,info_t0,opinion_update,info_update,msg_generator,dist_matrix)    
        msgs = node.generate_msg()
        self.assertEqual(len(msgs),nb_nodes)
        for i in range(0,len(msgs)):
            self.assertGreaterEqual(msgs[i][0],0)
            self.assertLessEqual(msgs[i][0],nb_nodes-1)
            self.assertEqual(msgs[i][1].shape[0],vector_size)

    def test_one_info_random_1(self):
        nb_nodes=100
        vector_size=10
        proba=0
        opinion=np.ones((vector_size,1))
        info_t0=[np.ones((vector_size,1))]*7
        info_update=lambda x : x.get_info()
        opinion_update=lambda x : x.get_opinion()
        msg_generator=lambda x : one_info_random(x,nb_nodes,proba)
        dist_matrix=np.array([[1,2],[4,1]])
        node=Node(1,opinion,info_t0,opinion_update,info_update,msg_generator,dist_matrix)    
        msgs = node.generate_msg()
        self.assertEqual(len(msgs),0)
        for i in range(0,len(msgs)):
            self.assertGreaterEqual(msgs[i][0],0)
            self.assertLessEqual(msgs[i][0],nb_nodes-1)
            self.assertEqual(msgs[i][1].shape[0],vector_size)
        
    def test_one_info_random_2(self):
        nb_nodes=10
        vector_size=10
        proba=1/(nb_nodes) #ensure to have a connexe graph at each timestep (property of Erdos-Rainyi graphs)
        opinion=np.ones((vector_size,1))
        info_t0=[np.ones((vector_size,1))]*7
        info_update=lambda x : x.get_info()
        opinion_update=lambda x : x.get_opinion()
        msg_generator=lambda x : one_info_random(x,nb_nodes,proba)
        dist_matrix=np.array([[1,2],[4,1]])
        node=Node(1,opinion,info_t0,opinion_update,info_update,msg_generator,dist_matrix)    
        msgs = node.generate_msg()
        #print(msgs)
        for i in range(0,len(msgs)):
            self.assertGreaterEqual(msgs[i][0],0)
            self.assertLessEqual(msgs[i][0],nb_nodes-1)
            self.assertEqual(msgs[i][1].shape[0],vector_size)

    def test_one_info_random_in_context(self):
        nb_nodes=100
        vector_size=10
        link_proba=1/nb_nodes
        func_init=lambda N,n,m : [[np.ones((n,1)) for i in range(N)],[[np.zeros((m,1))] for i in range(N)]]
        info_update=lambda x : x.get_info()
        opinion_update=lambda x: x.get_opinion()
        msg_generator=lambda x : one_info_random(x,nb_nodes,link_proba)       
        simulation=Simulation(nb_nodes,func_init,opinion_update,info_update,msg_generator,vector_size,vector_size)
        for i in range(0,5):
            simulation.update()

    def test_one_info_dist_global(self):
        nb_nodes=100
        vector_size=10
        proba=1/(nb_nodes) #ensure to have a connexe graph at each timestep (property of Erdos-Rainyi graphs)
        opinion=np.ones((vector_size,1))
        info_t0=[np.ones((vector_size,1))]*7
        info_update=lambda x : x.get_info()
        opinion_update=lambda x : x.get_opinion()
        msg_generator=lambda x : one_info_dist_global(x,nb_nodes,proba)
        dist_matrix=np.ones((nb_nodes,nb_nodes))
        node=Node(1,opinion,info_t0,opinion_update,info_update,msg_generator,dist_matrix)    
        msgs = node.generate_msg()
        #print(msgs)
        for i in range(0,len(msgs)):
            self.assertGreaterEqual(msgs[i][0],0)
            self.assertLessEqual(msgs[i][0],nb_nodes-1)
            self.assertEqual(msgs[i][1].shape[0],vector_size)


    def test_one_info_dist_global_in_context(self):
        nb_nodes=100
        vector_size=10
        link_proba=1/nb_nodes
        func_init=lambda N,n,m : [[np.ones((n,1)) for i in range(N)],[[np.zeros((m,1))] for i in range(N)]]
        info_update=lambda x : x.get_info()
        opinion_update=lambda x: x.get_opinion()
        msg_generator=lambda x : one_info_dist_global(x,nb_nodes,link_proba)       
        simulation=Simulation(nb_nodes,func_init,opinion_update,info_update,msg_generator,vector_size,vector_size)
        simulation.graph.update_dist_matrix(-1,"euclidean") #needed by the msg generation function
        for i in range(0,5):
            simulation.update()



if __name__=='__main__':
    unittest.main()
        