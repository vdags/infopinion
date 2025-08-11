import numpy as np
import unittest
import sys
import os
os.chdir(os.path.abspath(os.path.dirname(__file__)))
sys.path.append("..\\classes")
from node import Node
from graph import Graph

class TestGraph(unittest.TestCase):
    vector_size=10
    opinions=[np.ones((vector_size,1)),np.zeros((vector_size,1))]
    infos=[[np.ones((vector_size,1))]*5]*2
    #opinions=[np.ones((vector_size,1)),np.ones((vector_size,1))]
    #infos=[[np.ones((vector_size,1))],[np.ones((vector_size,1))]]
    info_update=lambda x : x.get_info()
    opinion_update=lambda x : x.get_opinion()
    msg_generator=lambda x : [(1,np.zeros((vector_size,1)))]
    graph=Graph(2,opinions,infos,opinion_update,info_update,msg_generator)

    def test___init__(self):
        vector_size=10
        opinions=[np.ones((vector_size,1))]*2
        infos=[[np.ones((vector_size,1))]*5]*2
        #opinions=[np.ones((vector_size,1)),np.ones((vector_size,1))]
        #infos=[[np.ones((vector_size,1))],[np.ones((vector_size,1))]]
        info_update=lambda x : x.get_info()
        opinion_update=lambda x : x.get_opinion()
        msg_generator=lambda x : [(1,np.zeros((vector_size,1)))]
        graph=Graph(2,opinions,infos,opinion_update,info_update,msg_generator)
        self.assertEqual(len(graph.nodes),2)

    def test_get_node(self):
        self.assertEqual(TestGraph.graph.get_node(0).__class__.__name__,"Node")

    def test_get_dist_matrix(self):
        self.assertEqual(TestGraph.graph.get_dist_matrix().shape,(2,2))
    
    def test_update_dist_matrix(self):
        vector_size=10
        opinions=[np.ones((vector_size,1))]*2
        infos=[[np.ones((vector_size,1))]*5]*2
        #opinions=[np.ones((vector_size,1)),np.ones((vector_size,1))]
        #infos=[[np.ones((vector_size,1))],[np.ones((vector_size,1))]]
        info_update=lambda x : x.get_info()
        opinion_update=lambda x : x.get_opinion()
        msg_generator=lambda x : [(1,np.zeros((vector_size,1)))]
        graph=Graph(2,opinions,infos,opinion_update,info_update,msg_generator)
        graph.update_dist_matrix()
        self.assertEqual(graph.nodes[0].dist_mat.all(),np.array([[0,0],[0,0]],dtype="float64").all())
        self.assertEqual(graph.nodes[0].dist_mat.all(),graph.nodes[1].dist_mat.all())
        
    def test_get_ndi(self):
        self.assertEqual(TestGraph.graph.get_ndi(-1).shape[0],TestGraph.vector_size)

if __name__=='__main__':
    #import os
    #os.chdir(os.path.abspath(os.path.dirname(__file__)))
    unittest.main()