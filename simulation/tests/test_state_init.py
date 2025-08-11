import numpy as np
import unittest
import sys
import os
os.chdir(os.path.abspath(os.path.dirname(__file__)))
sys.path.append("..//functions")
from state_init import *
sys.path.append("../classes")
from simulation import Simulation


class TestStateInit(unittest.TestCase):
    def test_independent_cst_normal_distribution(self):
        nb_nodes=100
        size_info=10
        size_opinion=200
        res = independent_cst_normal_distribution(nb_nodes,size_opinion,size_info)
        self.assertEqual(len(res[0]),nb_nodes)
        self.assertEqual(len(res[1]),nb_nodes)
        self.assertEqual(np.full((size_opinion,1),res[0][0]).all(),res[0][0].all())
        self.assertEqual(np.full((size_info,1),res[1][0][0]).all(),res[1][0][0].all())

    def test_independent_cst_normal_distribution_in_context(self):
        nb_nodes=100
        vector_size=10
        func_init=independent_cst_normal_distribution
        info_update=lambda x : x.get_info()
        opinion_update=lambda x : x.get_opinion()
        msg_generator=lambda x : [(1,np.zeros((vector_size,1)))]
        simulation=Simulation(nb_nodes,func_init,opinion_update,info_update,msg_generator,vector_size,vector_size)
        for i in range(0,5):
            simulation.update()
    
    def test_independent_normal_distribution(self):
        nb_nodes=100
        size_info=10
        size_opinion=200
        res = independent_normal_distribution(nb_nodes,size_info,size_opinion)
        self.assertEqual(len(res[0]),nb_nodes)
        self.assertEqual(len(res[1]),nb_nodes)

    def test_independent_normal_distribution_in_context(self):
        nb_nodes=100
        vector_size=10
        func_init=independent_normal_distribution
        info_update=lambda x : x.get_info()
        opinion_update=lambda x : x.get_opinion()
        msg_generator=lambda x : [(1,np.zeros((vector_size,1)))]
        simulation=Simulation(nb_nodes,func_init,opinion_update,info_update,msg_generator,vector_size,vector_size)
        for i in range(0,5):
            simulation.update()
    
    def test_independent_uniform_distribution(self):
        nb_nodes=100
        size_info=10
        size_opinion=200
        res = independent_normal_distribution(nb_nodes,size_info,size_opinion)
        self.assertEqual(len(res[0]),nb_nodes)
        self.assertEqual(len(res[1]),nb_nodes)
    
    def test_independent_uniform_distribution_in_context(self):
        nb_nodes=100
        vector_size=10
        func_init=independent_uniform_distribution
        info_update=lambda x : x.get_info()
        opinion_update=lambda x : x.get_opinion()
        msg_generator=lambda x : [(1,np.zeros((vector_size,1)))]
        simulation=Simulation(nb_nodes,func_init,opinion_update,info_update,msg_generator,vector_size,vector_size)
        for i in range(0,5):
            simulation.update()

    def test_quasi_gaussian_segment_generator(self):
        res = quasi_gaussian_segment_generator((10,1),-1,1)
        self.assertEqual(res.shape[0],10)
        self.assertLessEqual(res.all(),1)
        self.assertGreaterEqual(res.all(),-1)

if __name__=='__main__':
    unittest.main()
        