import numpy as np
import unittest
import os
import matplotlib.pyplot as plt
os.chdir(os.path.abspath(os.path.dirname(__file__)))
import sys
sys.path.append("../classes")
from node import Node
from graph import Graph 
from simulation import Simulation,load


class TestSimulation(unittest.TestCase):
    nb_nodes=100
    vector_size=10
    func_init=lambda N,n,m : [[np.ones((n,1)) for i in range(N)],[[np.zeros((m,1))] for i in range(N)]]
    info_update=lambda x : x.get_info()
    opinion_update=lambda x : x.get_opinion()
    msg_generator=lambda x : [(1,np.zeros((10,1)))] #set vector_size here
    simulation=Simulation(nb_nodes,func_init,opinion_update,info_update,msg_generator,vector_size,vector_size)

    def test___init__(self):
        nb_nodes=100
        vector_size=10
        func_init=lambda N,n,m : [[np.ones((n,1)) for i in range(N)],[[np.zeros((m,1))] for i in range(N)]]
        info_update=lambda x : x.get_info()
        opinion_update=lambda x : x.get_opinion()
        msg_generator=lambda x : [(1,np.zeros((vector_size,1)))]
        simulation=Simulation(nb_nodes,func_init,opinion_update,info_update,msg_generator,vector_size,vector_size)
        self.assertEqual(simulation.size,nb_nodes)
        self.assertEqual(simulation.graph.nodes[0].opinions[0].all(),np.ones((10,1)).all())
        self.assertEqual(simulation.graph.nodes[0].infos[0][0].all(),np.zeros((5,1)).all())
        self.assertEqual(simulation.graph.nodes[nb_nodes-1].opinions[0].all(),np.ones((10,1)).all())
        self.assertEqual(simulation.graph.nodes[nb_nodes-1].infos[0][0].all(),np.zeros((5,1)).all())

    def test_update_all_info(self):
        TestSimulation.simulation.update_all_infos()
        self.assertEqual(len(TestSimulation.simulation.graph.nodes[0].infos),2)
        self.assertEqual(len(TestSimulation.simulation.graph.nodes[0].infos[1]),1)
        self.assertEqual(TestSimulation.simulation.graph.nodes[0].infos[1][0].all(),np.zeros((5,1)).all())
        self.assertEqual(len(TestSimulation.simulation.graph.nodes[TestSimulation.simulation.size-1].infos),2)
        self.assertEqual(len(TestSimulation.simulation.graph.nodes[TestSimulation.simulation.size-1].infos[1]),1)
        self.assertEqual(TestSimulation.simulation.graph.nodes[TestSimulation.simulation.size-1].infos[1][0].all(),np.zeros((5,1)).all())

    def test_update_all_opinions(self):
        TestSimulation.simulation.update_all_opinions()
        self.assertEqual(len(TestSimulation.simulation.graph.nodes[0].opinions),2)
        self.assertEqual(TestSimulation.simulation.graph.nodes[0].opinions[1].all(),np.ones((10,1)).all())
        self.assertEqual(len(TestSimulation.simulation.graph.nodes[TestSimulation.simulation.size-1].opinions),2)
        self.assertEqual(TestSimulation.simulation.graph.nodes[TestSimulation.simulation.size-1].opinions[1].all(),np.ones((10,1)).all())

    def test_generate_all(self):
        TestSimulation.simulation.generate_all()
        self.assertEqual(len(TestSimulation.simulation.graph.nodes[0].sent),2)
        self.assertEqual(len(TestSimulation.simulation.graph.nodes[0].sent[1]),1)
        self.assertEqual(len(TestSimulation.simulation.graph.nodes[0].sent[1][0]),2)
        self.assertEqual(TestSimulation.simulation.graph.nodes[0].sent[1][0][0],1)
        self.assertEqual(TestSimulation.simulation.graph.nodes[0].sent[1][0][1].all(),np.zeros((5,1)).all())
        self.assertEqual(len(TestSimulation.simulation.graph.nodes[TestSimulation.simulation.size-1].sent),2)
        self.assertEqual(len(TestSimulation.simulation.graph.nodes[TestSimulation.simulation.size-1].sent[1]),1)
        self.assertEqual(len(TestSimulation.simulation.graph.nodes[TestSimulation.simulation.size-1].sent[1][0]),2)
        self.assertEqual(TestSimulation.simulation.graph.nodes[TestSimulation.simulation.size-1].sent[1][0][0],1)
        self.assertEqual(TestSimulation.simulation.graph.nodes[TestSimulation.simulation.size-1].sent[1][0][1].all(),np.zeros((5,1)).all())

    def test_send_all(self):
        TestSimulation.simulation.send_all()
        self.assertEqual(len(TestSimulation.simulation.graph.nodes[0].recieved),2)
        self.assertEqual(len(TestSimulation.simulation.graph.nodes[0].recieved[1]),0)
        self.assertEqual(len(TestSimulation.simulation.graph.nodes[TestSimulation.simulation.size-1].recieved[1]),0)
        self.assertEqual(len(TestSimulation.simulation.graph.nodes[1].recieved[1]),100)
        self.assertEqual(len(TestSimulation.simulation.graph.nodes[1].recieved[1][0]),2)
        self.assertEqual(TestSimulation.simulation.graph.nodes[1].recieved[1][0][0],0)
        self.assertEqual(TestSimulation.simulation.graph.nodes[1].recieved[1][TestSimulation.simulation.size-1][0],99)
        self.assertEqual(TestSimulation.simulation.graph.nodes[1].recieved[1][0][1].all(),np.zeros((5,1)).all())
        self.assertEqual(TestSimulation.simulation.graph.nodes[1].recieved[1][TestSimulation.simulation.size-1][1].all(),np.zeros((5,1)).all())


    def test_update(self):
        nb_nodes=100
        vector_size=10
        func_init=lambda N,n,m : [[np.ones((n,1)) for i in range(N)],[[np.zeros((m,1))] for i in range(N)]]
        info_update=lambda x : x.get_info()
        opinion_update=lambda x : x.get_opinion()
        msg_generator=lambda x : [(1,np.zeros((vector_size,1)))]
        simulation=Simulation(nb_nodes,func_init,opinion_update,info_update,msg_generator,vector_size,vector_size)
    
        simulation.update()

        self.assertEqual(len(simulation.graph.nodes[0].infos),2)
        self.assertEqual(len(simulation.graph.nodes[0].infos[1]),1)
        self.assertEqual(simulation.graph.nodes[0].infos[1][0].all(),np.zeros((5,1)).all())
        self.assertEqual(len(simulation.graph.nodes[simulation.size-1].infos),2)
        self.assertEqual(len(simulation.graph.nodes[simulation.size-1].infos[1]),1)
        self.assertEqual(simulation.graph.nodes[simulation.size-1].infos[1][0].all(),np.zeros((5,1)).all())

        self.assertEqual(len(simulation.graph.nodes[0].opinions),2)
        self.assertEqual(simulation.graph.nodes[0].opinions[1].all(),np.ones((10,1)).all())
        self.assertEqual(len(simulation.graph.nodes[simulation.size-1].opinions),2)
        self.assertEqual(simulation.graph.nodes[simulation.size-1].opinions[1].all(),np.ones((10,1)).all())
        self.assertEqual(len(simulation.graph.nodes[0].sent),2)
        self.assertEqual(len(simulation.graph.nodes[0].sent[1]),1)
        self.assertEqual(len(simulation.graph.nodes[0].sent[1][0]),2)
        self.assertEqual(simulation.graph.nodes[0].sent[1][0][0],1)
        self.assertEqual(simulation.graph.nodes[0].sent[1][0][1].all(),np.zeros((5,1)).all())
        self.assertEqual(len(simulation.graph.nodes[simulation.size-1].sent),2)
        self.assertEqual(len(simulation.graph.nodes[simulation.size-1].sent[1]),1)
        self.assertEqual(len(simulation.graph.nodes[simulation.size-1].sent[1][0]),2)
        self.assertEqual(simulation.graph.nodes[simulation.size-1].sent[1][0][0],1)
        self.assertEqual(simulation.graph.nodes[simulation.size-1].sent[1][0][1].all(),np.zeros((5,1)).all())

        self.assertEqual(len(simulation.graph.nodes[0].recieved),2)
        self.assertEqual(len(simulation.graph.nodes[0].recieved[1]),0)
        self.assertEqual(len(simulation.graph.nodes[simulation.size-1].recieved[1]),0)
        self.assertEqual(len(simulation.graph.nodes[1].recieved[1]),100)
        self.assertEqual(len(simulation.graph.nodes[1].recieved[1][0]),2)
        self.assertEqual(simulation.graph.nodes[1].recieved[1][0][0],0)
        self.assertEqual(simulation.graph.nodes[1].recieved[1][simulation.size-1][0],99)
        self.assertEqual(simulation.graph.nodes[1].recieved[1][0][1].all(),np.zeros((5,1)).all())
        self.assertEqual(simulation.graph.nodes[1].recieved[1][simulation.size-1][1].all(),np.zeros((5,1)).all())

    def test_get_opinions(self):
        simulation=Simulation(TestSimulation.nb_nodes,
                                TestSimulation.func_init,
                                TestSimulation.opinion_update,
                                TestSimulation.info_update,
                                TestSimulation.msg_generator,
                                10,
                                5)
        simulation.update()
        opinions=simulation.get_opinions()
        self.assertEqual(len(opinions),simulation.size)
        self.assertEqual(len(opinions[0]),2)
        self.assertEqual(len(opinions[simulation.size-1]),2)
        for i in range(0,10):
            simulation.update()
        self.assertEqual(len(opinions[0]),12)
        self.assertEqual(len(opinions[simulation.size-1]),12)

    def test_get_infos(self):
        simulation=Simulation(TestSimulation.nb_nodes,
                                TestSimulation.func_init,
                                TestSimulation.opinion_update,
                                TestSimulation.info_update,
                                TestSimulation.msg_generator,
                                10,
                                5)
        simulation.update()
        infos=simulation.get_infos()
        self.assertEqual(len(infos),simulation.size)
        self.assertEqual(len(infos[0]),2)
        self.assertEqual(len(infos[simulation.size-1]),2)
        for i in range(0,10):
            simulation.update()
        self.assertEqual(len(infos[0]),12)
        self.assertEqual(len(infos[simulation.size-1]),12)
    
    def test_get_sent(self):
        simulation=Simulation(TestSimulation.nb_nodes,
                                TestSimulation.func_init,
                                TestSimulation.opinion_update,
                                TestSimulation.info_update,
                                TestSimulation.msg_generator,
                                10,
                                5)
        simulation.update()
        sent=simulation.get_sent()
        self.assertEqual(len(sent),simulation.size)
        self.assertEqual(len(sent[0]),2)
        self.assertEqual(len(sent[simulation.size-1]),2)
        for i in range(0,10):
            simulation.update()
        self.assertEqual(len(sent[0]),12)
        self.assertEqual(len(sent[simulation.size-1]),12)
    
    def test_get_recieved(self):
        simulation=Simulation(TestSimulation.nb_nodes,
                                TestSimulation.func_init,
                                TestSimulation.opinion_update,
                                TestSimulation.info_update,
                                TestSimulation.msg_generator,
                                10,
                                5)
        simulation.update()
        recieved=simulation.get_recieved()
        self.assertEqual(len(recieved),simulation.size)
        self.assertEqual(len(recieved[0]),2)
        self.assertEqual(len(recieved[simulation.size-1]),2)
        for i in range(0,10):
            simulation.update()
        self.assertEqual(len(recieved[0]),12)
        self.assertEqual(len(recieved[simulation.size-1]),12)

    def test_time_plotting(self):
        nb_nodes=100
        vector_size=10
        func_init=lambda N,n,m : [[np.ones((n,1)) for i in range(N)],[[np.zeros((m,1))] for i in range(N)]]
        info_update=lambda x : x.get_info()
        opinion_update=lambda x : x.get_opinion()
        msg_generator=lambda x : [(1,np.zeros((vector_size,1)))]
        simulation=Simulation(nb_nodes,func_init,opinion_update,info_update,msg_generator,vector_size,vector_size)
        for i in range(0,10):
            simulation.update()
        simulation.time_plotting().show()
    
    def test_save_pkl(self):
        TestSimulation.simulation.save_pkl("./test_files/tmp_test_simulation.pkl")
        os.remove("./test_files/tmp_test_simulation.pkl")
    
    def test_load(self):        
        simulation = load("./test_files/test_simulation_0.pkl")
        self.assertEqual(simulation.size,1000)
        simulation.init
        simulation.update_opinion
        simulation.update_info
        simulation.generate_msg
        simulation.graph



if __name__=='__main__':
    unittest.main()