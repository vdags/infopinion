import numpy as np
import unittest
import sys
import os
os.chdir(os.path.abspath(os.path.dirname(__file__)))
sys.path.append("../classes")
from node import Node

class TestNode(unittest.TestCase):

    opinion=np.ones((10,1))
    info=[np.ones((5,1))]
    info_update=lambda x : x.get_info()
    opinion_update=lambda x : x.get_opinion()
    msg_generator=lambda x : [(1,np.zeros((5,1)))]
    dist_matrix=np.array([[1,2],[4,1]])
    node=Node(1,opinion,info,opinion_update,info_update,msg_generator,dist_matrix)

    def test___init__(self):
        opinion=np.ones((10,1))
        info=[np.ones((5,1))]
        info_update=lambda x : x.get_info()
        opinion_update=lambda x : x.get_opinion()
        msg_generator=lambda x : [(1,np.zeros((5,1)))]
        dist_matrix=np.array([[1,2],[4,1]])
        node=Node(1,opinion,info,opinion_update,info_update,msg_generator,dist_matrix)
        #self.assertEqual(node.opinions,[opinion])
        self.assertIsNone(np.testing.assert_array_equal(node.opinions,[opinion]))
        #self.assertEqual(node.infos,[info])
        self.assertIsNone(np.testing.assert_array_equal(node.infos,[info]))
        self.assertEqual(node.sent[0][0][0],0)
        self.assertEqual(node.sent[0][0][1].all(),np.zeros((5,1)).all())
        self.assertEqual(len(node.recieved[0]),1)
        self.assertEqual(len(node.recieved[0][0]),2)
        self.assertEqual(node.recieved[0][0][0],0)
        self.assertEqual(node.recieved[0][0][1].all(),np.zeros((5,1)).all())
        self.assertEqual(node.dist_mat.all(),np.array([[1,2],[4,1]]).all())

    def test___init__2(self):
        opinion=np.ones((10,1))
        info=[np.ones((5,1))]
        info_update=lambda x : x.get_info()
        opinion_update=lambda x : x.get_opinion()
        msg_generator=lambda x : [(1,np.zeros((5,1)))]
        dist_matrix=np.array([[1,2],[4,1]])
        node=Node(1,opinion,info,opinion_update,info_update,msg_generator,dist_matrix)
        #self.assertEqual(node.opinions,[opinion])
        self.assertIsNone(np.testing.assert_array_equal(node.opinions,[opinion]))
        #self.assertEqual(node.infos,[info])
        self.assertIsNone(np.testing.assert_array_equal(node.infos,[info]))
        self.assertEqual(node.sent[0][0][0],0)
        self.assertEqual(node.sent[0][0][1].all(),np.zeros((5,1)).all())
        self.assertEqual(len(node.recieved[0]),1)
        self.assertEqual(len(node.recieved[0][0]),2)
        self.assertEqual(node.recieved[0][0][0],0)
        self.assertEqual(node.recieved[0][0][1].all(),np.zeros((5,1)).all())
        self.assertEqual(node.dist_mat.all(),np.array([[[1,2],[4,1]]]).all())

    def test_get_opinion(self):
        self.assertIsNone(np.testing.assert_array_equal(TestNode.node.get_opinion(),np.ones((10,1)) ))


    def test_get_info(self):
        self.assertIsNone(np.testing.assert_array_equal(TestNode.node.get_info(),[np.ones((5,1))]))

    def test_get_history(self):
        test=TestNode.node.get_history()
        self.assertIsNone(np.testing.assert_array_equal(test[0],[np.ones((10,1))] ))
        self.assertIsNone(np.testing.assert_array_equal(test[1],[[np.ones((5,1))]] ))
        self.assertEqual(test[2][0][0][0],0)
        self.assertEqual(test[2][0][0][1].all(),np.zeros((5,1)).all())
        self.assertEqual(test[3][0][0][0],0)
        self.assertEqual(test[3][0][0][1].all(),np.zeros((5,1)).all())

    def test_update_info(self):
        TestNode.node.update_info()
        self.assertEqual(len(TestNode.node.infos),2)
        self.assertEqual(len(TestNode.node.infos[1]),1)
        self.assertEqual(TestNode.node.infos[1][0].all(),np.ones((5,1)).all())

    def test_update_opinion(self):
        TestNode.node.update_opinion()
        self.assertEqual(len(TestNode.node.opinions),2)
        self.assertEqual(TestNode.node.opinions[1].all(),np.ones((10,1)).all())

    def test_generate_msg(self):
        TestNode.node.generate_msg()
        self.assertEqual(len(TestNode.node.sent),2)
        self.assertEqual(len(TestNode.node.sent[1]),1)
        self.assertEqual(TestNode.node.sent[1][0][0],1)
        self.assertEqual(TestNode.node.sent[1][0][1].all(),np.zeros((5,1)).all())

    def test_get_recieved(self):
        self.assertEqual(len(TestNode.node.get_recieved()),1)
        self.assertEqual(len(TestNode.node.get_recieved()[0]),2)

    def test_get_recieved_all(self):
        self.assertEqual(len(TestNode.node.get_recieved_all()),1)
        self.assertEqual(len(TestNode.node.get_recieved_all()[0]),1)
        self.assertEqual(len(TestNode.node.get_recieved_all()[0][0]),2)


    def test_set_dist_matrix(self):
        opinion=np.ones((10,1))
        info=[np.ones((5,1))]
        info_update=lambda x,y,z : y
        opinion_update=lambda x,y : x
        msg_generator=lambda x,y : [(1,np.zeros((5,1)))]
        dist_matrix=np.array([[1,2],[4,1]])
        node=Node(1,opinion,info,opinion_update,info_update,msg_generator,dist_matrix)
        
        node.set_dist_matrix(np.array([[1,2],[4,1]]))
        self.assertEqual(node.dist_mat.all(),np.array([[[1,2],[4,1]],[[1,2],[4,1]]]).all())


if __name__=='__main__':
    unittest.main()
