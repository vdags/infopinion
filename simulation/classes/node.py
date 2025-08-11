import numpy as np

class Node:
    def __init__(self,id,opinion_0,info_0,update_opinion,update_info,generate_msg,dist_mat):
        """
        Create Node object with its intial state of opinion and information

        opinion_0: np.array(n,1), opinion vector of the node at time 0
        info_0: list(np.array(m,1)), list of pieces of information available at time 0
        update_info: func(list(np.array(m,1)),list(np.array(n,1)),np.array(n,1))--> np.array(n,1) , update info vector based on previous info and opinion states and recieved msg
        update_opinion: func(list(np.array(m,1)),list(np.array(n,1)))--> np.array(m,1) , update opinion vector based on previous info and opinion states and recieved msg
        generate_msg: func(list(np.array(m,1)),list(np.array(n,1)))--> list(Node,np.array(n,1)) , genrate a list of couple of Node and msg based on info and opinion history
        dist_mat: np.array(2,nb_nodes,nb_nodes), two last matrix of distance of the graph in wich the node is. Can be used by some internal functions as weigth or heuristic.
        """
        self.id=id
        self.opinions=[opinion_0]
        self.infos=[info_0]
        self.sent=[[(0,np.zeros(np.shape(info_0)))]]
        self.recieved=[[(0,np.zeros(np.shape(info_0[0])))]]
        self.func_info=update_info
        self.func_opinion=update_opinion
        self.msg_generator=generate_msg
        self.dist_mat=np.zeros((0,np.shape(dist_mat)[0],np.shape(dist_mat)[0]))
        self.dist_mat=np.append(self.dist_mat,np.array([dist_mat]),axis=0)


    def get_opinion(self,t=-1):
        return self.opinions[t]

    def get_info(self,t=-1):
        return self.infos[t]

    def get_info_all(self): #TODO test
        return self.infos
    
    def get_opnion_all(self): #TODO test
        return self.opinions

    def get_sent(self,t=-1): #TODO test
        return self.sent[-1]
    
    def get_sent_all(self): #TODO test
        return self.sent

    def get_recieved(self,t=-1):
        return self.recieved[t]
    
    def get_recieved_all(self):
        return self.recieved

    def get_history(self):
        return (self.opinions,self.infos,self.recieved,self.sent)

    def set_dist_matrix(self,dist_mat):
        self.dist_mat=np.array([self.dist_mat[-1,:,:]]) # limit nb of dist matrix to two to reduce memory complexity
        self.dist_mat=np.append(self.dist_mat,np.array([dist_mat]),axis=0)

    def get_dist_matrix(self,t=-1): #TODO test
        # t=-1 works always due to initialisation with first distance matrix
        return self.dist_mat[t,:,:]

    def update_info(self):
        new_info=self.func_info(self)
        self.infos.append(new_info)
        return new_info

    def update_opinion(self):
        new_opinion=self.func_opinion(self)
        self.opinions.append(new_opinion)
        return new_opinion

    def generate_msg(self):
        new_msgs=self.msg_generator(self)
        self.sent.append(new_msgs)
        return new_msgs





