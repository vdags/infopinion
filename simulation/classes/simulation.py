import os
import dill
import numpy as np
import matplotlib.pyplot as  plt
from copy import deepcopy
from graph import Graph
from inspect import getsource
def load(file_name:str):
    """
    Load simulation from pickle file. Return Simulation type object.
    """
    with open(file_name, "rb") as file:
        file.seek(0)
        return dill.load(file)


class Simulation:
    def __init__(self,nb_nodes,func_init,update_opinion,update_info,generate_msg,
                 embedding_size_opinion=10,embedding_size_information=5):
        """
        nb_nodes: int, number of nodes of the simulation

        update_info: func(list(np.array(m,1)),list(np.array(n,1)),np.array(n,1))--> np.array(n,1) , update info vector based on previous info and opinion states and recieved msg
        func_init: func(nb_nodes)--> list(list(np.array(n,1)),list(list(np.array(m,1)))), return initial values for opinions and informations
        update_opinion: func(list(np.array(m,1)),list(np.array(n,1)))--> np.array(m,1) , update opinion vector based on previous info and opinion states and recieved msg
        generate_msg: func(list(np.array(m,1)),list(np.array(n,1)))--> list(Node,np.array(n,1)) , genrate a list of couple of Node and msg based on info and opinion history, if no msg is recieved, a NoneType object is set.
        """
        self.size=nb_nodes
        self.init=func_init
        self.update_opinion=update_opinion
        self.update_info=update_info
        self.generate_msg=generate_msg
        init_values=self.init(self.size,embedding_size_opinion,embedding_size_information)
        self.graph=Graph(self.size,
                        init_values[0],
                        init_values[1],
                        self.update_opinion,
                        self.update_info,
                        self.generate_msg)
        self.fig=None   #TODO test
        self.opinion_size=embedding_size_opinion # TODO test
        self.info_size=embedding_size_information # TODO test

    def update_all_infos(self):
        for i in range(0,self.size):
            self.graph.nodes[i].update_info()

    def update_all_opinions(self):
        for i in range(0,self.size):
            self.graph.nodes[i].update_opinion()

    def generate_all(self):
        dist_mat=self.graph.get_dist_matrix(metric="euclidean")
        for i in range(0,self.size):
            self.graph.nodes[i].generate_msg()

    def send_all(self):
        recieved=[[] for i in range(0,self.size)]
        for i in range(0,self.size):
            for j in range(0,len(self.graph.nodes[i].sent[-1])):
                if self.graph.nodes[i].sent[-1][j]!=None:
                    recieved[self.graph.nodes[i].sent[-1][j][0]].append((i,self.graph.nodes[i].sent[-1][j][1]))
        for i in range(0,self.size):
            self.graph.nodes[i].recieved.append(recieved[i])

    def update(self):
        self.update_all_infos()
        self.update_all_opinions()
        self.graph.update_dist_matrix(t=-1,metric="euclidean")
        self.generate_all()
        self.send_all()

    def get_infos(self):
        infos=[]
        for i in range(0,self.size):
            infos.append(self.graph.nodes[i].infos)
        return infos

    def get_opinions(self):
        opinions=[]
        for i in range(0,self.size):
            opinions.append(self.graph.nodes[i].opinions)
        return opinions

    def get_sent(self):
        sent=[]
        for i in range(0,self.size):
           sent.append(self.graph.nodes[i].sent)
        return sent
    
    def get_recieved(self):
        recieved=[]
        for i in range(0,self.size):
           recieved.append(self.graph.nodes[i].sent)
        return recieved
    
#TODO add plotting

    def time_plotting(self):
        timesteps=len(self.get_opinions()[0])
        
        t_list=[i for i in range(0,timesteps)]
        height=int(self.opinion_size**0.5)+1
        width=int(self.opinion_size**0.5)+1
        fig, ax = plt.subplots(height,width,figsize=(width*5.5,height*3))
        
        # Memory size plot
        memory=deepcopy(self.get_infos()) # TODO test
        for i in range(0,self.size):
            for j in range(0,timesteps):
                memory[i][j]=len(memory[i][j])
        for i in range(0,self.size):
            ax[height-1,width-1].plot(t_list,memory[i])
        ax[height-1,width-1].set_title("Memory size")
        ax[height-1,width-1].set_ylabel("Nb of info in memory")
        ax[height-1,width-1].set_xlabel("timesteps")

        # Global Disagreement Index     #TODO test
        gdi=np.zeros((self.opinion_size,timesteps)) 
        for t in range(0,timesteps):
            gdi[:,t]=self.graph.get_gdi(t=t)[:,0]
        for i in range(0,self.opinion_size):
            ax[0,0].plot(t_list,gdi[i,:],label=f"Opinion {i}")    
        ax[0,0].legend(loc="upper left",fontsize='xx-small')
        ax[0,0].set_title("Global Disagreement Index (GDI)")
        ax[0,0].set_ylabel("GDI")
        ax[0,0].set_xlabel("timesteps")

        # # Avg opinion plot
        # op=[self.get_opinions()[i] for i in range(0,self.size)]
        # npop=np.zeros((self.size,timesteps))
        # #print(op)
        # for i in range(0,self.size):
        #     for j in range(0,timesteps):
        #         npop[i][j]= np.average(op[i][j])
        # op=npop.tolist()        
        # for i in op:
        #     ax[1,0].plot(t_list,i)
        # ax[1,0].set_title('Avg opinions')
        # ax[1,0].set_ylabel("Avg opinion")
        # ax[1,0].set_xlabel("timesteps")

        # Network Disagreement Index
        ndi=np.zeros((self.opinion_size,timesteps))
        for t in range(0,timesteps):
            ndi[:,t]=self.graph.get_ndi(t=t)[:,0]
        for i in range(0,self.opinion_size):
            ax[1,0].plot(t_list,ndi[i,:],label=f"Opinion {i}")    
        ax[1,0].legend(loc="upper left",fontsize='xx-small')
        ax[1,0].set_title("Network Disagreement Index (NDI)")
        ax[1,0].set_ylabel("NDI")
        ax[1,0].set_xlabel("timesteps")

        ## Opinion coordinate plots
        for op_coord in range(0,self.opinion_size):
            op_by_coord=np.zeros((self.size,timesteps))
            for node in range(0,self.size):
                for t in range(0,timesteps):
                    op_by_coord[node,t]=self.graph.get_node(node).get_opinion(t)[op_coord].tolist()[0]
                ax[(op_coord)//(width-1),op_coord%(width-1)+1].plot(t_list,op_by_coord[node,:],"+",ms = 2,alpha=.5)

            ax[(op_coord)//(width-1),op_coord%(width-1)+1].set_title(f"Opnion coord {op_coord}")
            ax[(op_coord)//(width-1),op_coord%(width-1)+1].set_ylabel("Opinion")
            ax[(op_coord)//(width-1),op_coord%(width-1)+1].set_xlabel("Timesteps")
        
        ## Avg Time since last sent message
        last_sent=np.zeros((self.size,timesteps))
        for node in range(0,self.size):
            for t in range(0,timesteps):
                c=1
                while self.graph.get_node(node).get_sent(t-c)==[] and c<t:
                    c+=1
                last_sent[node,t]=c
        avg_last_sent=np.zeros((timesteps))
        #print(last_sent)
        for t in range(0,timesteps):
            avg_last_sent[t]=last_sent[:,t].mean()

        ax[2,0].plot(t_list,avg_last_sent)
        ax[2,0].set_title("Avg time since last sent msg")
        ax[2,0].set_ylabel("Timesteps")
        ax[2,0].set_xlabel("Timesteps")

        ## Avg Time since last recieved message
        last_sent=np.zeros((self.size,timesteps))
        for node in range(0,self.size):
            for t in range(0,timesteps):
                c=1
                while self.graph.get_node(node).get_recieved(t-c)==[] and c<t:
                    c+=1
                last_sent[node,t]=c
        avg_last_sent=np.zeros((timesteps))
        #print(last_sent)
        for t in range(0,timesteps):
            avg_last_sent[t]=last_sent[:,t].mean()

        ax[3,0].plot(t_list,avg_last_sent)
        ax[3,0].set_title("Avg time since last recieved msg")
        ax[3,0].set_ylabel("Timesteps")
        ax[3,0].set_xlabel("Timesteps")





        fig.tight_layout()
        self.fig=fig
        return fig 
        #ax2=plt.subplot(2,2,3)
        #ax2.plot(t,[len(self.get_infos()) for i in range(0,self.size)])
        #ax2.set_title('Avg opinions')
        #ax2.set_ylabel("Avg opinion")
        #ax2.set_xlabel("timesteps")

    # def time_plotting(self,):
    #     timesteps=len(self.get_opinions()[0])
    #     t=[i for i in range(0,timesteps)]
    #     fig, ax = plt.subplots(2,1)
    #     memory=self.get_infos()
    #     for i in range(0,self.size):
    #         for j in range(0,timesteps):
    #             memory[i][j]=len(memory[i][j])
    #     for i in range(0,self.size):
    #         ax[0].plot(t,memory[i])
    #     ax[0].set_title("Memory size")
    #     ax[0].set_ylabel("Nb of info in memory")
    #     ax[0].set_xlabel("timesteps")

    #     #ax[1]=plt.subplot(2,2,2)
    #     op=[self.get_opinions()[i] for i in range(0,self.size)]
    #     npop=np.zeros((self.size,timesteps))
    #     #print(op)
    #     for i in range(0,self.size):
    #         for j in range(0,timesteps):
    #             npop[i][j]= np.average(op[i][j])
    #     op=npop.tolist()        
    #     for i in op:
    #         ax[1].plot(t,i)
    #     ax[1].set_title('Avg opinions')
    #     ax[1].set_ylabel("Avg opinion")
    #     ax[1].set_xlabel("timesteps")
    #     plt.tight_layout()
    #     self.fig=fig
    #     return fig 
    #     #ax2=plt.subplot(2,2,3)
    #     #ax2.plot(t,[len(self.get_infos()) for i in range(0,self.size)])
    #     #ax2.set_title('Avg opinions')
    #     #ax2.set_ylabel("Avg opinion")
    #     #ax2.set_xlabel("timesteps")

    def save_pkl(self,file_name="./simulation.pkl"):
        """
        Save the simulation object in a binary file (pickle).
        """
        
        with open(file_name, 'wb') as file:
            dill.dump(self, file)
    
    def save(self,file_name='./simulation'): ## TODO test
        """
        Save simulation object, figure and report

        file_name: str, is radical of the file name, **whithout extention**
        """
        self.save_pkl(file_name=file_name+".pkl")
        self.fig.savefig(file_name+".png",dpi=100)
        plt.close() # TODO test
        with open(file_name+".txt", "w") as f:
            f.write(self.report())
            f.close()
        
        

    def report(self):
        str=""
        str+=f"NB_NODES = {self.size}\n"
        str+=f"ENCODING_DIM = {self.get_opinions()[0][0].shape[0]}\n"
        str+=f"TIMESTEPS = {len(self.get_opinions()[0])-1}\n"
        if "lambda" in self.init.__str__():
            str+=f"INIT_FUNC ={getsource(self.init)}\n"
        else:
            str+=f"INIT_FUNC ={self.init}\n"
        if "lambda" in self.update_info.__str__():
            str+=f"UPDATE_INFO = {getsource(self.update_info)}\n"
        else:
            str+=f"UPDATE_INFO = {self.update_info}\n"
        if "lambda" in self.update_opinion.__str__():
            str+=f"UPDATE_OPINIONS = {getsource(self.update_opinion)}\n"        
        else:
            str+=f"UPDATE_OPINIONS = {self.update_opinion}\n"
        if "lambda" in self.generate_msg.__str__():
            str+=f"MSG_GENERATOR = {getsource(self.generate_msg)}\n"
        else:    
            str+=f"MSG_GENERATOR = {self.generate_msg}\n"
        avg_op_mat=np.array([self.get_opinions()[i][-1] for i in range(0,self.size)])
        avg_op=[avg_op_mat[i,:].mean() for i in range(0,self.get_opinions()[0][0].shape[0])]
        std_op=[avg_op_mat[i,:].std() for i in range(0,self.get_opinions()[0][0].shape[0])]
        str+=f"AVG_OPINION_FINAL: {avg_op}\n"
        str+=f"STD_OPINION_FINAL: {std_op}\n"
        ## TODO complete
        return str
