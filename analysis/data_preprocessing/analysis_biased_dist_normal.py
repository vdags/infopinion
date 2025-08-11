import numpy as np
import pandas as pd
from functools import partial
import os
os.chdir(os.path.abspath(os.path.dirname(__file__)))
import matplotlib.pyplot as plt
import sys
sys.path.append("..//classes")
from node import Node
from graph import Graph 
from simulation import Simulation,load
sys.path.append("""..//functions""")
import state_init
import update_info
import update_opinion
import msg_gen

def stabilise(simulation,t=-5,length=5,tol=0.01,ndi=None):
    """
    Evaluate if a simulation stabilise(1/0)

    length: int, nb of timestep for considering stabilisation
    tol: float, percetage of variation below stabilisation in considered as true. 
    """
    stab_mat=np.zeros((simulation.opinion_size,1))
    if type(ndi)!=type(np.array([1])):
        ndi=get_ndi(simulation)
    for i in range(0,simulation.opinion_size):
        if abs(ndi[i,t]-ndi[i,t+length-1])/length<=tol*ndi[i,0]:
            stab_mat[i,0]=1
        else:
            stab_mat[i,0]=0
    return stab_mat
   

def time_stab(simulation,length=5,tol_stab=0.01,ndi=None):
    timesteps =len(simulation.get_opinions()[0])
    stab_time=[]
    if type(ndi)!=type(np.array([1])):
        ndi=get_ndi(simulation)
    for i in range(0,simulation.opinion_size):
        t=timesteps-length
        while abs(ndi[i,t]-ndi[i,t+length-1])/length<=tol_stab*ndi[i,0] and t>0:
            t=t-1
        stab_time.append(t)
    return np.array(stab_time)


def get_ndi(simulation):
    timesteps=len(simulation.get_opinions()[0])
    ndi=np.zeros((simulation.opinion_size,timesteps))
    for t in range(0,timesteps):
        ndi[:,t]=simulation.graph.get_ndi(t)[:,0]
    return ndi

def get_gdi(simulation):
    timesteps=len(simulation.get_opinions()[0])
    gdi=np.zeros((simulation.opinion_size,timesteps))
    for t in range(0,timesteps):
        gdi[:,t]=simulation.graph.get_gdi(t)[:,0]
    return gdi

def get_time_full_memory(simulation):
    t=0
    full=0
    timesteps=len(simulation.get_opinions()[0])
    while t<timesteps and full==0:
        c=0
        for node in simulation.graph.nodes:
            if len(node.get_info(t))==7:
                c+=1
        if c==simulation.size:
            full=t
        t+=1
    return t

def type_stabilise(simulation,length=5,tol_stab=0.01,tol_pol=0.2,tol_gdi=0,ndi=None,gdi=None):
    """
    Return a mtrix encoding the type of stabilisation
    0: consensus
    1: uni-polarisation
    2: bi-polarisation
    -1:no stabilisation


    """
    consensus=np.zeros((simulation.opinion_size))
    bipolarisation=np.zeros((simulation.opinion_size))
    unipolarisation=np.zeros((simulation.opinion_size))
    no_stabilisation=np.zeros((simulation.opinion_size))
    timesteps=len(simulation.get_opinions()[0])    
    t=timesteps-length
    if type(ndi)!=type(np.array([1])):
        ndi=get_ndi(simulation)
    if type(gdi)!=type(np.array([1])):
        gdi=get_gdi(simulation)
    for i in range(0,simulation.opinion_size):
        if abs(ndi[i,t]-ndi[i,t+length-1])/length<=tol_stab*ndi[i,0]:
            if gdi[i,-1]>gdi[i,0]*(1+tol_gdi):
                bipolarisation[i]=1
            else:
                if (sum([simulation.graph.nodes[k].get_opinion()[i] for k in range(0,simulation.size)])/simulation.size <= tol_pol or 
                    sum([simulation.graph.nodes[k].get_opinion()[i] for k in range(0,simulation.size)])/simulation.size >= 1-tol_pol):
                    unipolarisation[i]=1
                else:
                    consensus[i]=1

        else:
            no_stabilisation[i]=1 
    return no_stabilisation,consensus,unipolarisation,bipolarisation
    
def get_df(file_min=2500,
           file_max=2505,
           path="../../../outputs/",
           opinion_param_var = "update_opinion_(coef,bias)",
           msg_param_var = "msg_gen_one_info_dist_global_coef"):
    
    simulation_list = []
    opinion_param=[]
    msg_param=[]
    for i in range(file_min,file_max+1):
        try:
            f=open(path+f"simulation_{i}.txt","r")
            report = f.readlines()
            for line in report:
                if opinion_param_var+" =" in line:
                    opinion_param.append(eval(line.split('=')[1].replace('\n','')))
                if msg_param_var+" =" in line:
                    msg_param.append(eval(line.split('=')[1].replace('\n','')))    
            #if opinion_func in report:
            simulation_list.append(load(path+f"simulation_{i}.pkl"))
            print(f"simulation {i} loaded")
        except:
            pass
    stab_list=[]
    stab_type=np.zeros((len(simulation_list),4))
    full_mem=[]

    ndi=[]
    gdi=[]
    for i,simulation in enumerate(simulation_list):
        full_mem.append(get_time_full_memory(simulation))
        ndi.append(get_ndi(simulation))
        gdi.append(get_gdi(simulation))
        stab_list.append(stabilise(simulation,ndi=ndi[-1]))
        #type_stabilise(simulation,ndi=ndi[-1])
        stab_type[i,:]=np.array([np.array(type_stabilise(simulation,ndi=ndi[-1],gdi=gdi[-1])).transpose()[:,i].mean() for i in range(0,4)])
        
    #print(stab_type)
    #print(ndi)
    #print(simulation_list)
    #print(stab_list)
    avg_stab=[i.mean() for i in stab_list]
    #print(avg_stab)
    #print(opinion_param)
    #print(msg_param)
    stab_time=[time_stab(simulation_list[i],ndi=ndi[i],tol_stab=0.01) for i in range(0,len(simulation_list))]
    avg_stab_time=[i.mean() for i in stab_time]
    std_stab_time=[i.std() for i in stab_time]
    opinion_param=np.array(opinion_param)
    df=pd.DataFrame({"opinion_param_1":opinion_param[:,0],
                  "opinion_param_2":opinion_param[:,1],      
                  "msg_param":msg_param,
                  "full_memory_time":np.array(full_mem),
                  "proba_stab":avg_stab,
                  "avg_stab_time":avg_stab_time,
                  "std_stab_time":std_stab_time,
                  "proba_no_stab":stab_type[:,0],
                  "proba_consensus":stab_type[:,1],
                  "proba_unipolarisation":stab_type[:,2],
                  "proba_bipolarisation":stab_type[:,3]
                  })    
    return df

if __name__=="__main__":
    os.chdir(os.path.abspath(os.path.dirname(__file__)))
    print(os.getcwd())
    path="../output/"#"../../../outputs/"
    file_min=3198#1500
    file_max=3208
    max_cache=2
    opinion_param_var = "update_opinion_(coef,bias)"
    msg_param_var = "msg_gen_one_info_dist_global_coef"
    filename_export="biased_dist_normal_2.csv"

    
    
    if file_max-file_min<max_cache:
        df = get_df(file_min,file_max,path=path,opinion_param_var=opinion_param_var,msg_param_var=msg_param_var)
    else:
            
        for i in range(0,(file_max-file_min)//max_cache):
            # df=pd.DataFrame({"opinion_param_1":[],
            #       "opinion_param_2":[],  
            #       "msg_param":[],
            #       "full_memory_time":[],
            #       "proba_stab":[],
            #       "avg_stab_time":[],
            #       "std_stab_time":[],
            #       "proba_no_stab":[],
            #       "proba_consensus":[],
            #       "proba_unipolarisation":[],
            #       "proba_bipolarisation":[]
            #       })
            df = get_df(file_min=file_min+i*max_cache,
                   file_max=file_min+i*max_cache+max_cache-1,
                   path=path,
                   opinion_param_var=opinion_param_var,
                   msg_param_var=msg_param_var)
            #df=pd.concat([df,df_temp])
            df.to_csv(filename_export,mode='a',index=False,header=False)
            del df
            print(f'batch {i}')
        df_temp = get_df(file_min=file_min+((file_max-file_min)//max_cache)*max_cache,
                file_max=file_max,
                path=path,
                opinion_param_var=opinion_param_var,
                msg_param_var=msg_param_var)
        #df=pd.concat([df,df_temp])
        df_temp.to_csv(filename_export,mode='a',index=False,header=False)
    #print(df[["opinion_param_1","opinion_param_2","msg_param","avg_stab_time","proba_no_stab","proba_consensus","proba_unipolarisation","proba_bipolarisation"]])
    #df.to_csv(filename_export,index=False)

    # for simulation in simulation_list:
    #   simulation.time_plotting().show()
    #   input("Press key to continue")

    


# #TODO redo graph for checking results

# import numpy as np
# import pandas as pd
# from functools import partial
# import os
# os.chdir(os.path.abspath(os.path.dirname(__file__)))
# import matplotlib.pyplot as plt
# import sys
# sys.path.append("..//classes")
# from node import Node
# from graph import Graph 
# from simulation import Simulation,load
# sys.path.append("""..//functions""")
# import state_init
# import update_info
# import update_opinion
# import msg_gen

# def stabilise(simulation,t=-5,length=5,tol=0.1,ndi=None):
#     """
#     Evaluate if a simulation stabilise(1/0)

#     length: int, nb of timestep for considering stabilisation
#     tol: float, percetage of variation below stabilisation in considered as true. 
#     """
#     stab_mat=np.zeros((simulation.opinion_size,1))
#     if type(ndi)!=type(np.array([1])):
#         ndi=get_ndi(simulation)
#     for i in range(0,simulation.opinion_size):
#         if abs(ndi[i,t]-ndi[i,t+length-1])/length<=tol:
#             stab_mat[i,0]=1
#         else:
#             stab_mat[i,0]=0
#     return stab_mat
   

# def time_stab(simulation,length=5,tol_stab=0.1,ndi=None):
#     timesteps =len(simulation.get_opinions()[0])
#     stab_time=[]
#     if type(ndi)!=type(np.array([1])):
#         ndi=get_ndi(simulation)
#     for i in range(0,simulation.opinion_size):
#         t=timesteps-length
#         while abs(ndi[i,t]-ndi[i,t+length-1])/length<=tol_stab and t>0:
#             t=t-1
#         stab_time.append(t)
#     return np.array(stab_time)


# def get_ndi(simulation):
#     timesteps=len(simulation.get_opinions()[0])
#     ndi=np.zeros((simulation.opinion_size,timesteps))
#     for t in range(0,timesteps):
#         ndi[:,t]=simulation.graph.get_ndi(t)[:,0]
#     return ndi

# def get_gdi(simulation):
#     timesteps=len(simulation.get_opinions()[0])
#     gdi=np.zeros((simulation.opinion_size,timesteps))
#     for t in range(0,timesteps):
#         gdi[:,t]=simulation.graph.get_gdi(t)[:,0]
#     return gdi

# def get_time_full_memory(simulation):
#     t=0
#     full=0
#     timesteps=len(simulation.get_opinions()[0])
#     while t<timesteps and full==0:
#         c=0
#         for node in simulation.graph.nodes:
#             if len(node.get_info(t))==7:
#                 c+=1
#         if c==simulation.size:
#             full=t
#         t+=1
#     return t

# def type_stabilise(simulation,length=5,tol_stab=0.1,tol_pol=0.2,tol_gdi=0,ndi=None,gdi=None):
#     """
#     Return a mtrix encoding the type of stabilisation
#     0: consensus
#     1: uni-polarisation
#     2: bi-polarisation
#     -1:no stabilisation


#     """
#     consensus=np.zeros((simulation.opinion_size))
#     bipolarisation=np.zeros((simulation.opinion_size))
#     unipolarisation=np.zeros((simulation.opinion_size))
#     no_stabilisation=np.zeros((simulation.opinion_size))
#     timesteps=len(simulation.get_opinions()[0])    
#     t=timesteps-length
#     if type(ndi)!=type(np.array([1])):
#         ndi=get_ndi(simulation)
#     if type(gdi)!=type(np.array([1])):
#         gdi=get_gdi(simulation)
#     for i in range(0,simulation.opinion_size):
#         if abs(ndi[i,t]-ndi[i,t+length-1])/length<=tol_stab:
#             if gdi[i,-1]>gdi[i,0]*(1+tol_gdi):
#                 bipolarisation[i]=1
#             else:
#                 if sum([simulation.graph.nodes[k].get_opinion()[i] for k in range(0,simulation.size)])/simulation.size < tol_pol:
#                     unipolarisation[i]=1
#                 else:
#                     consensus[i]=1

#         else:
#             no_stabilisation[i]=1 
#     return no_stabilisation,consensus,unipolarisation,bipolarisation
    
# def get_df(file_min=2500,
#            file_max=2505,
#            path="../../../outputs/",
#            opinion_param_var = "update_opinion_(coef,bias)",
#            msg_param_var = "msg_gen_one_info_dist_global_coef"):
    
#     simulation_list = []
#     opinion_param=[]
#     msg_param=[]
#     for i in range(file_min,file_max+1):
#         try:
#             f=open(path+f"simulation_{i}.txt","r")
#             report = f.readlines()
#             for line in report:
#                 if opinion_param_var+" =" in line:
#                     opinion_param.append(eval(line.split('=')[1].replace('\n','')))
#                 if msg_param_var+" =" in line:
#                     msg_param.append(eval(line.split('=')[1].replace('\n','')))    
#             #if opinion_func in report:
#             simulation_list.append(load(path+f"simulation_{i}.pkl"))
#             print(f"simulation {i} loaded")
#         except:
#             pass
#     stab_list=[]
#     stab_type=np.zeros((len(simulation_list),4))
#     full_mem=[]

#     ndi=[]
#     gdi=[]
#     for i,simulation in enumerate(simulation_list):
#         full_mem.append(get_time_full_memory(simulation))
#         ndi.append(get_ndi(simulation))
#         gdi.append(get_gdi(simulation))
#         stab_list.append(stabilise(simulation,ndi=ndi[-1]))
#         #type_stabilise(simulation,ndi=ndi[-1])
#         stab_type[i,:]=np.array([np.array(type_stabilise(simulation,ndi=ndi[-1],gdi=gdi[-1])).transpose()[:,i].mean() for i in range(0,4)])
        
#     #print(stab_type)
#     #print(ndi)
#     #print(simulation_list)
#     #print(stab_list)
#     avg_stab=[i.mean() for i in stab_list]
#     #print(avg_stab)
#     #print(opinion_param)
#     #print(msg_param)
#     stab_time=[time_stab(simulation_list[i],ndi=ndi[i]) for i in range(0,len(simulation_list))]
#     avg_stab_time=[i.mean() for i in stab_time]
#     std_stab_time=[i.std() for i in stab_time]
#     opinion_param=np.array(opinion_param)
#     df=pd.DataFrame({"opinion_param_1":opinion_param[:,0],
#                   "opinion_param_2":opinion_param[:,1],      
#                   "msg_param":msg_param,
#                   "full_memory_time":np.array(full_mem),
#                   "proba_stab":avg_stab,
#                   "avg_stab_time":avg_stab_time,
#                   "std_stab_time":std_stab_time,
#                   "proba_no_stab":stab_type[:,0],
#                   "proba_consensus":stab_type[:,1],
#                   "proba_unipolarisation":stab_type[:,2],
#                   "proba_bipolarisation":stab_type[:,3]
#                   })    
#     return df

# if __name__=="__main__":
#     os.chdir(os.path.abspath(os.path.dirname(__file__)))
#     print(os.getcwd())
#     path="../../../outputs/"
#     file_min=1500
#     file_max=1505 #1793
#     max_cache=2#10
#     opinion_param_var = "update_opinion_(coef,bias)"
#     msg_param_var = "msg_gen_one_info_dist_global_coef"
#     filename_export="biased_dist_normal.csv"

    
    
#     if file_max-file_min<max_cache:
#         df = get_df(file_min,file_max,path=path,opinion_param_var=opinion_param_var,msg_param_var=msg_param_var)
#     else:
#         df=pd.DataFrame({"opinion_param_1":[],
#                   "opinion_param_2":[],  
#                   "msg_param":[],
#                   "full_memory_time":[],
#                   "proba_stab":[],
#                   "avg_stab_time":[],
#                   "std_stab_time":[],
#                   "proba_no_stab":[],
#                   "proba_consensus":[],
#                   "proba_unipolarisation":[],
#                   "proba_bipolarisation":[]
#                   })    
#         for i in range(0,(file_max-file_min)//max_cache):
#             df_temp = get_df(file_min=file_min+i*max_cache,
#                    file_max=file_min+i*max_cache+max_cache-1,
#                    path=path,
#                    opinion_param_var=opinion_param_var,
#                    msg_param_var=msg_param_var)
#             df=pd.concat([df,df_temp])
#             print(f'batch {i}')
#         df_temp = get_df(file_min=file_min+((file_max-file_min)//max_cache)*max_cache,
#                 file_max=file_max,
#                 path=path,
#                 opinion_param_var=opinion_param_var,
#                 msg_param_var=msg_param_var)
#         df=pd.concat([df,df_temp])
    
#     print(df)
#     df.to_csv(filename_export,index=False)

#     # for simulation in simulation_list:
#     #   simulation.time_plotting().show()
#     #   input("Press key to continue")

    


# # import numpy as np
# # import pandas as pd
# # from functools import partial
# # import os
# # os.chdir(os.path.abspath(os.path.dirname(__file__)))
# # import matplotlib.pyplot as plt
# # import sys
# # sys.path.append("..//classes")
# # from node import Node
# # from graph import Graph 
# # from simulation import Simulation,load
# # sys.path.append("""..//functions""")
# # import state_init
# # import update_info
# # import update_opinion
# # import msg_gen

# # def stabilise(simulation,t=-5,length=5,tol=0.1,ndi=None):
# #     """
# #     Evaluate if a simulation stabilise(1/0)

# #     length: int, nb of timestep for considering stabilisation
# #     tol: float, percetage of variation below stabilisation in considered as true. 
# #     """
# #     stab_mat=np.zeros((simulation.opinion_size,1))
# #     if type(ndi)!=type(np.array([1])):
# #         ndi=get_ndi(simulation)
# #     for i in range(0,simulation.opinion_size):
# #         if abs(ndi[i,t]-ndi[1,t+length-1])<=tol*0.5*ndi[:,0].mean():
# #             stab_mat[i,0]=1
# #         else:
# #             stab_mat[i,0]=0
# #     return stab_mat
   

# # def time_stab(simulation,length=5,tol_stab=0.1,ndi=None):
# #     timesteps =len(simulation.get_opinions()[0])
# #     stab_time=[]
# #     if type(ndi)!=type(np.array([1])):
# #         ndi=get_ndi(simulation)
# #     for i in range(0,simulation.opinion_size):
# #         t=timesteps-length
# #         while abs(ndi[i,t]-ndi[i,t+length-1])<=tol_stab*ndi[i,t+length-1] and t>0 :#tol_stab*0.5*ndi[:,0].mean() and t>0:
# #             t=t-1
# #         stab_time.append(t)
# #     return np.array(stab_time)


# # def get_ndi(simulation):
# #     timesteps=len(simulation.get_opinions()[0])
# #     ndi=np.zeros((simulation.opinion_size,timesteps))
# #     for t in range(0,timesteps):
# #         ndi[:,t]=simulation.graph.get_ndi(t)[:,0]
# #     return ndi

# # def type_stabilise(simulation,length=5,tol_stab=0.1,tol_pol=0.2,ndi=None):
# #     """
# #     Return a mtrix encoding the type of stabilisation
# #     0: consensus
# #     1: uni-polarisation
# #     2: bi-polarisation
# #     -1:no stabilisation


# #     """
# #     consensus=np.zeros((simulation.opinion_size))
# #     bipolarisation=np.zeros((simulation.opinion_size))
# #     unipolarisation=np.zeros((simulation.opinion_size))
# #     no_stabilisation=np.zeros((simulation.opinion_size))
# #     timesteps=len(simulation.get_opinions()[0])    
# #     t=timesteps-length
# #     if type(ndi)!=type(np.array([1])):
# #         ndi=get_ndi(simulation)
# #     for i in range(0,simulation.opinion_size):
# #         if abs(ndi[i,t]-ndi[i,t+length-1])<=tol_stab*ndi[i,t+length-1]:#tol_stab*0.5*ndi[:,0].mean():
# #             if ndi[i,-1]>ndi[i,0]:
# #                 bipolarisation[i]=1
# #             else:
# #                 if sum([simulation.graph.nodes[k].get_opinion()[i] for k in range(0,simulation.size)])/simulation.size < tol_pol:
# #                     unipolarisation[i]=1
# #                 else:
# #                     consensus[i]=1

# #         else:
# #             no_stabilisation[i]=1 
# #     return no_stabilisation,consensus,unipolarisation,bipolarisation
  
# # def get_df(file_min=1502,
# #            file_max=1502,
# #            path="../../../outputs/",
# #            opinion_param_var = "update_opinion_(coef,bias)",
# #            msg_param_var = "msg_gen_one_info_dist_global_coef"):
    
# #     simulation_list = []
# #     opinion_param=[]
# #     msg_param=[]
# #     for i in range(file_min,file_max+1):
# #         try:
# #             f=open(path+f"simulation_{i}.txt","r")
# #             report = f.readlines()
# #             for line in report:
# #                 if opinion_param_var+" =" in line:
# #                     opinion_param.append(eval(line.split('=')[1].replace('\n','')))
# #                 if msg_param_var+" =" in line:
# #                     msg_param.append(eval(line.split('=')[1].replace('\n','')))    
# #             #if opinion_func in report:
# #             simulation_list.append(load(path+f"simulation_{i}.pkl"))
# #             print(f"simulation {i} loaded")
# #         except:
# #             pass
# #     stab_list=[]
# #     stab_type=np.zeros((len(simulation_list),4))
# #     ndi=[]
    
# #     for i,simulation in enumerate(simulation_list):
# #         ndi.append(get_ndi(simulation))
# #         stab_list.append(stabilise(simulation,ndi=ndi[-1]))
# #         #type_stabilise(simulation,ndi=ndi[-1])
# #         stab_type[i,:]=np.array([np.array(type_stabilise(simulation,ndi=ndi[-1])).transpose()[:,j].mean() for j in range(0,4)])
        
# #     #print(stab_type)
# #     #print(ndi)
# #     #print(simulation_list)
# #     #print(stab_list)
# #     avg_stab=[i.mean() for i in stab_list]
# #     #print(avg_stab)
# #     #print(opinion_param)
# #     #print(msg_param)
# #     stab_time=[time_stab(simulation_list[i],ndi=ndi[i]) for i in range(0,len(simulation_list))]
# #     avg_stab_time=[i.mean() for i in stab_time]
# #     std_stab_time=[i.std() for i in stab_time]
# #     opinion_param=np.array(opinion_param)
# #     df=pd.DataFrame({"opinion_param_1":opinion_param[:,0],
# #                   "opinion_param_2":opinion_param[:,1],   
# #                   "msg_param":msg_param,
# #                   "proba_stab":avg_stab,
# #                   "avg_stab_time":avg_stab_time,
# #                   "std_stab_time":std_stab_time,
# #                   "proba_no_stab":stab_type[:,0],
# #                   "proba_consensus":stab_type[:,1],
# #                   "proba_unipolarisation":stab_type[:,2],
# #                   "proba_bipolarisation":stab_type[:,3]
# #                   })    
# #     return df

# # if __name__=="__main__":
# #     os.chdir(os.path.abspath(os.path.dirname(__file__)))
# #     print(os.getcwd())
# #     path="../../../outputs/"
# #     file_min=1500
# #     file_max=1793
# #     max_cache=10 #40
# #     opinion_param_var = "update_opinion_(coef,bias)"
# #     msg_param_var = "msg_gen_one_info_dist_global_coef"
# #     filename_export="biased_dist_normal.csv"

    
    
# #     if file_max-file_min<max_cache:
# #         df = get_df(file_min,file_max,path=path,opinion_param_var=opinion_param_var,msg_param_var=msg_param_var)
# #     else:
# #         df=pd.DataFrame({"opinion_param_1":[],
# #                   "opinion_param_2":[],
# #                   "msg_param":[],
# #                   "proba_stab":[],
# #                   "avg_stab_time":[],
# #                   "std_stab_time":[],
# #                   "proba_no_stab":[],
# #                   "proba_consensus":[],
# #                   "proba_unipolarisation":[],
# #                   "proba_bipolarisation":[]
# #                   })    
# #         for i in range(0,(file_max-file_min)//max_cache):
# #             df_temp = get_df(file_min=file_min+i*max_cache,
# #                    file_max=file_min+i*max_cache+max_cache-1,
# #                    path=path,
# #                    opinion_param_var=opinion_param_var,
# #                    msg_param_var=msg_param_var)
# #             df=pd.concat([df,df_temp])
# #             print(f'batch {i}')
# #         df_temp = get_df(file_min=file_min+((file_max-file_min)//max_cache)*max_cache,
# #                 file_max=file_max,
# #                 path=path,
# #                 opinion_param_var=opinion_param_var,
# #                 msg_param_var=msg_param_var)
# #         df=pd.concat([df,df_temp])
# #     print(df[["proba_stab","proba_no_stab","proba_consensus","proba_unipolarisation","proba_bipolarisation"]])
# #     df.to_csv(filename_export,index=False)

# #     # for simulation in simulation_list:
# #     #   simulation.time_plotting().show()
# #     #   input("Press key to continue")
        

# # # if __name__=="__main__":
# # #     os.chdir(os.path.abspath(os.path.dirname(__file__)))
# # #     print(os.getcwd())
# # #     path="../../../outputs/"
# # #     file_min=1500
# # #     file_max=1793

# # #     opinion_param_var = "update_opinion_(coef,bias)"
# # #     msg_param_var = "msg_gen_one_info_dist_global_coef"
# # #     filename_export="biased_dist.csv"

# # #     simulation_list = []
# # #     opinion_param=[]
# # #     msg_param=[]
# # #     for i in range(file_min,file_max+1):
# # #         try:
# # #             f=open(path+f"simulation_{i}.txt","r")
# # #             report = f.readlines()
# # #             for line in report:
# # #                 if opinion_param_var+" =" in line:
# # #                     opinion_param.append(eval(line.split('=')[1].replace('\n','')))
# # #                 if msg_param_var+" =" in line:
# # #                     msg_param.append(eval(line.split('=')[1].replace('\n','')))    
# # #             #if opinion_func in report:
# # #             simulation_list.append(load(path+f"simulation_{i}.pkl"))
# # #             print(f"simulation {i} loaded")
# # #         except:
# # #             pass
# # #     stab_list=[]
# # #     stab_type=np.zeros((len(simulation_list),4))
# # #     ndi=[]
    
# # #     for i,simulation in enumerate(simulation_list):
# # #         ndi.append(get_ndi(simulation))
# # #         stab_list.append(stabilise(simulation,ndi=ndi[-1]))
# # #         #type_stabilise(simulation,ndi=ndi[-1])
# # #         stab_type[i,:]=np.array([np.array(type_stabilise(simulation,ndi=ndi[-1])).transpose()[:,i].mean() for i in range(0,4)])
        
# # #     #print(stab_type)
# # #     #print(ndi)
# # #     #print(simulation_list)
# # #     #print(stab_list)
# # #     avg_stab=[i.mean() for i in stab_list]
# # #     #print(avg_stab)
# # #     #print(opinion_param)
# # #     #print(msg_param)
# # #     stab_time=[time_stab(simulation_list[i],ndi=ndi[i]) for i in range(0,len(simulation_list))]
# # #     avg_stab_time=[i.mean() for i in stab_time]
# # #     std_stab_time=[i.std() for i in stab_time]
# # #     opinion_param=np.array(opinion_param)
# # #     df=pd.DataFrame({"opinion_param_1":opinion_param[:,0],
# # #                      "opinion_param_2":opinion_param[:,1],
# # #                   "msg_param":msg_param,
# # #                   "proba_stab":avg_stab,
# # #                   "avg_stab_time":avg_stab_time,
# # #                   "std_stab_time":std_stab_time,
# # #                   "proba_no_stab":stab_type[:,0],
# # #                   "proba_consensus":stab_type[:,1],
# # #                   "proba_unipolarisation":stab_type[:,2],
# # #                   "proba_bipolarisation":stab_type[:,1]
# # #                   })
# # #     print(df)
# # #     #df.to_csv(filename_export,index=False)

# # #     # for simulation in simulation_list:
# # #     #   simulation.time_plotting().show()
# # #     #   input("Press key to continue")

    
