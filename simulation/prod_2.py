import numpy as np
from functools import partial
import os
os.chdir(os.path.abspath(os.path.dirname(__file__)))
import matplotlib.pyplot as plt
import sys
sys.path.append(".//classes")
from node import Node
from graph import Graph 
from simulation import Simulation
sys.path.append(""".//functions""")
import state_init
import update_info
import update_opinion
import msg_gen

if __name__=="__main__":
    os.chdir(os.path.abspath(os.path.dirname(__file__)))
    print(os.getcwd())
    nb_nodes=100
    vector_size=10
    timesteps=100

    list_func_init=[state_init.independent_normal_distribution] #state_init.independent_uniform_distribution
    list_info_update=[update_info.latest_info_random]
    
    update_opinion_coord_biased_coef=[1]#0.01,0.02,0.05,0.1,0.2,0.5,1]
    update_opinion_coord_biased_bias=[1]#0.01,0.02,0.05,0.1,0.2,0.5,1]*len(update_opinion_coord_biased_coef)
    update_opinion_coord_biased_coef = update_opinion_coord_biased_coef*int(len(update_opinion_coord_biased_bias)/len(update_opinion_coord_biased_coef))
    update_opinion_couple_param=list(zip(update_opinion_coord_biased_coef,update_opinion_coord_biased_bias))
    list_opinion_update=[lambda x : update_opinion.biased_assimilation  (x,i[0],i[1]) 
                         for i in update_opinion_couple_param]
    print(update_opinion_couple_param)
    
    #msg_gen_one_info_random_coef=np.array([0.1,0.2,0.5,1,2,5])*(1/nb_nodes)
    msg_gen_one_info_dist_global_coef= [0.2,0.5]#0.001,0.01,0.05,0.1,0.2,0.5]
    list_msg_generator=[lambda x :msg_gen.one_info_dist_global(x,nb_nodes,i) for i in msg_gen_one_info_dist_global_coef]
                    #[lambda x : msg_gen.one_info_random(x,nb_nodes,i) for i in msg_gen_one_info_random_coef]+
                    #[lambda x :msg_gen.one_info_dist_global(x,nb_nodes,i) for i in msg_gen_one_info_dist_global_coef]
    filenb=3198
    for func_init in list_func_init:
      for info_update in list_info_update:
         for opinion_update in range(0,len(list_opinion_update)):
            for msg_generator in range(0,len(list_msg_generator)):
              filename=f"simulation_{filenb}"
              print(filename)
              simulation=Simulation(nb_nodes,
                                  func_init,
                                  list_opinion_update[opinion_update],
                                  info_update,
                                  list_msg_generator[msg_generator],
                                  vector_size,
                                  vector_size
                                  )
              for i in range(0,timesteps):
                if i%(timesteps//10)==0:
                  print(f"Timestep {i}/{timesteps}")
                simulation.update()
              report=simulation.report()
              #print(report)
              simulation.time_plotting()#.show()
              simulation.save("./output/"+filename)
              with open(f"./output/{filename}.txt","a") as report:
                  try:
                    report.write(f"update_opinion_(coef,bias) = {update_opinion_couple_param[opinion_update]}\n")  
                  except:
                    report.write(f"update_opinion_(coef,bias) = None\n")
                  try:
                    report.write(f"msg_gen_one_info_random_coef = {msg_gen_one_info_random_coef[msg_generator]}\n")
                  except:
                    pass
                  try:
                    report.write(f"msg_gen_one_info_dist_global_coef = {msg_gen_one_info_dist_global_coef[msg_generator]}\n")
                  except:
                    pass
                  report.close()
              del simulation
              filenb+=1
