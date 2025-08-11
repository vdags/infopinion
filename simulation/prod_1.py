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
    timesteps=50

    list_func_init=[state_init.independent_normal_distribution] #state_init.independent_normal_distribution
    list_info_update=[update_info.latest_info_random]
    
    update_opinion_coord_avg_coef=[0.01,0.05,0.1,0.15,0.2,0.3,0.4,0.5,0.8,1]
    list_opinion_update=[lambda x : update_opinion.coord_avg(x,i) for i in update_opinion_coord_avg_coef]
    
    
    #msg_gen_one_info_random_coef=np.array([0.1,0.5,1,5])*(1/nb_nodes)
    msg_gen_one_info_dist_global_coef= [0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    list_msg_generator=[lambda x :msg_gen.one_info_dist_global(x,nb_nodes,i) for i in msg_gen_one_info_dist_global_coef]
                      #[lambda x : msg_gen.one_info_random(x,nb_nodes,i) for i in msg_gen_one_info_random_coef]
                      
                    
    filenb=2500
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
                    report.write(f"update_opinion_coord_avg = {update_opinion_coord_avg_coef[opinion_update]}\n")  
                  except:
                    report.write(f"update_opinion_coord_avg = None\n")
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
