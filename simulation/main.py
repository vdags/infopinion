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

GUI =False

# nb_nodes=1000
# func_init=lambda n : [[np.ones((10,1)) for i in range(n)],[[np.zeros((5,1))] for i in range(n)]]
# info_update=lambda x,y,z : y[-1]
# opinion_update=lambda x,y : x[-1]
# msg_generator=lambda x,y : [(1,np.zeros((5,1)))]

if __name__=="__main__":
    os.chdir(os.path.abspath(os.path.dirname(__file__)))
    print(os.getcwd())
    if GUI==True:
      #os.chdir(os.path.abspath(os.path.dirname(__file__)))
      os.system("streamlit run  ./gui/gui.py")
    else:
      nb_nodes=100
      vector_size=10
      timesteps=50
      update_opinion_couple_param=(0.3,0.5)
      msg_gen_one_info_random_coef=0.3
      msg_gen_one_info_dist_global_coef=0.3
      func_init=state_init.independent_uniform_distribution#independent_normal_distribution
      info_update=update_info.latest_info_random
      opinion_update=lambda x: update_opinion.coord_avg(x,0.3)
      # opinion_update=lambda x: update_opinion.biased_assimilation(x,
      #                                                             update_opinion_couple_param[0],
      #                                                             update_opinion_couple_param[1])
      msg_generator=lambda x : msg_gen.one_info_dist_global(x,nb_nodes,0.05)
      #lambda x : msg_gen.one_info_dist_global(x,nb_nodes,msg_gen_one_info_dist_global_coef)
      simulation=Simulation(nb_nodes,
                           func_init,
                           opinion_update,
                           info_update,
                           msg_generator,
                           vector_size,
                           vector_size
                           )
      for i in range(0,timesteps):
         if i%(timesteps//10)==0:
          print(f"Timestep {i}/{timesteps}")
         simulation.update()
      report=simulation.report()
      print(report)
      simulation.time_plotting().show()
      input("Press key to continue...")
      filename="simulation_2016"
      simulation.save("./output/"+filename)
      with open(f"./output/{filename}.txt","a") as report:
          try:
            report.write(f"update_opinion_(coef,bias) = {update_opinion_couple_param}\n")  
          except:
            report.write(f"update_opinion_(coef,bias) = None\n")
          try:
            report.write(f"msg_gen_one_info_random_coef = {msg_gen_one_info_random_coef}\n")
          except:
            pass
            #report.write(f"msg_gen_one_info_random_coef = None\n")
          try:
            report.write(f"msg_gen_one_info_dist_global_coef = {msg_gen_one_info_dist_global_coef}\n")
          except:
            pass
            #report.write(f"msg_gen_one_info_dist_global_coef = None\n")
          report.close()
