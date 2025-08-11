import numpy as np
import pandas as pd
from functools import partial
import os
os.chdir(os.path.abspath(os.path.dirname(__file__)))
import matplotlib.pyplot as plt
import seaborn as sns

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

def time_stab_mat(df):
    rows= df.msg_param.unique().tolist()
    cols=df.opinion_param_1.unique().tolist()
    mat=np.zeros((len(rows),len(cols)))
    for i in range(0,df.shape[0]):
        row_ind=rows.index(df.msg_param.iloc[i])
        col_ind=cols.index(df.opinion_param_1.iloc[i])
        mat[row_ind,col_ind]=df.avg_stab_time.iloc[i]
    return pd.DataFrame(mat,columns = cols, index = rows)

def time_stab_mat_2(df):
    rows= df.msg_param.unique().tolist()
    cols=df.opinion_param_1.unique().tolist()
    mat=np.zeros((len(rows),len(cols)))
    for i in range(0,df.shape[0]):
        row_ind=rows.index(df.msg_param.iloc[i])
        col_ind=cols.index(df.opinion_param_1.iloc[i])
        mat[row_ind,col_ind]=df.std_stab_time.iloc[i]
    return pd.DataFrame(mat,columns = cols, index = rows)


if __name__=="__main__":
    os.chdir(os.path.abspath(os.path.dirname(__file__)))
    print(os.getcwd())
    df= pd.read_csv("avg_rand_normal.csv")
    mat=time_stab_mat(df)
    print(mat)
    print(df.describe()[["full_memory_time","proba_consensus","avg_stab_time","std_stab_time"]])
    #mat.style.background_gradient(cmap ='viridis')#.set_properties(**{'font-size': '20px'})

    # Displaying dataframe as an heatmap 
    # with diverging colourmap as RdYlGn
    sns.heatmap(mat, cmap ='Blues', linewidths = 0.30, annot = True)
    plt.xlabel("Relative weight c")
    plt.ylabel("Probability p of edge existance")
    plt.title("Time for convergence (in nb of timesteps)")
    #plt.savefig("avg_time_to_stab_avg_rand_normal.png")
    plt.show()

    mat_std=time_stab_mat_2(df)
    sns.heatmap(mat_std, cmap ='Blues', linewidths = 0.30, annot = True)
    plt.xlabel("Relative weight c")
    plt.ylabel("Probability p of edge existance")
    plt.title("Std of the time for convergence (in nb of timesteps)")
    #plt.savefig("std_time_to_stab_avg_rand_normal.png")
    plt.show()

