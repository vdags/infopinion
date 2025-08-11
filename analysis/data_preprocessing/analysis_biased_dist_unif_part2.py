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


def get_heatmap(df,attr_1,attr_2,attr_out):
    rows= df[attr_1].unique().tolist()
    cols=df[attr_2].unique().tolist()
    mat=np.zeros((len(rows),len(cols)))
    for i in range(0,df.shape[0]):
        row_ind=rows.index(df[attr_1].iloc[i])
        col_ind=cols.index(df[attr_2].iloc[i])
        mat[row_ind,col_ind]=df[attr_out].iloc[i]
    return pd.DataFrame(mat,columns = cols, index = rows)


if __name__=="__main__":
    os.chdir(os.path.abspath(os.path.dirname(__file__)))
    print(os.getcwd())
    df= pd.read_csv("biased_dist_unif.csv")
    #print(df.shape)
    #df.dropna(axis=0,inplace=True)
    #print(df.shape)
    #df["opinion_param_2"]=[0.01,0.02,0.05,0.1,0.2,0.5,1]*int((df.shape[0]-1)/7)
    

    mat=get_heatmap(df,"opinion_param_1","msg_param","proba_unipolarisation")
    print(mat)
    print(df.describe()[["full_memory_time","proba_no_stab","proba_consensus","proba_unipolarisation","proba_bipolarisation","avg_stab_time","std_stab_time"]])
    #print(df[["opinion_param_1","msg_param","proba_no_stab","avg_stab_time","proba_consensus",]].query("proba_consensus<1"))
    #mat.style.background_gradient(cmap ='viridis')#.set_properties(**{'font-size': '20px'})


    fig, ax = plt.subplots(2,3,figsize=(14, 7))

    # Displaying dataframe as an heatmap 
    # with diverging colourmap as RdYlGn
    sns.heatmap(mat, cmap ='Reds', linewidths = 0.30, annot = True,ax=ax[0,0])
    ax[0,0].set_ylabel("Relative weight c, bias b")
    ax[0,0].set_xlabel("Distance-based sending threshold p")
    ax[0,0].set_title("Probability of uni-polarisation")
    #plt.savefig("avg_time_to_stab_avg_dist_normal.png")
    #plt.show()
    
    mat=get_heatmap(df,"opinion_param_1","msg_param","proba_bipolarisation")
    sns.heatmap(mat, cmap ='Reds', linewidths = 0.30, annot = True,ax=ax[0,1])
    ax[0,1].set_ylabel("Relative weight c, bias b")
    ax[0,1].set_xlabel("Distance-based sending threshold p")
    ax[0,1].set_title("Probability of bi-polarisation")
    #plt.savefig("avg_time_to_stab_avg_dist_normal.png")
    #plt.show()

    mat=get_heatmap(df,"opinion_param_1","msg_param","proba_no_stab")
    sns.heatmap(mat, cmap ='Reds', linewidths = 0.30, annot = True,ax=ax[0,2])
    ax[0,2].set_ylabel("Relative weight c, bias b")
    ax[0,2].set_xlabel("Distance-based sending threshold p")
    ax[0,2].set_title("Probability of non stabilisation")
    #plt.savefig("avg_time_to_stab_avg_dist_normal.png")
    #plt.show()

    mat_time=get_heatmap(df,"opinion_param_1","msg_param","avg_stab_time")
    sns.heatmap(mat_time, cmap ='Blues', linewidths = 0.30, annot = True,ax=ax[1,0])
    ax[1,0].set_ylabel("Relative weight c, bias b")
    ax[1,0].set_xlabel("Distance-based sending threshold p")
    ax[1,0].set_title("Time for stability (in nb of timesteps)")
    #plt.savefig("avg_time_to_stab_avg_dist_normal.png")
    #plt.show()

    mat_std=get_heatmap(df,"opinion_param_1","msg_param","std_stab_time")
    sns.heatmap(mat_std, cmap ='Blues', linewidths = 0.30, annot = True,ax=ax[1,1])
    ax[1,1].set_ylabel("Relative weight c, bias b")
    ax[1,1].set_xlabel("Distance-based sending threshold p")
    ax[1,1].set_title("Std of the time for stability (in nb of timesteps)")
    #plt.savefig("std_time_to_stab_avg_dist_normal.png")
    ax[-1, -1].axis('off')
    fig.tight_layout()
    #fig.savefig("biased_dist_unif.png")
    plt.show()

