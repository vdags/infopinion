import numpy as np
import networkx as nx
#from pyvis import network as net
from functools import partial
import streamlit as st
import wx
import os
from inspect import getmembers, isfunction, getdoc
os.chdir(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
import sys
#sys.path.append(".\\gui")
from pyvis_ext import NetworkExt 
sys.path.append(".\\classes")
from node import Node
from graph import Graph 
from simulation import Simulation
from simulation import load
sys.path.append(""".\\functions""")
import state_init
import update_info
import update_opinion
import msg_gen



MAX_VALUE_PLACEHOLDER=10**10 # Arbitrary max value for number_input fields.
pyvis_options = {
          "nodes":{
              "font":{
                  "size": 50,
                  "bold":True
              }
          },
          "edges":{
              "color":'red',
              "smooth":False
          },
          "physics":{
              "barnesHut":{
                  "gravitationalConstant":-500000,
                  "centralGravity":12,
                  "springLength": 50,
                  "springConstant": 0.7,
                  "damping": 3,
                  "avoidOverlap": 10
              }
          },
          "interaction":{   
               "selectConnectedEdges": True

}}


# Session variable initialisation
if "same_encoding_size" not in st.session_state:
    st.session_state.same_encoding_size= False
if "running" not in st.session_state:
    st.session_state.running= False
if "simulation" not in st.session_state:
    st.session_state.simulation= None
if "timestep" not in st.session_state:
    st.session_state.timestep= -1

# Callbacks
def btn_callbk():
    st.session_state.same_encoding_size = not st.session_state.same_encoding_size

@st.cache_data
def last_version(_simulation):
    _simulation.update()
    return _simulation

def run(nb_nodes,nb_steps,func_init,update_opinion,update_info,msg_gen,component_graph,component_chart):
    st.session_state.running= True
    simulation=Simulation(nb_nodes,
                          func_init,#partial(state_init.dummy, o0=0.5,i0=0.1),
                          update_opinion,
                          update_info,
                          msg_gen
                          )
    for i in range(0,nb_steps):
        last_version.clear()
        simulation = last_version(simulation)
    #    if i%10 ==0:

    st.session_state.simulation=simulation
    draw(simulation)
    st.session_state.running=False
    print("DONE!")

def draw(simulation,timestep=-1):
    
    network = nx.from_numpy_array(simulation.graph.get_dist_matrix()) # Create network from distance matrix
    #Convert it into pyvis.network.Network extended class type for html generation
    pyvis_net = NetworkExt(height='400px', width='100%',notebook=True,heading='',cdn_resources='remote')
    #pyvis_net.from_nx(network)
    pyvis_net.from_nx_ext(network,hide_edges=True)
    pyvis_net.show_buttons(filter_=['physics'])
    #pyvis_net.toggle_hide_edges_on_drag(True)
    pyvis_net.show("tmp.html")
    # Import html for rendering into streamlit 
    HtmlFile = open("tmp.html", 'r', encoding='cp1252')
    source_code = HtmlFile.read()
    # if timestep==-1:
    #     st.slider("Timestep",0,len(simulation.get_opinions()[0]),len(simulation.get_opinions()[0]),1)
    # else:
    #     st.slider("Timestep",0,len(simulation.get_opinions()[0]),timestep,1)
    #st.pyplot(simulation.time_plotting())
    #component_graph.empty()
    st.empty()
    #with component_graph:
    st.components.v1.html(source_code, height = 900,width=900)
        
    

def save():
    if st.session_state.simulation==None:
        st.warning("No simulation to save. Please compute or load one.")
    else:
        app=[]; app = wx.App(None)
        dialog = wx.FileDialog(None, "Open", "", "","Pickle files (*.pkl)|*.pkl",wx.FD_SAVE)
        if dialog.ShowModal() == wx.ID_OK:
            filepath = dialog.GetPath() # folder_path will contain the path of the folder you have selected as string
        #listfiles=os.listdir(path)
        #listfiles= [i.replace("simulation_","").replace(".pkl","") if ("simulation_" and ".pkl" in i) else "" for i in listfiles].remove("")
        #max_index=max(listfiles)
            simulation = st.session_state.simulation
            simulation.save(filepath)
            draw(simulation,-1)
    

def loader():
    app=[]; app = wx.App(None)
    dialog = wx.FileDialog(None, "Open", "", "","Pickle files (*.pkl)|*.pkl",wx.FD_OPEN | wx.FD_FILE_MUST_EXIST)
    if dialog.ShowModal() == wx.ID_OK:
        filepath = dialog.GetPath() # folder_path will contain the path of the folder you have selected as string
        if filepath.split(".")[-1]!="pkl":
            st.error("Please select a pickle file (.pkl)")
        else:
            st.session_state.simulation=load(filepath)
            draw(st.session_state.simulation,-1)
        

# UI

## Global layout
filesys=st.sidebar
#settings, netvis , charts = st.columns([1, 3, 1])
settings=st.sidebar
charts=st.container()
netvis=st.container()
# filesystem
filesys.title("Menu")
filesys.button("Load",on_click=loader)
filesys.button("save",on_click=save)

## Settings' column
settings.title("Parameters")
settings.markdown("Here are simulation constants or functions")
nb_nodes=settings.number_input('Number of nodes', 2, MAX_VALUE_PLACEHOLDER)
nb_steps=settings.number_input('Number of time steps', 2, MAX_VALUE_PLACEHOLDER)
n=settings.number_input('Opinion encoding size',1,MAX_VALUE_PLACEHOLDER)
m=settings.number_input('Information encoding size',1,MAX_VALUE_PLACEHOLDER,disabled=st.session_state.same_encoding_size,value=n)
equal_encoding_size = settings.checkbox("Same size", on_change=btn_callbk)
if equal_encoding_size:
    st.session_state.same_encoding_size=True
    m=n

### Create dict of available functions for initialisating vectors, updating information and opinions and genrating msgs.
state_init_func=dict()
for i in getmembers(state_init, isfunction):
    state_init_func[i[0].replace("_"," ")]=i

opinion_update_models=dict()
for i in getmembers(update_opinion, isfunction):
    opinion_update_models[i[0].replace("_"," ")]=i

info_update_models=dict()
for i in getmembers(update_info, isfunction):
    info_update_models[i[0].replace("_"," ")]=i

msg_generator_func=dict()
msg_generator_doc=dict()
for i in getmembers(msg_gen, isfunction):
    msg_generator_func[i[0].replace("_"," ")]=i
    msg_generator_doc[i[0].replace("_"," ")]=getdoc(i)
#print(msg_generator_doc)
### Get docs from functions just retrieved in dict



opinion_update = settings.selectbox('Opinion update function', list(opinion_update_models.keys()))
info_update=settings.selectbox("Information update function",list(info_update_models.keys()))
msg_generator =settings.selectbox("Message generator function",list(msg_generator_func.keys()))
funct_init = settings.selectbox('Opinion Init function', list(state_init_func.keys()))

if st.session_state.simulation==None:
    timestep = st.slider("Timestep",0,1,1,1)
else:
    max_t=len(st.session_state.simulation.get_opinions()[0])
    timestep = st.slider("Timestep",0,max_t,max_t,1)
    draw(st.session_state.simulation,timestep)

runbt=settings.button("Run",
            on_click=run,
            args=(nb_nodes,
                  nb_steps,
                  eval("state_init."+state_init_func[funct_init][0]), 
                  eval("update_opinion."+opinion_update_models[opinion_update][0]),
                  eval("update_info."+info_update_models[info_update][0]), 
                  eval("msg_gen."+msg_generator_func[msg_generator][0]),
                  netvis, charts
                        )
)


