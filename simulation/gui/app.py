import numpy as np
import networkx as nx
#from pyvis import network as net
from functools import partial
import streamlit as st
import os
from inspect import getmembers, isfunction, getdoc
os.chdir(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
import sys
sys.path.append(".\\gui")
from pyvis_ext import NetworkExt 
sys.path.append(".\\classes")
from node import Node
from graph import Graph 
from simulation import Simulation
sys.path.append(""".\\functions""")
import state_init
import update_info
import update_opinion
import msg_gen


pg = st.navigation(["pages/.py", page_2])
pg.run()