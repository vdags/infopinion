""" This file is for lauching all tests at the same time. """

import os

if __name__=="__main__":
    os.chdir(os.path.abspath(os.path.dirname(__file__)))
    os.system("python ./tests/test_graph.py")
    os.system("python ./tests/test_msg_gen.py")
    os.system("python ./tests/test_node.py")
    os.system("python ./tests/test_simulation.py")
    os.system("python ./tests/test_state_init.py")
    os.system("python ./tests/test_update_info.py")
    os.system("python ./tests/test_update_opinion.py")