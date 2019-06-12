import networkx as nx 
import numpy as np 
import mlp_graph as mlp

ann_config_path = 'xor_ann.yml'
graph_out_path = 'xor_ann.dot'

xor_mlp = mlp.NeuralNetGraph(ann_config_path)
xor_mlp.CreateGraphNodes()
xor_mlp.CreateGraphEdges()
xor_mlp.ComputeForwardPass([100,200])
xor_mlp.ExportGraphAsDot(graph_out_path)