import networkx as nx 
import numpy as np 
import ann

numInputNodes = 3
numHiddenLayers = 1
numNodesPerHiddenLayer = 2
numOutputNodes = 1       
    
    
ANN = ann.NeuralNet(numInputNodes,numHiddenLayers,numNodesPerHiddenLayer,numOutputNodes)

ANN.createInputLayer()
ANN.createOutputLayer()
ANN.createHiddenLayers()

G = nx.Graph()
for each_node in ANN.inputLayer:
    G.add_node(each_node ,
              nodeType = each_node.nodeType,
              nodeID = each_node.nodeID,
              layer_id = each_node.layerID)
for each_layer in ANN.hiddenlayers:
    for each_node in each_layer:
        G.add_node(each_node,
              nodeType = each_node.nodeType,
              nodeID = each_node.nodeID,
              layer_id = each_node.layerID)
for each_node in ANN.outputLayer:
    G.add_node(each_node,
              nodeType = each_node.nodeType,
              nodeID = each_node.nodeID,
              layer_id = each_node.layerID)
for node_idx ,each_node in enumerate(G.nodes()):
    if each_node.nodeType != 'Output':
        neighbors = list(filter (lambda node : node.layerID == each_node.layerID+1 , G.nodes()))
        for neigh_node in neighbors:
            G.add_edge(each_node,neigh_node)
        
    print(each_node.layerID,each_node.nodeType)




print(G.number_of_nodes())
print(G.number_of_edges())
nx.nx_agraph.write_dot(G,'ann.dot')