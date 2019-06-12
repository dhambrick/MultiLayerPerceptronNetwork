from dataclasses import dataclass
import networkx as nx
import numpy as np
import yaml
import ann_special_funcs as spc_fc


@dataclass
class NeuralNode:
    nodeType :str
    nodeID : int
    layerID : int
    activation_function : str

def LoadANNConfigFile(config_path):
        stream = open(config_path,"r")
        ann_config = yaml.load(stream)
        stream.close() 
        return ann_config

class NeuralNet:
    def __init__(self,config_path):
        self.config =   LoadANNConfigFile(config_path)
        self.numInputNodes = self.config["Layers"]["Input"]["NumInputNodes"]
        self.numHiddenLayers = self.config["Layers"]["Hidden"]["NumHiddenLayers"]
        self.numNodesPerHiddenLayer = self.config["Layers"]["Hidden"]["NumNodesPerHiddenLayer"]
        self.numOutputNodes = self.config["Layers"]["Output"]["NumOutputNodes"]
        self.numNodes = self.numInputNodes + self.numHiddenLayers * self.numNodesPerHiddenLayer + self.numOutputNodes 
        self.numLayers = self.numHiddenLayers + 2
        self.inputLayer = []
        self.hiddenlayers = []
        self.outputLayer = []   
    
    def createInputLayer(self):
        layer_id = 0
        self.inputLayer = [NeuralNode('Input',idx,layer_id,None) for idx in range(self.numInputNodes)]
    
    def createHiddenLayers(self):
        activation_function = self.config["Layers"]["Hidden"]["ActivationFunction"]
        if self.numHiddenLayers > 0:
            self.hiddenlayers = []
            for layerID in range(1,self.numHiddenLayers+1):
                self.hiddenlayers.append([NeuralNode('Hidden',nodeID,layerID,activation_function) for nodeID in range(self.numNodesPerHiddenLayer)])
   

    def createOutputLayer(self):
        activation_function = self.config["Layers"]["Output"]["ActivationFunction"]
        layer_id = self.numLayers-1
        self.outputLayer = [NeuralNode('Output',idx,layer_id,activation_function) for idx in range(self.numOutputNodes)]
    
    def createNeuralNet(self):
        self.createInputLayer()
        self.createHiddenLayers()
        self.createOutputLayer()
    
class NeuralNetGraph:
    def __init__(self,config_path):
        self.neural_net = NeuralNet(config_path)
        self.neural_net.createNeuralNet()
        self.neural_net_graph = None
        self.weights = None
        self.numLayers = self.neural_net.numLayers
    def CreateGraphNodes(self):
        self.neural_net_graph = nx.Graph()
        #Create the input layer nodes
        for each_node in self.neural_net.inputLayer:
            node_tag = 'L' + str(each_node.layerID) + ' ' + str(each_node.nodeID)
            self.neural_net_graph.add_node(
                    node_tag ,
                    nodeType = each_node.nodeType,
                    nodeID = each_node.nodeID,
                    layer_id = each_node.layerID,
                    activation_function=each_node.activation_function)
        #Create Hidden Layer Nodes
        for each_layer in self.neural_net.hiddenlayers:
            for each_node in each_layer:
                node_tag = 'L' + str(each_node.layerID) + ' ' + str(each_node.nodeID)
                self.neural_net_graph.add_node(
                    node_tag,
                    nodeType = each_node.nodeType,
                    nodeID = each_node.nodeID,
                    layer_id = each_node.layerID,
                    activation_function=each_node.activation_function)
        #Create Output Layer Nodes
        for each_node in self.neural_net.outputLayer:
            node_tag = 'L' + str(each_node.layerID) + ' ' + str(each_node.nodeID)
            self.neural_net_graph.add_node(
                    node_tag,
                    nodeType = each_node.nodeType,
                    nodeID = each_node.nodeID,
                    layer_id = each_node.layerID,
                    activation_function=each_node.activation_function)
    def CreateGraphEdges(self):
        for node_idx ,each_node in enumerate(self.neural_net_graph.nodes(data=True)):
            if each_node[1]['nodeType'] != 'Output':
                node_tag = each_node[0]
                neighbors = list(filter (lambda node : node[1]['layer_id'] == each_node[1]['layer_id']+1 , self.neural_net_graph.nodes(data=True)))
                weights = np.random.normal(size=len(neighbors))
                for idx,neigh_node in enumerate(neighbors):
                    neigh_tag = neigh_node[0]
                    self.neural_net_graph.add_edge(node_tag,neigh_tag,weight=weights[idx])
    def ComputeForwardPass(self,input):
        for layer_idx in range(self.numLayers):
            if layer_idx == 0:
                input_nodes = list(filter (lambda node : node[1]['layer_id'] == layer_idx , self.neural_net_graph.nodes(data=True)))
                for ndx,node in enumerate(input_nodes):
                    node_tag = node[0]
                    self.neural_net_graph.node[node_tag]["input"] = input[ndx]
                    self.neural_net_graph.node[node_tag]["output"] = input[ndx]
           
    def ExportGraphAsDot(self, out_path):
        nx.nx_agraph.write_dot(self.neural_net_graph,out_path)