from dataclasses import dataclass
import networkx as nx
import numpy as np
import yaml
import ann_special_funcs as spc_fc
import scipy.linalg as sp_lin
import copy


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
        self.training_cost = self.config["Training_Cost"]
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
        self.training_cost = self.neural_net.training_cost
    def CreateGraphNodes(self):
        self.neural_net_graph = nx.DiGraph()
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
    
    def CreateANNGraph(self):
        self.CreateGraphNodes()
        self.CreateGraphEdges()


    def ComputeLayerResponse(self,weights,inputs,activation_functions,calc_deriv=False):
        n_weights_ = len(weights)
        weights = weights
        weights_ = sp_lin.block_diag(*weights)
        inputs = np.asarray(inputs).flatten()
        response_vec = np.dot(weights_,inputs)
        act_fs = [spc_fc.activations[act_fct] for act_fct in activation_functions ]
        func_responses =  list(map(lambda f , x: f['f'](x) ,act_fs , list(response_vec)))
        if calc_deriv:
            deriv_responses = list(map(lambda f , x: f['DfDx'](x) ,act_fs , list(response_vec)))
            return {'f_x':func_responses,'Df_Dx':deriv_responses}
        else :
            return {'f_x':func_responses,'Df_Dx':None}

    def ComputeForwardPass(self,graph,input_,calc_deriv=False):
        if graph is None:
            neural_net_graph = self.neural_net_graph
        else:
            neural_net_graph = graph
       
        for layer_idx in range(self.numLayers):
            if layer_idx == 0:
                input_nodes = list(filter (lambda node : node[1]['layer_id'] == layer_idx , self.neural_net_graph.nodes(data=True)))
                for ndx,node in enumerate(input_nodes):
                    node_tag = node[0]
                    neural_net_graph.node[node_tag]["input"] = input_[ndx]
                    neural_net_graph.node[node_tag]["output"] = [input_[ndx]]
            else:
                layer_nodes = list(filter (lambda node : node[1]['layer_id'] == layer_idx , self.neural_net_graph.nodes(data=True)))
                layer_inputs = []
                layer_weights = []
                layer_activations = []
                
                for node in layer_nodes:
                    node_inputs = []
                    weights = []
                    node_tag = node[0]
                    incoming_edges = neural_net_graph.in_edges(node_tag,data=True)
                    layer_activations.append(node[1]['activation_function'])
                    
                    for edge in incoming_edges:
                        edge_weight = edge[2]["weight"]
                        node_input = neural_net_graph.node[edge[0]]["output"][0]
                        print(node_input)
                        weights.append(edge_weight)
                        node_inputs.append(node_input)
                    
                    layer_inputs.append(node_inputs)
                    neural_net_graph.node[node_tag]["input"] = [node_inputs]             
                    layer_weights.append(weights)
               
                layer_response = self.ComputeLayerResponse(layer_weights,layer_inputs,layer_activations,calc_deriv)
                for ndx,node in enumerate(layer_nodes):
                    node_tag = node[0]
                    neural_net_graph.node[node_tag]["output"] = [layer_response['f_x'][ndx],layer_response['Df_Dx'][ndx]]

    def Train(self,input_,truth):
        self.extended_graph = copy.deepcopy(self.neural_net_graph)
        self.ComputeForwardPass(self.extended_graph,input_,calc_deriv=True)
        output_nodes = list(filter(lambda node : node[1]['nodeType'] == 'Output' , self.extended_graph.nodes(data=True)))
        for each_node in output_nodes:
            node_tag = 'L' + str(each_node[1]['layer_id']+1) + ' ' + str(each_node[1]['nodeID'])
            net_output = each_node[1]['output'][0]
            error_f = spc_fc.activations[self.training_cost]['f'](net_output,truth)
            error_d = spc_fc.activations[self.training_cost]['DfDx'](net_output,truth) 

            self.extended_graph.add_node(
                    node_tag,
                    nodeType = "Error",
                    nodeID = each_node[1]['nodeID'],
                    layer_id = each_node[1]['layer_id']+1,
                    activation_function=self.training_cost,
                    input = net_output,
                    output = [error_f,error_d])
            self.extended_graph.add_edge(each_node[0],node_tag,weight=1)

    def ExportGraphAsDot(self, out_path, graph=None):
        if graph is None:
            nx.nx_agraph.write_dot(self.neural_net_graph,out_path)
        else :
            nx.nx_agraph.write_dot(graph,out_path)