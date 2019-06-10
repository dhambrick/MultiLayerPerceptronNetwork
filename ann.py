import numpy as np
from scipy.stats import  logistic


class NeuralNet:
    def __init__(self,numInputNodes, numHiddenLayers, numNodesPerHiddenLayer ,numOutputNodes):
        self.numInputNodes = numInputNodes
        self.numHiddenLayers = numHiddenLayers
        self.numNodesPerHiddenLayer = numNodesPerHiddenLayer
        self.numOutputNodes = numOutputNodes
        self.numNodes = numInputNodes + numHiddenLayers * numNodesPerHiddenLayer + numOutputNodes 
        self.numLayers = numHiddenLayers + 2
    def createInputLayer(self):
        if self.numHiddenLayers == 0:
            fanIn = None
            fanOut = self.numOutputNodes
            self.inputLayer = [NeuralNode('Input',idx,0,fanIn,fanOut) for idx in range(self.numInputNodes)]
        else:
            fanIn = None
            fanOut = self.numNodesPerHiddenLayer
            self.inputLayer = [NeuralNode('Input',idx,0,fanIn,fanOut) for idx in range(self.numInputNodes)]
    
    def createHiddenLayers(self):
        if self.numHiddenLayers > 0:
            self.hiddenlayers = []
            #The first hidden layer and the last hidden layer are special , so we will deal with them seperately
           
            #For each node in the first hidden layer , we have N inputs from the input layer and 
            # M outputs where M is the number of nodes in each hidden layer           
            fanIn = self.numInputNodes
            fanOut = self.numNodesPerHiddenLayer
            layerID = 1
            firstHiddenLayer =  [NeuralNode('Hidden',nodeID,layerID,fanIn,fanOut) for nodeID in range(fanOut)]           
            self.hiddenlayers.append(firstHiddenLayer)
           
            #For hidden layers [1,N-1] we can go ahead and create them in a loop
            fanIn = self.numNodesPerHiddenLayer
            fanOut = fanIn
            for layerID in range(2,self.numHiddenLayers-1):
                print(layerID)
                self.hiddenlayers.append([NeuralNode('Hidden',nodeID,layerID,fanIn,fanOut) for nodeID in range(fanOut)])
        
            #For each node in the last  hidden layer , we have N inputs from the previous layer and 
            # M outputs where M is the number of nodes in the output layer           
            if self.numHiddenLayers > 1:
                fanIn = self.numNodesPerHiddenLayer
                fanOut = self.numOutputNodes
                layerID =  self.numHiddenLayers-1
                lastHiddenLayer =  [NeuralNode('Hidden',nodeID,layerID,fanIn,fanOut) for nodeID in range(fanIn)]           
                self.hiddenlayers.append(lastHiddenLayer)
        else:
            self.hiddenlayers = None

        

    def createOutputLayer(self):
        if self.numHiddenLayers == 0:
            fanIn = self.numInputNodes
            fanOut = 1
            self.outputLayer = [NeuralNode('Output',idx,self.numLayers-1,fanIn,fanOut) for idx in range(self.numOutputNodes)]
        else:
            fanIn = self.numNodesPerHiddenLayer
            fanOut = 1
            self.outputLayer = [NeuralNode('Output',idx,self.numLayers-1,fanIn,fanOut) for idx in range(self.numOutputNodes)]
    
    def printNet(self , outFileName):
        file = open(outFileName , "w")
        header = "graph TB \n"
        #Print Input Nodes
        inputlayerStrings = []
        inputlayerStrings.append(header)
        
        if self.numHiddenLayers == 0:
            for eachNode in self.inputLayer:
                for eachOutNode in self.outputLayer: 
                    inputlayerStrings.append("     " + eachNode.nodeType + str(eachNode.nodeID) + " --> " + eachOutNode.nodeType + str(eachOutNode.nodeID) + "\n" )
        else:
            for eachNode in self.inputLayer:
                for eachHiddenNode in self.hiddenlayers[0]: 
                    inputlayerStrings.append("     " + eachNode.nodeType + "-L" + str(eachNode.layerID)+ "-N" +str(eachNode.nodeID) + " --> " + eachHiddenNode.nodeType + "-L" + str(eachHiddenNode.layerID)+ "-N" + str(eachHiddenNode.nodeID) + "\n" )
            
            for idx in range(0,self.numHiddenLayers-1):
                for eachNodeL in self.hiddenlayers[idx]:
                    for eachNodeR in self.hiddenlayers[idx+1]:
                        inputlayerStrings.append("     " + eachNodeL.nodeType + "-L" + str(eachNodeL.layerID)+ "-N" + str(eachNodeL.nodeID) + " --> " + eachNodeR.nodeType +  "-L" + str(eachNodeR.layerID)+"-N" +  str(eachNodeR.nodeID) + "\n" )

            for eachNode in self.hiddenlayers[self.numHiddenLayers-1]:
                for eachOutNode in self.outputLayer: 
                        inputlayerStrings.append("     " + eachNode.nodeType + "-L" + str(eachNode.layerID)+ "-N" + str(eachNode.nodeID) + " --> " + eachOutNode.nodeType +  "-L" + str(eachOutNode.layerID)+"-N" +  str(eachOutNode.nodeID) + "\n" )

        for eachString in inputlayerStrings:
            file.write(eachString)       
        file.close()     
    def forwardPropogate(self , inputData):
        return None
class NeuralNode:
    def __init__(self,nodeType,nodeID,layerID,fanIn,fanOut):
        self.nodeType = nodeType
        self.nodeID = nodeID
        self.layerID = layerID
        self.fanIn = fanIn
        self.fanOut = fanOut
        if self.fanIn == None :
            self.weights = None
        else:    
            self.weights = np.random.randn(self.fanIn)
    
    def printNode(self):
        print("Node Type: ", self.nodeType)    
        print("Node ID: ", self.nodeID)  
        print("Layer ID: ", self.layerID)
        print("fanIn: ", self.fanIn)
        print("fanOut: ", self.fanOut)
        print("weights: ", self.weights)  
    
    def computeResponse(self,inputData):
      if self.nodeType == "Input":
          self.response = inputData
      else:   
        self.response = np.dot(inputData,self.weights)
        self.response = logistic.cdf(self.response)
   
                  

        


# numInputNodes = 784
# numHiddenLayers = 2
# numNodesPerHiddenLayer = 512
# numOutputNodes = 10       
    
    
# ANN = NeuralNet(numInputNodes,numHiddenLayers,numNodesPerHiddenLayer,numOutputNodes)

# ANN.createInputLayer()
# ANN.createOutputLayer()
# ANN.createHiddenLayers()
# ANN.printNet("ANN.mmd")
# print("Input Layer \n")
# for node in ANN.inputLayer:
#     node.printNode()
#     print("\n")

# print("Hidden Layers \n")
# for layers in ANN.hiddenlayers:
#     for node in layers:
#         node.printNode()   
#         print("\n")
# print("Output Layer \n")
# for node in ANN.outputLayer:
#     node.printNode()   
#     print("\n")

    
