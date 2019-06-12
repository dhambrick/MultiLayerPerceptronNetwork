import numpy as np 
import scipy as sp 
import pandas as pd 
from scipy.stats import  logistic



def ReLU(x):
    return np.maximum(0,x)

def dReLUdx(x):
    temp = []
    for x_val in x:
        if x_val < 0:
            temp.append(0)
        else:
            temp.append(1)
    return temp


def Sigmoid(x):
    return logistic.cdf(x)

def DSigmoidDx(x):
    return Sigmoid(x)*(1-Sigmoid(x))

def ErrorSquared(pred_x,true_x):
    return np.sum((pred_x-true_x)**2)

activations =  {"ReLU" :{
                    "f":ReLU,
                    "DfDx":dReLUdx},
                "Sigmoid" :{
                    "f":Sigmoid,
                    "DfDx":Sigmoid}}
training_costs = {"Error_Squared":ErrorSquared}



    