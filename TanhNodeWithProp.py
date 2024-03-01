import math
import random

from Training.RMSPropOptimizer import *

class TanhNode():
    def __init__(self,Beta=0.9, Epsilon=1e-8):
        #-------------Forward Propagation Params----------------#
        #Init weights and bias for node
        self.STW = random.uniform(-1, 1) #STW = Short Term Weight
        self.IW = random.uniform(-1, 1) #IW = Input Weight
        self.Bias = random.uniform(-1, 1) #Bias = bias
        
        #-------------Backpropagation Params--------------------#
        #For RMSProp we have to keep track of moving averages of each param
        #These are all exponential weighted avgs
        self.STW_Weighted_Avg = 0
        self.IW_Weighted_Avg = 0
        self.Bias_Weighted_Avg = 0

        self.beta = Beta
        self.Epsilon = Epsilon

        self.Output = 0.0

    #Function for that Tan H Node
    def TanH(self,Input):
        return math.tanh(Input)

    def Forward(self,Input,STM):#STM = Short Term Memory
        #Basically the Same thing as Sigmoid Node
        #(STM * STM Weight) + (Input * Input Weight)

        Sum = (STM * self.STW) + (Input * self.IW)

        #To get our X coord input for the TanH function we just add the bias term (Sum += Bias)
        X_Input = Sum + self.Bias

        #Get our Ouput from the TanH (Tanh turns any input into a num between -1,1) for the potential long term memory
        self.Output = self.TanH(X_Input)

        return self.Output
    
    def Backward(self, AvgGradient, STM, Input, learning_rate):

        TanH_D = 1 - math.tanh(self.Output)**2
        input_gradient = AvgGradient * TanH_D

        # Calculate dW for each parameter (STW, IW, Bias)
        STW_dB = input_gradient * STM
        IW_dB = input_gradient * Input
        Bias_dW = input_gradient

        # Calculate SdW for each parameter
        self.STW_Weighted_Avg = CalculateSDW(self.STW_Weighted_Avg, STW_dB)
        self.IW_Weighted_Avg = CalculateSDW(self.IW_Weighted_Avg, IW_dB)
        self.Bias_Weighted_Avg = CalculateSDW(self.Bias_Weighted_Avg, Bias_dW)

        # Update parameters using RMSProp
        self.STW -= RMSProp(learning_rate, self.STW, STW_dB, self.STW_Weighted_Avg, self.Epsilon)
        self.IW -= RMSProp(learning_rate, self.IW, IW_dB, self.IW_Weighted_Avg, self.Epsilon)
        self.Bias -= RMSProp(learning_rate, self.Bias, Bias_dW, self.Bias_Weighted_Avg, self.Epsilon)