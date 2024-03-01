import random
import math

from Training.RMSPropOptimizer import *

class SigmoidNode():
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

    #Function for that Sigmoid Node
    def SigmoidFunction(self,z):
        return 1 / (1 + math.exp(-z))

    def Forward(self,Input,STM):
        #So input is batches what do we do?
        #1) matmul this is no external lib so thats a bit hard
        #2) loops easy to implement but does sacrifice performance

        #Bcs its in batches it would be fed into the AI as like this [1,23,4,3434,343....] so its just a for loop
        #print(f"Input {Input}")
        Sum = (STM * self.STW) + (Input * self.IW)

        #To get our X coord input for the TanH function we just add the bias term (Sum += Bias)
        X_Input = Sum + self.Bias

        #Get our Ouput from the TanH (Tanh turns any input into a num between -1,1) for the potential long term memory
        self.Output = self.SigmoidFunction(X_Input)

        return self.Output

    def Backward(self, AvgGradient, STM, Input, learning_rate):
        SigmoidD = self.Output * (1 - self.Output)
        input_gradient = AvgGradient * SigmoidD

        #Calculate DW for each param(STW,IW,Bias)
        STWdB = input_gradient * STM 
        IWdB = input_gradient * Input
        BIASdW = input_gradient

        #Calculate Sdw for each param(CalculateSDW is in RMSProp file)
        self.STW_Weighted_Avg = CalculateSDW(self.STW_Weighted_Avg,STWdB)#Same thing as before just set them to be equal saves processing power
        self.IW_Weighted_Avg = CalculateSDW(self.IW_Weighted_Avg,IWdB)
        self.Bias_Weighted_Avg = CalculateSDW(self.Bias_Weighted_Avg,BIASdW)

        #Update params after all that so right now the optimizer is Batch + RMSProp. Cool!
        self.STW -= RMSProp(learning_rate, self.STW, STWdB, self.STW_Weighted_Avg, self.Epsilon)
        self.IW -= RMSProp(learning_rate, self.IW, IWdB, self.IW_Weighted_Avg, self.Epsilon)
        self.Bias -= RMSProp(learning_rate, self.Bias, BIASdW, self.Bias_Weighted_Avg, self.Epsilon)
