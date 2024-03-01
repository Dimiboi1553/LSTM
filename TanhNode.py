import math
import random

class TanhNode():
    def __init__(self):
        #Init weights and bias for node
        self.STW = random.uniform(-1, 1) #STW = Short Term Weight
        self.IW = random.uniform(-1, 1) #IW = Input Weight

        self.Bias = random.uniform(-1, 1) #Bias = bias

        self.Ouput = 0.0
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

        self.STW -= learning_rate *  input_gradient * STM
        self.IW -= learning_rate * input_gradient * Input
        self.Bias -= learning_rate * input_gradient