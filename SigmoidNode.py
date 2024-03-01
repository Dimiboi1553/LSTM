import random
import math

class SigmoidNode():
    def __init__(self):
        #Init weights and bias for node
        self.STW = random.uniform(-1, 1) #STW = Short Term Weight
        self.IW = random.uniform(-1, 1) #IW = Input Weight

        self.Bias = random.uniform(-1, 1) #Bias = bias

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

        self.STW -= learning_rate * input_gradient * STM
        self.IW -= learning_rate * input_gradient * Input
        self.Bias -= learning_rate * input_gradient