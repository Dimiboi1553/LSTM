import random

from MSE import MSE_Gradient, MSE
from SigmoidNode import SigmoidNode
from TanhNode import TanhNode
from Training.MiniBatchTraining import PreprocessData,GetMean
from Training.MessagePreprocessing import *

class LSTM():
    def __init__(self):
        self.learning_rate = 1

        self.LongTermMemory = random.randint(-1, 4)
        self.ShortTermMemory = random.randint(-1, 4)

        #Each LSTM module has 3 Sigmoid Modules and 1 TanH Module
        self.SigmoidNode = SigmoidNode()
        self.SigmoidNode2 = SigmoidNode()
        self.SigmoidNode3 = SigmoidNode()

        self.TanH = TanhNode()

    def Forward(self,Input):
        #Input is now in batches 
        #Step 1 Forget Gate: (% Long Term to Remember)
        self.LongTermMemory *= self.SigmoidNode.Forward(Input,self.ShortTermMemory)

        #Step 2 Input Gate: 
        #a) % of Potential Memory to Remember 
        SigmoidOutput = self.SigmoidNode2.Forward(Input,self.ShortTermMemory)

        #b) Potential Long Term Memory
        TanhOuput = self.TanH.Forward(Input,self.ShortTermMemory)

        #c) Multiply TanhOutput with SigmoidOutput and then add it to the Long term Memory
        self.LongTermMemory += SigmoidOutput * TanhOuput

        #Step 3: Update Short Term Memory
        self.ShortTermMemory = self.OutputGate(Input)
        
        return self.ShortTermMemory
    
    def OutputGate(self,Input):
        
        #Get Output from Tanh Function with input Long term memory
        TanHOutput = self.TanH.TanH(self.LongTermMemory)

        #Final Output by the final Sigmoid Node to use as input to the Output node
        FinalInput = self.SigmoidNode3.Forward(Input,self.ShortTermMemory)

        return TanHOutput * FinalInput
    
    def Learn(self, Input: list,Epochs: int,BatchSize: int):
        #Processes that data into batches
        Input = PreprocessData(Input,BatchSize)

        for i,x in enumerate(range(Epochs)):

            #print(f"Batch: {Input}")

            for Batches in Input:

                Gradients = []

                for i in range(len(Batches) - 1):
                    #First Forward pass
                    Forward = self.Forward(Batches[i])

                    #Calculate MSE gradient and then add it to the Gradients
                    mse = MSE_Gradient(Batches[i+1],Forward)
                    
                    Gradients.append(mse)

                if len(Batches) > 1:
                    AvgError = GetMean(Gradients)
                else:
                    AvgError = mse

                self.Backwards(AvgError,Batches)
            
            # if x % 20 == 0:
            #     print(f"Epoch: {x} Avg Error: {AvgError}")
            
    def Backwards(self, Gradient, Input):
        for i in reversed(range(len(Input))):
            x = Input[i]
            
            self.UpdateWeights(Gradient,x)
            
    def UpdateWeights(self,Gradient,x):
        #Update Sigmoid nodes
        self.SigmoidNode.Backward(Gradient, self.ShortTermMemory, x, self.learning_rate)
        self.SigmoidNode2.Backward(Gradient, self.ShortTermMemory, x, self.learning_rate)
        self.SigmoidNode3.Backward(Gradient, self.ShortTermMemory, x, self.learning_rate)

        #Update TanH node
        self.TanH.Backward(Gradient, self.ShortTermMemory, x, self.learning_rate)

    def LearnWithBatchedData(self,Data,Epochs,BatchSize):
        for i in Data:
            self.Learn(i,Epochs,BatchSize)
            

lstm = LSTM()

# Define a sequence of inputs
TrainingData = [0.1, 0.2, 0.3, 0.4, 0.5]

# Feed the sequence through the LSTM
lstm.Learn(TrainingData,100,2)

TestData = [0.1,0.2,0.3]

for i in TestData:
    output = lstm.Forward(i)
    print(f"Input: {i}, Output: {output}")

