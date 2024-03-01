import math

def MSE(actual,predicted):
    return ((actual - predicted) ** 2).mean()

def MSE_Gradient(actual,predicted):
    return (predicted - actual)

class RNN():
    def __init__(self):
        self.W_1 = 1.8
        self.W_2 = -0.5
        self.W_3 = 1.1

        self.B_1 = 0.0
        self.B_2 = 0.0

    def TanH(self,Input):
        return math.tanh(Input)
    
    def Predict(self, Inputs: list):
        x_axis = Inputs[0] * self.W_1 + self.B_1
        Length = len(Inputs)-1
    
        for i in range(1,Length):
            y_axis = self.TanH(x_axis)

            if i < Length:
                x_axis = (Inputs[i+1] * self.W_1) + (y_axis * self.W_2)

        y_axis *= self.W_3
        FinalPrediction = y_axis + self.B_2

        return FinalPrediction
    
    def Learn(self,num_epochs, LearningRate, Training_Data):
        TotalLoss = 0.0
        for Epochs in range(num_epochs):
            
            for Inputs,Target in Training_Data:
                Prediction = self.Predict(Inputs)

                loss = MSE(Target,Prediction)
                TotalLoss += loss

                DLoss = MSE_Gradient(Target,Prediction)

                self.Backpropagate(Inputs,DLoss,LearningRate)

            print(f"Epoch {Epochs+1}, Average Loss: {TotalLoss / len(Training_Data)}")

    def Backpropagate(self, Inputs: list, DLoss:float, LearningRate: float):
        dW_1 = 0.0
        dW_2 = 0.0
        dW_3 = 0.0
        dB_1 = 0.0
        dB_2 = 0.0

        dx_axis = 0.0
        dy_axis = 0.0

        # Backpropagation through time
        for i in reversed(range(len(Inputs))):
            x_i = Inputs[i]
            y_i = self.TanH(x_i * self.W_1 + self.B_1)

            # Gradient of output with respect to W_3 and B_2
            dW_3 = DLoss * y_i
            dB_2 = DLoss

            # Update derivatives
            dy_axis = DLoss * self.W_3
            dx_axis = dy_axis * (1 - y_i**2) * self.W_1

            # Gradient of output with respect to W_2
            dW_2 += dy_axis * y_i
            # Gradient of output with respect to W_1
            dW_1 += dx_axis * x_i
            # Gradient of output with respect to B_1
            dB_1 += dx_axis

        # Update parameters
        self.W_1 -= LearningRate * dW_1
        self.W_2 -= LearningRate * dW_2
        self.W_3 -= LearningRate * dW_3
        self.B_1 -= LearningRate * dB_1
        self.B_2 -= LearningRate * dB_2



