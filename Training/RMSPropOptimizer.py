import math

def RMSProp(learning_rate, ValueToChange, dW, Sdw, Epsilon=1e-8):
    #Equation is this For RMSProp(Root Mean Squared Propagation):
    # W = W - learning rate * dW(or dB)/sqrt(Sdb + Epsilon)
    return ValueToChange - learning_rate * dW/math.sqrt(Sdw + Epsilon)
   

def CalculateSDW(SDWprev,dW,B=0.999):
    #Equation is: Sdw = B * Sd(w or b)prev + (1-B) * (dW or dB)**2
    return B * SDWprev + (1-B) * dW**2