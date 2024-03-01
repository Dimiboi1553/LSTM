def MSE(actual,predicted):
    return ((actual - predicted) ** 2).mean()

def MSE_Gradient(actual,predicted):
    return (predicted - actual)