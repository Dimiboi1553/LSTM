import random

def PreprocessData(data, BatchSize):
    # Step 1: Shuffle data 
    #random.shuffle(data)

    # Step 2: Split into batches
    # Calculate the number of full batches possible
    FullBatchesPossible = len(data) // BatchSize
    
    ProcessedBatches = []

    # Create and append full batches
    for i in range(FullBatchesPossible):
        Batch = data[i*BatchSize : (i+1)*BatchSize]
        ProcessedBatches.append(Batch)

    # Handle any remaining data that doesn't fit into a full batch
    RemainingElementsStartIndex = FullBatchesPossible * BatchSize
    if RemainingElementsStartIndex < len(data):
        RemainingData = data[RemainingElementsStartIndex:]
        ProcessedBatches.append(RemainingData)

    return ProcessedBatches

def GetMean(Input: list):
    Total = 0
    
    for i in Input:
        Total += i
    
    return Total/len(Input)