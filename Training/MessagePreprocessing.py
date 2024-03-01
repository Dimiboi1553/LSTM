import random

def SentencesToNumbers(Sentences):
  Wordsdict = {}
  for Sentence in Sentences:
    Words = Sentence.split(" ")
    for i in Words:
      if Wordsdict.get(i, 0) == 0:
        Wordsdict[i] = round(random.uniform(-1, 1), 2)

  return Wordsdict


def SentencesToNumRep(Sentences, dict):
  Finals = []

  for Sentence in Sentences:
    Converted = []
    words = Sentence.split(" ")
    for i in words:
      Converted.append(dict.get(i, 0))

    Finals.append(Converted)
  return Finals

def findNearestInDict(trainSet, output):
    reversed_dict = {v: k for k, v in trainSet.items()}
    
    # Find the key with the closest value to output
    nearest_value = min(reversed_dict.keys(), key=lambda x: abs(x - output))
    return reversed_dict[nearest_value]