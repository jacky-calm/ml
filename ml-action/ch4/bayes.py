import numpy as np

def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]  # 1 is abusive, 0 not
    return postingList, np.array(classVec)

def createVocabList(dataSet):
    vocabSet = set([])  #create empty set
    for document in dataSet:
        vocabSet = vocabSet | set(document) #union of the two sets
    return sorted(list(vocabSet))


def setOfWords2Vec(vocabList, inputSet):
    return [inputSet.count(v) for v in vocabList]

def trainNB0(trainMatrix, trainCategory):
    pOw = trainMatrix.sum(axis=0, dtype=float)/trainMatrix.sum()
    pOc1 = trainCategory.sum(dtype=float)/trainCategory.size
    pOc0 = 1 - pOc1

    xOc1 = trainMatrix[trainCategory.astype(bool)]
    xOc0 = trainMatrix[np.logical_not(trainCategory)]
    pOwGc1 = (xOc1.sum(axis=0, dtype=float) + 1) / (xOc1.sum() + 2)
    pOwGc0 = (xOc0.sum(axis=0, dtype=float) + 1) / (xOc0.sum() + 2)

    return np.log(pOw), pOc1, pOc0, np.log(pOwGc1), np.log(pOwGc0)

def classifyNB(vec2Classify, pOw, pOc1, pOc0, pOwGc1, pOwGc0):
    # pOw is not needed as it is the same for for all classes
    pOc1OV = sum(vec2Classify * pOwGc1) + np.log(pOc1)
    pOc0OV = sum(vec2Classify * pOwGc0) + np.log(pOc0)
    # print pOc1OV, pOc0OV
    return pOc1OV > pOc0OV

postingList, classVec = loadDataSet()
vocabList = createVocabList(postingList)
print vocabList
trainX = np.array([setOfWords2Vec(vocabList, posting) for posting in postingList])
print trainX
pOw, pOc1, pOc0, pOwGc1, pOwGc0 = trainNB0(trainX, classVec)
print pOw
print pOc1, pOc0
print pOwGc1
print pOwGc0

testEntry = ['love', 'my', 'dalmation']
thisDoc = np.array(setOfWords2Vec(vocabList, testEntry))
print testEntry, 'classified as: ', classifyNB(thisDoc, pOw, pOc1, pOc0, pOwGc1, pOwGc0)
testEntry = ['stupid', 'garbage']
thisDoc = np.array(setOfWords2Vec(vocabList, testEntry))
print testEntry,'classified as: ', classifyNB(thisDoc, pOw, pOc1, pOc0, pOwGc1, pOwGc0)