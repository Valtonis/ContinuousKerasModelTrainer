
# coding: utf-8

# In[ ]:

import copy
from random import shuffle

class GeneratorForKerasNetwork(Sequence):
    def __init__(self, indexes, dataPuller, batchSize, temporaryStorageSize=1):
        self.__dataPuller = dataPuller
        self.__indexes = indexes
        if batchSize > len(indexes):
            batchSize = len(indexes)
        self.__batchSize = batchSize
        
        modulo = len(self.__indexes) % self.__batchSize
        if modulo != 0:
            self.__len = math.ceil(len(self.__indexes) / self.__batchSize)
        else:
            self.__len = math.floor(len(self.__indexes) / self.__batchSize)
        if temporaryStorageSize > self.__len:
            temporaryStorageSize = self.__len
        self.__temporaryStorageSize = temporaryStorageSize
        
    def __len__(self):
        return self.__len
    
    def __next__(self):
        
        def getLocalIndexCounter(maxIndex):
            if getLocalIndexCounter.hasattr("idx"):
                getLocalIndexCounter.idx += 1
                getLocalIndexCounter.idx = getLocalIndexCounter.idx % maxIndex
            else:
                getLocalIndexCounter.idx = 0
            return getLocalIndexCounter.idx
        
        idx = getLocalIndexCounter(self.__len__())
        data = self.__getitem__(idx)
        return data
    
    def __getitem__(self, idx):
        if self.__loadedBatches is None or idx not in self.__loadedBatches:
            self.__firstLoadedBatchIndex = idx
            batchesToLoad = []
            for shift in range(self.__temporaryStorageSize):
                batchesToLoad.append((idx + shift) % self.__len)
            self._loadRecordsInTemporaryStorage(batchesToLoad)
        return self._getPreloadedBatch(idx - self.__firstLoadedBatchIndex)
    
    def _calculateBatchRecordsDistribution(self):
        self.__batchRecordIndexes = [None] * __len
        idx = 0
        while idx < __len:
            beginIndexInSequence = idx * self.__batchSize
            if idx < self.__len__() - 1:
                self.__batchRecordIndexes[idx] = self.__indexes[beginIndexInSequence : beginIndexInSequence + self.__batchSize]
            else:
                self.__batchRecordIndexes[idx] = self.__indexes[beginIndexInSequence : len(self.__indexes)]
            idx += 1
        
    def _loadRecordsInTemporaryStorage(self, batchIndexes):
        recordIndexes = []
        for batchNum in batchIndexes:
            recordIndexes += self.__batchRecordIndexes[batchNum]
        self.__loadedBatches = batchIndexes
        batchesData = self.__dataPuller.getData(recordIndexes)
        # shuffle data to make learning process more chaotic
        z = list(zip(batchesData[0], batchesData[1]))
        shuffle(z)
        self.__temporaryStorage = zip(*z)
        
    def _getPreloadedBatch(self, num):
        beginIndexInSequence = num * self.__batchSize
        X = None
        y = None
        if idx < len(self.__loadedBatches) - 1:
            X = self.__temporaryStorage[0][beginIndexInSequence : beginIndexInSequence + self.__batchSize]
            y = self.__temporaryStorage[1][beginIndexInSequence : beginIndexInSequence + self.__batchSize]
        else:
            X = self.__temporaryStorage[0][beginIndexInSequence : len(self.__temporaryStorage[0])]
            y = self.__temporaryStorage[1][beginIndexInSequence : len(self.__temporaryStorage[0])]
        return (X, y)
    
    # Fields:
    
    __dataPuller = None
    __indexes = None
    __batchSize = None
    __len = None
    
    __batchRecordIndexes = None
    __temporaryStorage = None
    __temporaryStorageSize = None
    __loadedBatches = None
                                               
    __firstLoadedBatchIndex = None


# In[ ]:



