
# coding: utf-8

# In[ ]:

import copy

from keras.utils import Sequence

# blockSize - maximum number of records that could be loaded into video ram
# blockSize * temporaryStorageSize - maximum number of records that could be stored in ram

class GeneratorForKerasNetworkInfinite(Sequence):
    def __init__(self, indexes, dataPuller, blockSize, temporaryStorageSize=1):
        self.__dataPuller = dataPuller
        self.__indexes = indexes
        if blockSize > len(indexes):
            blockSize = len(indexes)
        self.__blockSize = blockSize
        self.__temporaryStorageSize = max(temporaryStorageSize, 1)
        if self.__blockSize * self.__temporaryStorageSize >= len(indexes):
            self.__loadAllInRam = True
            self.__amountOfRecordsInRam = len(indexes)
        else:
            self.__loadAllInRam = False
            self.__amountOfRecordsInRam = self.__blockSize * self.__temporaryStorageSize
        
        self.__loadNextBlocksOfDataFromIdx = 0
        self._loadNextPieceOfData()
    
    def __next__(self):
        X = None
        y = None
        if self.__loadAllInRam == True:
            indexesToSend = []
            i = 0
            while i < self.__blockSize:
                indexesToSend.append(self.__indexes[self.__currentIndex])
                self.__currentIndex += 1
                self.__currentIndex %= len(self.__indexes)
                i += 1
            X = self.__temporaryStorage[0][indexesToSend]
            y = self.__temporaryStorage[1][indexesToSend]
        elif self.__currentBlock < self.__temporaryStorageSize:
            X = self.__temporaryStorage[0][self.__currentBlock * blockSize : (self.__currentBlock + 1) * blockSize]
            y = self.__temporaryStorage[1][self.__currentBlock * blockSize : (self.__currentBlock + 1) * blockSize]
            self.__currentBlock += 1
        else:
            self._loadNextPieceOfData()
            return self.__next__()
        return (X, y)
    
    def _loadNextPieceOfData(self):
        indexesToLoad = []
        i = 0
        while i < self.__amountOfRecordsInRam:
            indexesToLoad.append(self.__indexes[self.__loadNextBlocksOfDataFromIdx])
            self.__loadNextBlocksOfDataFromIdx += 1
            self.__loadNextBlocksOfDataFromIdx %= len(self.__indexes)
            i += 1
        self.__temporaryStorage = self.__dataPuller.getData(indexesToLoad)
        self.__currentBlock = 0
    
    # Fields:
    
    __dataPuller = None
    __indexes = None
    __blockSize = None
    __amountOfRecordsInRam = None
    
    __loadNextBlocksOfDataFromIdx = None
    __currentBlock = None
    
    __loadAllInRam = None
    __currentIndex = None
    
    __temporaryStorage = None


# In[ ]:



