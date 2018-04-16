
# coding: utf-8

# In[ ]:

import copy
import mutex

from keras.utils import Sequence

# blockSize - maximum number of records that could be loaded into video ram
# blockSize * temporaryStorageSize - maximum number of records that could be stored in ram

class GeneratorForKerasNetworkFinite(Sequence):
    def __init__(self, indexes, dataPuller, blockSize, temporaryStorageSize=1):
        self.__dataPuller = dataPuller
        self.__indexes = indexes
        if blockSize > len(indexes):
            blockSize = len(indexes)
        self.__blockSize = blockSize
        self.__indexesBlockBorders = self._getBlockBorders(self.__blockSize, len(self.__indexes))
        self.__len = len(self.__indexesBlockBorders)
        if temporaryStorageSize > self.__len:
            temporaryStorageSize = self.__len
        self.__temporaryStorageSize = max(temporaryStorageSize, 1)
        
        self.__getItemMutex = mutex.mutex()
        
    def __len__(self):
        return self.__len
    
    def __getitem__(self, idx):
        self.__getItemMutex.lock()
        if self._isBlockLoaded(idx) == False:
            self._loadFollowingBlocks(idx)
        res = self._getLoadedBlock(idx - self.__firstLoadedBlock)
        self.__getItemMutex.unlock()
        return res
    
    def _isBlockLoaded(self, idx):
        if idx > self.__lastLoadedBlock and idx < self.__firstLoadedBlock:
            return False
        elif idx > self.__lastLoadedBlock:
            return False
        elif idx < self.__firstLoadedBlock:
            return False
        return True
    
    def _loadFollowingBlocks(self, firstBlock):
        self.__firstLoadedBlock = firstBlock % self.__len
        i = 0
        recordIndexesToLoad = []
        while i < temporaryStorageSize:
            blockNum = (self.__firstLoadedBlock + i) % self.__len
            borders = self.__indexesBlockBorders[blockNum]
            recordIndexesToLoad += self.indexes[borders[0] : borders[1] + 1]
            i += 1
        self.__lastLoadedBlock = (self.__firstLoadedBlock + i - 1) % self.__len
        self.__temporaryStorage = self.__dataPuller.getData(recordIndexesToLoad)
        self.__loadedDataBlockBorders = self._getBlockBorders(self.__blockSize, len(self.__temporaryStorage))
        
    def _getLoadedBlock(self, shift):
        border = self.__loadedDataBlockBorders[shift]
        X = self.__temporaryStorage[0][border[0] : border[1] + 1]
        y = self.__temporaryStorage[1][border[0] : border[1] + 1]
        return (X, y)
    
    def _getBlockBorders(self, blockSize, numberOfRecords):
        borders = []
        numberOfBlocks = None
        modulo = numberOfRecords % blockSize
        if modulo != 0:
            numberOfBlocks = math.ceil(numberOfRecords / blockSize)
        else:
            numberOfBlocks = math.floor(numberOfRecords / blockSize)
        i = 0
        while i < numberOfBlocks - 1:
            firstIndex = i * blockSize
            borders.append((firstIndex, firstIndex + blockSize - 1))
            i += 1
        return borders
        
    # Fields:
    
    __dataPuller = None
    __indexes = None
    __blockSize = None
    __len = None
    
    __firstLoadedBlock = None
    __lastLoadedBlock = None
    
    __indexesBlockBorders = None
    __loadedDataBlockBorders = None
    
    __temporaryStorage = None
    __temporaryStorageSize = None
    
    __getItemMutex = None



