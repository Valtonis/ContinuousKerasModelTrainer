
# coding: utf-8

# descendants of this class perform a function of data delivery
# regardless of place where it is stored (in some database or remotely)
class DataPuller:
    # this function should return tuple (X, y)
    # containing training records X (with indexes in indexList) and corresponding labels y
    def getData(self, indexList):
        pass
    # this function should return amount of all available records
    def getNumberOfRecords(self):
        pass

