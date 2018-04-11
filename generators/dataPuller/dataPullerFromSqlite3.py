
# coding: utf-8

import sqlite3
import numpy as np
import io

def padSequence(sequence, necessaryNumberOfWordsInDocument):
    if len(sequence) == 0:
        return None
    wordVectorLength = None
    res = None
    dtype = sequence[0].dtype
    if len(sequence[0].shape) == 2:
        wordVectorLength = sequence[0].shape[-1]
        res = np.zeros(shape=(len(sequence), necessaryNumberOfWordsInDocument, wordVectorLength), dtype=dtype)
    else:
        wordVectorLength = 1
        res = np.zeros(shape=(len(sequence), necessaryNumberOfWordsInDocument), dtype=dtype)
    i = 0
    while i < len(sequence):
        realNumberOfWordsInDocument = sequence[i].shape[0]
        difference = necessaryNumberOfWordsInDocument - realNumberOfWordsInDocument
        if difference > 0:
            res[i] = np.insert(sequence[i], [0] * difference, 0, axis=0)
        else:
            difference *= -1
            #res[i] = np.delete(sequence[i], [-1] * difference, axis=0)
            seqLen = len(sequence[i])
            res[i] = np.delete(sequence[i], [seqLen - g - 1 for g in range(difference)], axis=0)
        i += 1
    return res

# getting data from sqlite3 databases. table in database should have three columns id, X, y

class DataPullerFromSqlite3(DataPuller):
    def __init__(self, databaseFilename, tableName, padSize):
        def adapt_array(arr):
            """
            http://stackoverflow.com/a/31312102/190597 (SoulNibbler)
            """
            out = io.BytesIO()
            np.save(out, arr)
            out.seek(0)
            return sqlite3.Binary(out.read())

        def convert_array(text):
            out = io.BytesIO(text)
            out.seek(0)
            return np.load(out)


        # Converts np.array to TEXT when inserting
        sqlite3.register_adapter(np.ndarray, adapt_array)

        # Converts TEXT to np.array when selecting
        sqlite3.register_converter("numpyArray", convert_array)
        
        self.__databaseFilename = databaseFilename
        self.__tableName = tableName
        self.__padSize = padSize
        
        with sqlite3.connect(self.__databaseFilename, detect_types=sqlite3.PARSE_DECLTYPES) as cur:
            cur.execute("SELECT COUNT(*) FROM %s", self.__tableName)
            self.__numberOfRecords = cur.fetchone()[0]
    
    def getData(self, indexList):
        with sqlite3.connect(self.__databaseFilename, detect_types=sqlite3.PARSE_DECLTYPES) as cur:
            indexString = ""
            for index in indexList:
                indexString += str(index) + ","
            indexString = indexString[:-1]
            cur.execute("SELECT X, y FROM %s WHERE id IN (%s)", self.__tableName, indexString)
            X = []
            y = []
            for rec in cur:
                X.append(rec[0])
                y.append(rec[1])
            X = padSequence(X, self.__padSize)
            return (X, y)
        return None
    
    def getNumberOfRecords(self):
        return self.__numberOfRecords
            
        
    __databaseFilename = None
    __tableName = None
    __padSize = None
    __numberOfRecords = None

