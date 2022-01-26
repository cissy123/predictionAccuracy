import numpy as np
from collections import defaultdict
import random
import xlrd


def readExcelData(fileName, specialAlleleList,SHUFFLE=True):
    exampleList = defaultdict(list)
    alleleList = list()
    alleleData = list()
    alleleDataIndex = list()
    alleleDataNullIndex = list()
    newAlleleData = list()
    specialIndex = list()
    shuffleGroupName = list()

    ###open###
    wb = xlrd.open_workbook(fileName)
    sh = wb.sheet_by_index(0)
    table = wb.sheets()[0]
    # readData
    nrows = table.nrows
    ncols = table.ncols
    for i in range(1, nrows):
        if '' not in table.row_values(i)[3:] and ' ' not in table.row_values(i)[3:]:
            alleleData.append(table.row_values(i)[3:])
            alleleDataIndex.append(i)
        else:
            alleleDataNullIndex.append(i)

    # exampleList['exampleName'] = readTableElement(table.col_values(0), alleleDataIndex)
    # exampleList['populationName'] = readTableElement(table.col_values(1), alleleDataIndex)
    # exampleList['groupName'] = readTableElement(table.col_values(2), alleleDataIndex)
    groupName = readTableElement(table.col_values(2), alleleDataIndex)
    alleleList = table.row_values(0)[3:]

    for i in specialAlleleList:
        if i in alleleList:
            ii = alleleList.index(i)
            specialIndex.append(ii)

    for i in range(len(groupName)):
        for j in range(len(alleleList)):
            # if alleleData[i][j] is None:
            #     alleleData[i][j] = -999
            #     newAlleleData.append(alleleData[i][j])
            if alleleData[i][j] in ('na', ' na', ' na '):
                # print(alleleData[i][j])
                # print(i)
                alleleData[i][j] = -999
                newAlleleData.append(alleleData[i][j])
            elif type(alleleData[i][j]) == str:
                # print(alleleData[i][j])
                s = alleleData[i][j].split(',')
                # newAlleleData.append(float(s[0]))

                if j in specialIndex:
                    if len(s) > 2:
                        alleleData[i][j] = -999
                        newAlleleData.append(-999)
                        newAlleleData.append(-999)
                        continue
                    newAlleleData.append(float(s[0]))
                    if s[-1] == '':
                        newAlleleData.append(float(s[0]))
                    else:
                        newAlleleData.append(float(s[-1]))
                else:
                    newAlleleData.append(float(s[0]))
            elif j in specialIndex:
                newAlleleData.append(alleleData[i][j])
                newAlleleData.append(alleleData[i][j])
            else:
                newAlleleData.append(alleleData[i][j])
    if len(specialAlleleList) > 0:
        alleleList = updateList(alleleList, specialAlleleList)

    alleleNum = len(alleleList)
    data = np.array(newAlleleData).reshape(int(len(groupName)), alleleNum)
    #####################################
    # return groupName,alleleList,data

    #shuffle
    if SHUFFLE:
        cc = list(zip(data, [i for i in range(len(groupName))]))
        random.Random(len(groupName)).shuffle(cc)
        shuffleData,shuffleIndex = zip(*cc)

        for i in shuffleIndex:
            shuffleGroupName.append(groupName[i])

        return shuffleGroupName, alleleList, shuffleData,data,groupName,shuffleIndex
    else:
        return groupName,alleleList,data

def updateList(oldList, name):
    indexList = list()
    for i in name:
        if i in oldList:
            indexList.append(oldList.index(i))

    counter = 0
    for x in indexList:
        oldList.insert(x + counter, name[counter])
        counter += 1
    return oldList


def readTableElement(table, index):
    element = list()
    for i in index:
        # element.append(str(table[i]).replace(' ', '-'))
        element.append(str(table[i]).strip())
    return element
