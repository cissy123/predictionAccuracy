import os
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn import svm, metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
import read_data_1213
import xlwt
import os


def read_data(dataset, file, specialAlleleList):
    file = os.path.join(dataset, file)
    print(file)
    data = read_data_1213.readExcelData(file, specialAlleleList)
    return data


def predict(shuffle_data, alg, k_fold_num, score_result, file_num):
    alleleGroupName, alleleList, alleleData, data, groupName, shuffleIndex = shuffle_data
    compare_result = [[] for i in range(len(alg))]
    original_compare_result = [[] for i in range(len(data))]

    # k fold loop
    loop_num = 1
    kf = KFold(n_splits=k_fold_num)
    for train, test in kf.split(alleleGroupName):
        x_train = [alleleData[i] for i in train]
        x_test = [alleleData[i] for i in test]
        y_train = [alleleGroupName[i] for i in train]
        y_test = [alleleGroupName[i] for i in test]

        # train label
        unique_group = np.unique(y_train)
        in_uniqueGroup = len([i for i in range(len(y_test)) if y_test[i] in unique_group]) / len(
            y_test)  # in case of y_test not in y_train

        for alg_idx, alg_type in enumerate(alg):
            if alg_type == 'knn':
                classifer = KNeighborsClassifier(1)
            elif alg_type == 'naiveBayes':
                classifer = GaussianNB()
            elif alg_type == 'logisticRegression':
                classifer = LogisticRegression(penalty="l2", C=1, multi_class="ovr", solver="newton-cg")
            elif alg_type == 'svm':
                classifer = svm.SVC()
            elif alg_type == 'decisionTree':
                classifer = DecisionTreeClassifier(criterion='entropy')
            elif alg_type == 'randomForest':
                classifer = RandomForestClassifier(n_estimators=170)

            classifer.fit(x_train, y_train)
            pred_result = classifer.predict(x_test)
            compare_result[alg_idx] = compare_result[alg_idx] + \
                                      [i for i in map(lambda x, y: x if x != y else '',
                                                      pred_result, y_test)]
            score_result[file_num][alg_idx] += metrics.accuracy_score(y_test, pred_result) / in_uniqueGroup
            loop_num += 1
            print('algorithm {} score={}'.format(alg_type, metrics.accuracy_score(y_test,
                                                                                                         pred_result) / in_uniqueGroup))

    # return original index
    compare_result_t = list(map(list, zip(*compare_result)))
    for i in range(len(data)):
        idx = shuffleIndex.index(i)
        original_compare_result[i] = compare_result_t[idx]
    original_compare_result_tmp = list(map(list, zip(*original_compare_result)))
    return data, groupName, original_compare_result_tmp


def save_pred_result(result, file_name, data, groupName, alg, compare_result):
    alg_num = len(alg)
    # write
    sheet_result = result.add_sheet(file_name, cell_overwrite_ok=True)
    sheet_result.write(0, 0, u'group')
    for alg_idx, alg_type in enumerate(alg):
        sheet_result.write(0, alg_idx + 1, alg_type)
    # sheet1.write(0,1,u'knn')
    # sheet1.write(0,2,u'naiveBayes')
    # sheet1.write(0,3,u'logisticRegression')
    # sheet1.write(0,4,u'svm')
    # sheet1.write(0,5,u'decesion tree')
    # sheet1.write(0,6,u'random forest')
    for i in range(1, len(data) + 1):
        sheet_result.write(i, 0, groupName[i - 1])
        for j in range(1, alg_num + 1):
            sheet_result.write(i, j, compare_result[j - 1][i - 1])
        for j in range(alg_num + 1, len(data[0]) + alg_num + 1):
            sheet_result.write(i, j, data[i - 1][j - (alg_num + 1)])
    return


def save_score_result(result, dataset_dirs, score_result):
    sheet2 = result.add_sheet(u'score', cell_overwrite_ok=True)
    sheet2.write(0, 0, u'method')
    sheet2.write(0, 1, u'knn')
    sheet2.write(0, 2, u'naiveBayes')
    sheet2.write(0, 3, u'logisticRegression')
    sheet2.write(0, 4, u'svm')
    sheet2.write(0, 5, u'decesion tree')
    sheet2.write(0, 6, u'random forest')

    for i in range(1, len(dataset_dirs) + 1):
        sheet2.write(i, 0, dataset_dirs[i - 1])
    for i in range(1, len(dataset_dirs) + 1):
        for j in range(1, 7):
            sheet2.write(i, j, round(score_result[i - 1][j - 1] / 5.0, 3))


def run(dataset, specialAlleleList, alg, output, k_fold_num):
    try:
        dataset_dirs = os.listdir(dataset)
        score_result = [[0 for _ in range(len(alg))] for j in range(len(dataset_dirs))]
        result = xlwt.Workbook()

        for file_idx, file_name in enumerate(dataset_dirs):
            shuffle_data = read_data(dataset, dataset_dirs[file_idx], specialAlleleList)
            data, groupName, compare_result = predict(shuffle_data, alg, k_fold_num, score_result, file_idx)
            save_pred_result(result, file_name, data, groupName, alg, compare_result)
        save_score_result(result, dataset_dirs, score_result)
        result.save(output)
    except:
        print('compute error')
    else:
        print('compute success. {} is written'.format(output))
    return


if __name__ == '__main__':
    dataset_folder = "./dataset_test"
    pred_alg = ['knn', 'randomForest', 'decisionTree', 'naiveBayes', 'logisticRegression', 'svm']
    output_file_name = 'result_xxx.xlsx'
    specialAlleleList = ['DYS385', 'DYF387S1']
    k_fold_num = 5

    run(dataset_folder, specialAlleleList, pred_alg, output_file_name, k_fold_num)
