import codecs
import sys
import numpy as np
import os


feature_num = 136
label_num = 5


def train(train_path, fold_num):
    train_fd = codecs.open(train_path, "r", encoding="UTF-8")
    # real_rank = 0
    predict_rank = 0
    weights = []  # weight
    thresholds = []  # bias
    for i in range(0, feature_num):
        weights.append(float(0))  # init to 0
    for i in range(0, label_num - 1):
        thresholds.append(float(0))  # init to 0
    thresholds.append(sys.float_info.max)
    # print(w)
    # print(b)
    line = train_fd.readline()
    while line:
        line = line.rstrip("\r\n")
        lines = line.split(" ")
        real_rank = int(lines[0])  # 相关等级 实际值
        features = []
        for i in range(0, feature_num):
            feature = [float(lines[i + 2].split(":")[1])]  # 对应的特征值
            features.append(feature)
        #  print(features)
        features_mat = np.array(features)  # transfer to np array
        #    print(features_mat)
        weight_mat = np.array(weights)
        dot = np.dot(weight_mat, features_mat)  # 得到点乘实值
        # print(dot)
        # predict_rank = label_num
        for j in range(0, label_num):
            if dot - thresholds[j] < 0:
                predict_rank = j
                break
        # print(predict_rank)
        label_boolean = []
        if predict_rank != real_rank:
            for i in range(0, label_num):
                if real_rank > float(i):
                    label_boolean.append(float(1))  # real_rank左边的设为1
                else:
                    label_boolean.append(float(-1))  # real_rank右边（包含）的设为-1
            # print(label_boolean)
            t = []
            for i in range(0, label_num):
                temp = (dot - thresholds[i]) * label_boolean[i]
                if temp > 0:  # 分类正确
                    t.append(0)
                else:  # 分类错误
                    t.append(label_boolean[i])
            t_sum = 0
            for i in range(0, len(t)):
                t_sum += t[i]
            for i in range(0, feature_num):
                # print(features[i][0])
                weights[i] = weights[i] + (t_sum * features[i][0])
            for i in range(0, label_num):
                thresholds[i] = thresholds[i] - t[i]
        line = train_fd.readline()
    train_fd.close()
    print("fold"+str(fold_num) + "训练完成！！！")
    return weights, thresholds


def test(test_path, weights, thresholds, fold_num):
    test_fd = codecs.open(test_path, "r", encoding="UTF-8")
    line = test_fd.readline()
    correct_count = float(0)
    all_count = float(0)
    while line:
        all_count += 1
        line = line.rstrip("\r\n")
        lines = line.split(" ")
        real_rank = int(lines[0])
        features = []
        for i in range(0, feature_num):
            feature = [float(lines[i + 2].split(":")[1])]
            features.append(feature)
        features_mat = np.array(features)
        weight_mat = np.array(weights)
        dot = np.dot(weight_mat, features_mat)
        predict_rank = label_num
        for j in range(0, label_num):
            if dot - thresholds[j] < 0:
                predict_rank = j
                break
        if real_rank == predict_rank:
            # print(predict_rank)
            correct_count += 1
        line = test_fd.readline()
    test_fd.close()
    accuary = float(correct_count / all_count)
    print("fold" +  str(fold_num) + "的正确率：" + str(accuary))
    return accuary


if __name__ == '__main__':
    dir = r"E:\data\LTR\Folds"
    files = os.listdir(dir)
    fold_num = 1
    for file in files:
        print(file)
        one_fold_dir = dir + "\\" + file
        train_path = one_fold_dir + "\\" + "train.txt"
        vali_path = one_fold_dir + "\\" + "vali.txt"
        w,b = train(train_path, fold_num)
        test(vali_path, w, b, fold_num)
        fold_num += 1




