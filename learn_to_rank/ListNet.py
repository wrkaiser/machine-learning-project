import os
import codecs
import numpy as np
import math


epoch = 5
feature_num = 46
learning_rate = 0.00003


def train(train_path, fold_num):
    # train_fd = pd.read_csv(train_path, sep=',', engine = 'python', iterator=True, encoding="UTF-8")
    train_fd = codecs.open(train_path, "r", encoding="UTF-8")
    # chunk_size = 1000
    pre_qid = "-1"
    # query_index = 0
    query_doc_frequency_list = []
    features_list = []
    label_list = []
    doc_frequency_of_one_query = 0
    line = train_fd.readline()
    line_count = 1
    while line:
        line = line.rstrip("\r\n")
        # print(line_count)
        line_count += 1
        lines = line.split(" ")
        qid = lines[1].split(":")[1]
        # print(type(qid))
        if qid != pre_qid:
            if pre_qid != "-1":
                query_doc_frequency_list.append(doc_frequency_of_one_query)
            # query_index += 1
            pre_qid = qid
            doc_frequency_of_one_query = 0
        doc_frequency_of_one_query += 1
        feature_list = []
        for i in range(2, 48):
            feature_list.append(float(lines[i].split(":")[1]))
        features_list.append(feature_list)
        line = train_fd.readline()
    query_doc_frequency_list.append(doc_frequency_of_one_query)
    train_fd.close()
    # print(query_doc_frequency_list)
    # print(features_list)
    # print(label_list)
    weights = []
    for i in range(0, feature_num):
        weights.append(float(1))
    for i in range(0, epoch):
        now_index = 0
        for j in range(0, len(query_doc_frequency_list)):
            one_query_doc_frequency = query_doc_frequency_list[j]
            f = []
            for k in range(0, one_query_doc_frequency):
                f.append(float(0))
            for k in range(0, one_query_doc_frequency):
                feature = features_list[now_index + k]
                features_mat = np.array(feature)  # transfer to np array
                weights_mat = np.array(weights)
                f[k] = np.dot(weights_mat, features_mat)  # 采用不考虑b的线性神经网络
            delta_w = []
            a = []
            c = []
            for l in range(0, feature_num):
                a.append(float(0))
                c.append(float(0))
                delta_w.append(float(0))
            denominator = float(0)  # luce模型 近似分母  TOK-1
            probs = []
            for l in range(0, one_query_doc_frequency):
                denominator = denominator + math.exp(f[l])
            for l in range(0, one_query_doc_frequency):
                probs.append(math.exp(f[l])/denominator) # n个文档分别排top1的概率
            for l in range(0, one_query_doc_frequency):  # a
                for k in range(0, feature_num):
                    feature = features_list[now_index + l]
                    a[k] = a[k] + probs[l] * feature[k]
            b = denominator
            for l in range(0, one_query_doc_frequency):
                for k in range(0, feature_num):
                    feature = features_list[now_index + l]
                    c[k] = c[k] + math.exp(f[l]) * feature[k]
            for l in range(0, feature_num):
                delta_w[l] = (-1)*a[l] + (1.0/b)*c[l]
            for l in range(0, feature_num):
                weights[l] = weights[l] - learning_rate*delta_w[l]
            now_index = now_index + one_query_doc_frequency
        # print("epoch-" + str(i) + "completed!")
    print("fold" + str(fold_num) + "completed!")
    return weights


def test(test_path, weights, fold_num):
    test_fd = codecs.open(test_path, "r", encoding="UTF-8")
    correct_count = float(0)
    all_count = float(0)
    line = test_fd.readline()
    pre_qid = -1
    label_list = []
    features_list = []
    weights_mat = np.array(weights)
    while line:
        all_count += 1
        line = line.rstrip("\r\n")
        lines = line.split(" ")
        qid = lines[1].split(":")[1]
        real_rank = int(lines[0])  # 相关等级 实际值
        if qid != pre_qid:
            if pre_qid != -1:
                key_score_list = []
                for i in range(0, len(features_list)):
                    feature = features_list[i]
                    features_mat = np.array(feature)
                    score = np.dot(weights_mat, features_mat)
                    key_score_pair= []
                    key_score_pair.append(i)
                    key_score_pair.append(score)
                    key_score_list.append(key_score_pair)
                key_score_list.sort(key=lambda temp: temp[1], reverse=True)
                for i in range(0, len(key_score_list)):
                    key_score_pair = key_score_list[i]
                    index = key_score_pair[0]
                    if index == label_list[i]:
                        correct_count += 1
                features_list = []
                label_list = []
            pre_qid = qid
        label_list.append(real_rank)
        feature_list = []
        for i in range(2, 48):
            feature_list.append(float(lines[i].split(":")[1]))
        features_list.append(feature_list)
        line = test_fd.readline()
    test_fd.close()
    accuary = float(correct_count / all_count)
    print("fold" + str(fold_num) + "的正确率：" + str(accuary))
    return accuary


if __name__ == '__main__':
    dir = r"E:\data\LTR\MQ2007"
    files = os.listdir(dir)
    fold_num = 1
    for file in files:
        print(file)
        one_fold_dir = dir + "\\" + file
        train_path = one_fold_dir + "\\" + "train.txt"
        vali_path = one_fold_dir + "\\" + "vali.txt"
        w = train(train_path, fold_num)
        test(vali_path, w, fold_num)
        fold_num += 1







