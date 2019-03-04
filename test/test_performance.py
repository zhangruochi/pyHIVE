#coding: utf-8


import pickle
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import matthews_corrcoef
import numpy as np
import warnings

from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split
import pandas as pd
from matplotlib import pyplot as plt

def plot_roc(auc_value,fpr,tpr):
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', linewidth=lw, label='ROC curve (area = %0.4f)' % auc_value)
    plt.plot([0, 1], [0, 1], color='navy', linewidth=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Transcation Suspicious Activities Detection')
    plt.legend(loc="lower right")
    plt.show()

# 读取 csv 文件
def prepare(filename):

    with open(filename, "rb") as f:
        dataset = pickle.load(f)

    # 生成类标
    labels = []
    for name in dataset.index.tolist():
        if name.startswith("normal"):
            labels.append(0)
        else:
            labels.append(1)

    labels = np.array(labels)
    return dataset, labels


# 选择分类器 D-tree,SVM,NBayes,KNN
def select_estimator(case):

    if case == 0:
        estimator = SVC()
    elif case == 1:
        estimator = RandomForestClassifier(random_state=7)
    elif case == 2:
        estimator = DecisionTreeClassifier(random_state=7)
    elif case == 3:
        estimator = GaussianNB()
    elif case == 4:
        estimator = LogisticRegression()
    elif case == 5:
        estimator = KNeighborsClassifier()

    return estimator


def evaluate(estimator, X, y, skf):
    warnings.filterwarnings("ignore")
    acc_list, sn_list, sp_list, mcc_list = [], [], [], []
    for train_index, test_index in skf.split(X, y):
        estimator.fit(X.iloc[train_index], y[train_index])
        y_predict = estimator.predict(X.iloc[test_index])
        y_true = y[test_index]

        # 索引
        predict_index_p = (y_predict == 1)  # 预测为正类的
        predict_index_n = (y_predict == 0)  # 预测为负类

        index_p = (y_true == 1)  # 实际为正类
        index_n = (y_true == 0)  # 实际为负类

        Tp = float(sum(y_true[predict_index_p]))  # 正确预测的正类  （实际为正类 预测为正类）
        # 正确预测的负类   (实际为负类 预测为负类)
        Tn = float(sum([1 for x in list(y_true[predict_index_n]) if x == 0]))
        Fn = float(sum(y_predict[index_n]))  # 错误预测的负类  （实际为负类 预测为正类）
        Fp = float(sum(y_true[predict_index_n]))  # 错误预测的正类   (实际为正类 预测为负类)

        try:
            acc = (Tp + Tn) / (Tp + Tn + Fp + Fn)
        except:
            acc = 0
        try:
            sn = Tp / (Tp + Fn)
        except:
            sn = 0
        try:
            sp = Tn / (Tn + Fp)
        except:
            sp = 0
        try:
            mcc = matthews_corrcoef(y_true, y_predict)
        except:
            mcc = 0

        acc_list.append(acc)
        sn_list.append(sn)
        sp_list.append(sp)
        mcc_list.append(mcc)

    return np.mean(acc_list), np.mean(sn_list), np.mean(sp_list), np.mean(mcc_list)


def ruc_auc():
    # -------------参数------------------
    filename = "Img_LBP.pkl"
    # -------------参数------------------

    dataset, labels = prepare(filename)
    print("dataset shape:", dataset.shape)
    labels = np.array(labels)

    X_train, X_test, y_train, y_test = train_test_split(
    dataset, labels, test_size=0.3, random_state=0)

    clf = LogisticRegression(solver="liblinear")
    #clf = RandomForestClassifier()
    clf.fit(X_train, y_train)

    print(clf.score(X_test,y_test))

    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))
    print()

    
    print("plot the orc curve:\n")
    y_pred_pro = clf.predict_proba(X_test)

    y_scores = pd.DataFrame(y_pred_pro, columns=clf.classes_.tolist())[1].values
    auc_value = roc_auc_score(y_true, y_scores)
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1.0)
    plot_roc(auc_value,fpr,tpr)

# 主函数
def main():
    # -------------参数------------------
    n = 10  # 采用 n 折交叉验证
    filename = "Img_HESSIAN_GLCM.pkl"
    # -------------参数------------------

    dataset, labels = prepare(filename)
    print("dataset shape:", dataset.shape)

    labels = np.array(labels)
    estimator_list = [0, 1, 2, 3, 4, 5]
    skf = StratifiedKFold(n_splits=n, random_state=7)

    for i in estimator_list:
        acc, sn, sp, mcc = evaluate(select_estimator(i), dataset, labels, skf)
        print("Acc: ", acc)
        print("Sn: ", sn)
        print("Sp: ", sp)
        print("Mcc: ", mcc)
        print("\n")

if __name__ == '__main__':
    ruc_auc()
