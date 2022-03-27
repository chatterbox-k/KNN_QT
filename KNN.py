from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import warnings
warnings.filterwarnings('ignore')

def scaling_the_data(ori_data,label_name='Outcome'):
    '''
    数据标准化处理
    @param ori_data: 原始数据 最后一列为标签['label']/['Outcome']
    @param label_name: 原始数据的标签字段名
    '''
    diabetes_data_copy = ori_data.copy(deep=True)
    sc_X = StandardScaler()

    # 读取特征的字段名,并删除标签字段名
    feature_columns = list(diabetes_data_copy.columns.values)
    feature_columns.remove(label_name)

    X = pd.DataFrame(sc_X.fit_transform(diabetes_data_copy.drop([label_name], axis=1), ),
                     columns=feature_columns)

    # 标准化处理的属性
    sc_X.fit(X)
    print('样本数量：',sc_X.n_samples_seen_)
    print('特征平均值：',sc_X.mean_)
    print('特征方差：',sc_X.var_)
    print('标准差：',sc_X.scale_)

    # x = sc_X.fit_transform(X)
    y = diabetes_data_copy[label_name]
    return X,y

def StratifiedKFold_mean_scoure(k_cv,base_estimator,X,y):
    k_fold = StratifiedKFold(k_cv)  # 10折交叉验证
    test_score = 0
    train_score = 0
    for k, (train, val) in enumerate(k_fold.split(X, y)):
        X_train = X.iloc[train, :]
        y_train = y.iloc[train]
        X_test = X.iloc[val, :]
        y_test = y.iloc[val]
        base_estimator.fit(X_train, y_train)
        train_score += base_estimator.score(X_train, y_train)
        test_score += base_estimator.score(X_test, y_test)
    train_score /= float(k_cv)
    test_score /= float(k_cv)
    return train_score, test_score

def KNN(X,y):
    # 生成训练集 和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 / 3, random_state=42, stratify=y)

    # 绘制不同K值的得分图
    test_scores = []
    train_scores = []
    for k_KNN in range(1, 15):
        train_score, test_score = StratifiedKFold_mean_scoure(k_cv=10, base_estimator=KNeighborsClassifier(k_KNN), X=X, y=y)
        train_scores.append(train_score)
        test_scores.append(test_score)
    plt.figure(figsize=(12, 5))
    p = sns.lineplot(range(1, 15), train_scores, marker='*', label='Train Score')
    p = sns.lineplot(range(1, 15), test_scores, marker='o', label='Test Score')
    plt.show()

    # 选取结果得分最高 K值模型 并保存
    best_k_value = test_scores.index(max(test_scores))
    score = test_scores[best_k_value-1]

    knn = KNeighborsClassifier(best_k_value)
    joblib.dump(knn, "./model/test.pkl")
    knn.fit(X_train, y_train)
    print('Best K_value is %i ,Mean model score is %f:' % (best_k_value, score))

    # 绘制预测结果图
    y_pred = knn.predict(X_test)
    cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
    p = sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt='g')
    plt.title('Confusion matrix', y=1.1)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.show()

    # 绘制ROC 曲线
    y_pred_proba = knn.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr, label='Knn')
    plt.xlabel('fpr')
    plt.ylabel('tpr')
    plt.title('Knn(n_neighbors=%i) ROC curve' %test_scores.index(max(test_scores)))
    plt.show()
    ROC = roc_auc_score(y_test, y_pred_proba)
    print('ROC:', ROC)

def main(data,run_model,model_path=None):
    """
    主程序 训练/应用
    @param data: 数据
    @param run_model: train/apply
    @param model_path: train 不定义model_path
                        apply 需要选择model的路径
    """
    data = pd.read_csv(data)
    if run_model == 'train':
        X, y = scaling_the_data(ori_data=data)
        KNN(X, y)

    # if run_model =='apply':
    #     X, y = scaling_the_data(ori_data=data)
    #     trained_model = joblib.load(model_path)
    #     # trained_model.fit(X,y)
    #     predict_values= trained_model.predict(X[0:1])
    #     print(predict_values)


if __name__ == '__main__':
   main(data='database.csv', run_model='train')
    # main(data='database.csv', run_model='apply', model_path='model/test.pkl')

