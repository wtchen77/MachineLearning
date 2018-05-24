# coding: utf-8
"""
利用鸢尾花数据集，建立一个机器学习项目；
1.导入数据
2.概述数据
3.数据可视化
4.评估算法
5.实施预测
"""
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# 导入数据
path = 'iris.data'
iris_feature = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
data = pd.read_csv(path, names=iris_feature)

# 数据统计信息
print('数据维度：行： %d，列 %d' % data.shape)
print('\n数据前10行：')
print(data.head(10))
print('\n数据描述性统计信息：')
print(data.describe())

# 数据分类分布
print(data.groupby('class').size())

# 数据可视化
# 箱线图
data.plot(kind='box', subplots=True, layout=(2, 2), sharex=False, sharey=False)
plt.show()
# 直方图
data.hist()
plt.show()
# 散点矩阵图
scatter_matrix(data)
plt.show()


# 分离数据集
array = data.values
x = array[:, :4]
y = array[:, 4]
seed = 7
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=seed)

# 算法评估
models = {}
models['LR'] = LogisticRegression()
models['LDA'] = LinearDiscriminantAnalysis()
models['KNN'] = KNeighborsClassifier()
models['CART'] = DecisionTreeClassifier()
models['NB'] = GaussianNB()
models['SVM'] = SVC()

print('\n算法比较：')
results = []
for key in models:
    kfold = KFold(n_splits=10, random_state=seed)
    cv_results = cross_val_score(models[key], x_train, y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    print('%s: %f (%f)' % (key, cv_results.mean(), cv_results.std()))

# 箱线图比较算法
fig = plt.figure()
fig.suptitle('algorithm comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(models.keys())
plt.show()

# 使用测试集评估SVM模型
clf = SVC()
clf.fit(x_train, y_train)
pred = clf.predict(x_test)
print('\nSVM模型准确率：%.2f%%' % (accuracy_score(y_test, pred)*100))
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))