
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
import pandas as pd
from sklearn.manifold import TSNE
from  sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

# from RF import RandomForestClassifier
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn import tree
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from RandomForestClassification import RandomForestClassifier
import numpy as np
import random
import os
import torch

seed_value = 2022   # 设定随机数种子

np.random.seed(seed_value)
random.seed(seed_value)
os.environ['PYTHONHASHSEED'] = str(seed_value)  # 为了禁止hash随机化，使得实验可复现。

torch.manual_seed(seed_value)     # 为CPU设置随机种子
torch.cuda.manual_seed(seed_value)      # 为当前GPU设置随机种子（只用一块GPU）
torch.cuda.manual_seed_all(seed_value)   # 为所有GPU设置随机种子（多块GPU）

torch.backends.cudnn.deterministic = True

#使用np.eye(2)生成单位矩阵,然后乘以一个随机生成得均匀分布值组成单位矩阵得值
x0 = np.random.multivariate_normal([10,30], np.eye(2)*np.random.uniform(4,5,2), 200)
x1 = np.random.multivariate_normal([35,30], np.eye(2)*np.random.uniform(4,5,2), 200)
y0=np.array([0]*200)
y1=np.array([1]*200)
y=np.append(y0,y1,axis=0)
x=np.append(x0,x1,axis=0)
x0_1 = np.random.multivariate_normal([35,50], np.eye(2)*np.random.uniform(4,5,2), 200)
x1_1 = np.random.multivariate_normal([10,50], np.eye(2)*np.random.uniform(4,5,2), 200)
# x2 = np.random.multivariate_normal(np.random.uniform(-50,50,2), np.eye(2)*np.random.uniform(5,20,2), 1000)
#numpy中append得用法
y0_1=np.array([0]*200)
y1_1=np.array([1]*200)
y=np.append(y,y0_1,axis=0)
y=np.append(y,y1_1,axis=0)
x=np.append(x,x0_1,axis=0)
x=np.append(x,x1_1,axis=0)

x0=np.concatenate((x0,x0_1),axis=0)
x1=np.concatenate((x1,x1_1),axis=0)

x_train=pd.DataFrame(x,columns=['1','2'])
y_train=pd.Series(y)

#使用np.eye(2)生成单位矩阵,然后乘以一个随机生成得均匀分布值组成单位矩阵得值
x0_test = np.random.multivariate_normal([10,30], np.eye(2)*np.random.uniform(4,5,2), 50)
x1_test = np.random.multivariate_normal([35,30], np.eye(2)*np.random.uniform(4,5,2), 50)
y0_test=np.array([0]*50)
y1_test=np.array([1]*50)
y_test=np.append(y0_test,y1_test,axis=0)
x_test=np.append(x0_test,x1_test,axis=0)
x0_1_test = np.random.multivariate_normal([35,50], np.eye(2)*np.random.uniform(4,5,2), 50)
x1_1_test = np.random.multivariate_normal([10,50], np.eye(2)*np.random.uniform(4,5,2), 50)
# x2 = np.random.multivariate_normal(np.random.uniform(-50,50,2), np.eye(2)*np.random.uniform(5,20,2), 1000)
#numpy中append得用法
y0_1_test=np.array([0]*50)
y1_1_test=np.array([1]*50)
y_test=np.append(y_test,y0_1_test,axis=0)
y_test=np.append(y_test,y1_1_test,axis=0)
x_test=np.append(x_test,x0_1_test,axis=0)
x_test=np.append(x_test,x1_1_test,axis=0)

x0_test=np.concatenate((x0_test,x0_1_test),axis=0)
x1_test=np.concatenate((x1_test,x1_1_test),axis=0)

x_test=pd.DataFrame(x_test,columns=['1','2'])
y_test=pd.Series(y_test)

# #将生成得图可视化
# import matplotlib.pyplot as plt
# plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
# plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
# plt.figure(dpi=100, figsize=(15, 15))
# plt.xticks(fontsize=10)
# plt.xlim((0,60))
# plt.ylim((0,60))
#
# plt.scatter(x0[:,0],x0[:,1])
# plt.scatter(x1[:,0],x1[:,1])
# #first代表x轴，second代表y轴
# first=[0,60]
# second=[52.57114886418613,52.57114886418613]
# first1=[]
# second1=[]
# first2=[]
# second2=[]
# print(first,second)
# print(first1,second1)
# print(first2,second2)
# plt.plot(first,second,color='red')
# plt.plot(first1,second1,color='yellow')
# plt.plot(first2,second2,color='green')
# plt.savefig('tu_ceshi_gini')
# plt.show()
mm=100
zi='m'
rf2 = RandomForestClassifier(n_estimators=1,
                             min_samples_leaf=1,
                             min_split_gain=-100.0,
                             istree=True,
                             max_depth=mm,
                             rand_num=1600,
                             issort_min=True,
                             w_class=0.5, w_class2=1.2,
                             colsample_bytree="sqrt",
                             # criterion='gini_var_class_sort',
                             # criterion='gini_var3',
                             # criterion='gini_class2',
                             criterion='gini',
                             subsample=1.0,
                             random_state=2022)
rf2.fit(x_train, y_train)
y_train_p_class = rf2.predict(x_train)
train_acc = accuracy_score(y_train, y_train_p_class)
print(train_acc)
y_p = rf2.predict(x_test)
y_train_p = rf2.predict(x_train)
fpr, tpr, threshold = roc_curve(y_test, y_p)  ###计算真正率和假正率
roc_auc = auc(fpr, tpr)  ###计算auc的值
f1_0a = f1_score(y_test, y_p, average='micro')
f1_1a = f1_score(y_test, y_p, average='macro')
f1_1b = f1_score(y_test, y_p)
score_arra = confusion_matrix(y_test, y_p, labels=[1, 0])
acc = accuracy_score(y_test, y_p)
recall = recall_score(y_test, y_p)
prec = precision_score(y_test, y_p)
train_acc = accuracy_score(y_train, y_train_p)

print('最小')
print("f1分数(多类)为：", f1_0a)
print("f1分数（多类）为：", f1_1a)
print("f1分数（二类）为：", f1_1b)
print("auc:", roc_auc)
print("召回率：", recall)
print("精确率：", prec)
print("准确率:", acc)
print("树高：", rf2.depth)
print("叶子节点数:", rf2.num_node)
print(score_arra)

# spiltting_value=rf2.spiltting_value
# spiltting_feature1,spiltting_value1=spiltting_value[1]
# spiltting_feature2,spiltting_value2=spiltting_value[2]
# spiltting_feature5,spiltting_value5=spiltting_value[5]
# max_dd=max(spiltting_value.keys())
# spiltting_feature_,spiltting_value_=spiltting_value[max_dd]

# #将生成得图可视化
import matplotlib.pyplot as plt
def tree_plt(tree,max_depth):
    if tree.leaf_value == None:
        if tree.split_feature == '1':
            first = [tree.split_value, tree.split_value]
            second = [tree.y_down, tree.y_up]
        else:
            first = [tree.x_down, tree.x_up]
            second = [tree.split_value, tree.split_value]
        plt.plot(first, second, color='black',linewidth=2)
        if tree.depth<max_depth:
            tree_plt(tree.tree_left,max_depth)
            tree_plt(tree.tree_right,max_depth)
    else:
        return
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
plt.rcParams['xtick.direction'] = 'in'  # 将x周的刻度线方向设置向内
plt.rcParams['ytick.direction'] = 'in'  # 将y轴的刻度方向设置向内
for i in [100]:
    plt.figure(dpi=300, figsize=(10, 10))
    plt.rc('font', family='Times New Roman')
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.xlim((0,50))
    plt.ylim((0,60))
    ax = plt.gca();  # 获得坐标轴的句柄
    ax.spines['bottom'].set_linewidth(2);  ###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(2);  ####设置左边坐标轴的粗细
    ax.spines['top'].set_linewidth(2);  ###设置底部坐标轴的粗细
    ax.spines['right'].set_linewidth(2);  ####设置左边坐标轴的粗细
    # front是标签属性：包括字体、大小等
    font = {'family': 'Times New Roman',
            'weight': 'normal',
            'size': 25,
            }
    # plt.xlabel("x1", font)
    # plt.ylabel("x2", font)
    # plt.title("(a)", loc='top')
    plt.text(24,-7,'('+zi+')',font)

    plt.scatter(x0[:, 0], x0[:, 1],marker='+',s=75)
    plt.scatter(x1[:, 0], x1[:, 1],color='w', marker='o', edgecolors='#FF4500', s=75)
    point_dict={}
    tree=rf2.trees[0]
    tree_plt(tree,i)
    plt.savefig('./black_white/gini/'+'gini_depth_ceshi'+str(mm)+'.svg',format='svg',dpi=400)
    # plt.show()



# best_split_feature, best_split_value, best_split_gain=rf2.root_best
# best_split_feature1, best_split_value1, best_split_gain1=rf2.sec_best1
# best_split_feature2, best_split_value2, best_split_gain2=rf2.sec_best2
#
# left_dataset = x_train[x_train[best_split_feature] <= best_split_value].values
# left_targets = y_train[x_train[best_split_feature] <= best_split_value].values.flatten()
# right_dataset = x_train[x_train[best_split_feature] > best_split_value].values
# right_targets = y_train[x_train[best_split_feature] > best_split_value].values.flatten()
# aver_l = np.zeros((2, left_dataset.shape[1]))
# aver_r = np.zeros((2, left_dataset.shape[1]))
# for i in [0,1]:
#     index_ = np.where(left_targets == i)
#     set_ = np.array(left_dataset[index_])
#     # v = np.var(set_, axis=0)
#     # v1 = np.sum(v) / len(v)
#     # var_all_left = var_all_left + v1
#     aver_l[i] = np.mean(set_, axis=0)
# for j in [0,1]:
#     index_ = np.where(right_targets == j)
#     set_ = np.array(right_dataset[index_])
#     # v = np.var(set_, axis=0)
#     # v1 = np.sum(v) / len(v)
#     # var_all_right = var_all_right + v1
#     aver_r[j] = np.mean(set_, axis=0)
#
# #将生成得图可视化
# import matplotlib.pyplot as plt
# plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
# plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
# plt.figure(dpi=100, figsize=(15, 15))
# plt.xticks(fontsize=10)
# plt.xlim((0,60))
# plt.ylim((0,60))
#
# plt.scatter(x0[:,0],x0[:,1])
# plt.scatter(x1[:,0],x1[:,1])
#
# plt.scatter(aver_l[:,0],aver_l[:,1],color='red')
# plt.scatter(aver_r[:,0],aver_r[:,1],color='yellow')
# if best_split_feature=='1':
#     first=[best_split_value,best_split_value]
#     second=[0,60]
# else:
#     first = [0,60]
#     second = [best_split_value,best_split_value]
#
# if best_split_feature1=='1':
#     first1=[best_split_value1,best_split_value1]
#     second1=[0,60]
# else:
#     first1 = [0,60]
#     second1= [best_split_value1,best_split_value1]
# if best_split_feature2=='1':
#     first2=[best_split_value2,best_split_value2]
#     second2=[0,60]
# else:
#     first2 = [0,60]
#     second2 = [best_split_value2,best_split_value2]
# print(first,second)
# print(first1,second1)
# print(first2,second2)
# plt.plot(first,second,color='red')
# plt.plot(first1,second1,color='yellow')
# plt.plot(first2,second2,color='green')
# plt.savefig('tu_ceshi_gini')
# plt.show()
