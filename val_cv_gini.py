from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
import pandas as pd
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
var_class_all = []
lda_class_all = []
dist_class_all = []

label_name = 'class'
label_nom = 1  # 小类的标签将少数类作为正例，聚焦在少数类上因此异常应为1
path = 'haberman'
path_result = './haberman/result/tree_class/'
path_result_dt='./haberman/result/tree_class/describe_tree/describe'
names = ['Age', ' Year', 'pp', 'class']
names_sort = ['Age', ' Year', 'pp', 'class']
names_nolabel = ['Age', ' Year', 'pp']
numfeatures = ['Age', ' Year', 'pp']
names_nolabel_nosort = ['Age', ' Year', 'pp']
no_num_features = []  # 注意不需加上Label
tolerance_rate =2
issame_rate = 0.2
knn_rate = 2#有自编码器
# knn_rate=0.61#无自编码器
# knn_rate = 2#有自编码器(cross)
# knn_rate = 2.2#有自编码器(re3)

f1_all_1a = []
f1_all_2a = []
f1_all_2b=[]
arra = []
cana = []
deptha = []
roc_alla = []
acc_alla = []
prec_alla = []
recall_alla = []
k=5.0
n=-1
num_class=2
resample_num=10
f1_all_1a = []
f1_all_2a = []
f1_all_2b=[]
arra = []
cana = []
deptha = []
roc_alla = []
acc_alla = []
prec_alla = []
recall_alla = []
num_nodea=[]
can_dt=[]
describe_treea=[]
f1_all_2_dt=[]
arr_dt = []
num_node_dt = []
depth_dt = []
roc_all_dt = []
acc_all_dt = []
prec_all_dt = []
recall_all_dt = []
score_train_dt=[]
k=5.0
n=-1
num_class=2

for b in range(5):
    df = pd.read_csv('./haberman/train_haberman_'+str(b)+'.csv', names=names, header=None)
    df_test = pd.read_csv('./haberman/test_haberman_'+str(b)+'.csv', names=names,
                          header=None)
    df_xy =df
    n=n+1
    x = df[names_nolabel_nosort]
    y = df[label_name]

    x_test = df_test[names_nolabel_nosort]
    y_test = df_test[label_name]

    x_train = x
    y_train = y

    # # 进行特征工程的处理
    # # 进行特征工程的处理
    # dict = DictVectorizer(sparse=False)
    # x_train = dict.fit_transform(x_train.to_dict(orient="records"))
    # print(dict.get_feature_names())
    # x_test = dict.transform(x_test.to_dict(orient="records"))
    # x_train=pd.DataFrame(x_train,columns=dict.get_feature_names())
    # x_test=pd.DataFrame(x_test,columns=dict.get_feature_names())

    num_try = round(np.sqrt(x_train.shape[1]))
    num_try_1=round(np.log2(x_train.shape[1]))
    num_try_2=len(x_train)

    f1_all_1 = []
    f1_all_2 = []
    f1_all_3=[]
    arr = []
    score_train = []
    score_val = []
    can = []
    depth = []
    roc_all = []
    acc_all = []
    prec_all = []
    recall_all = []
    list_t = []
    num_node = []
    describe_tree=[]
    # for i in [1, 3, 4, 5, 6, 7, 8, 9, 10, 20, 40, 50, 100, 120, 200]:
    #     for j in [1, 2, 5, 10, 30]:
    for i in [1]:
        # for j in [1,2,3,5,10,30]:
        for j in [2]:
            rf = RandomForestClassifier(n_estimators=i,
                                 min_samples_leaf=j,
                                 min_split_gain=0.0,
                                        istree=True,
                                        criterion='gini',
                                 colsample_bytree="sqrt",
                                 subsample=0.65,
                                 random_state=2022)
            rf.fit(x_train, y_train)
            y_p= rf.predict(x_test)
            y_train_p = rf.predict(x_train)
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

            lda_class=rf.lda_ratio
            var_class=rf.var_ratio
            dist_class=rf.class_dist
            lda_class_all.append(lda_class)
            var_class_all.append(var_class)
            dist_class_all.append(dist_class)

            print(str(i) + "棵树,最小叶内样本为" + str(j))

            print("f1分数(多类)为：", f1_0a)
            print("f1分数（多类）为：", f1_1a)
            print("f1分数（二类）为：", f1_1b)
            print("auc:", roc_auc)
            print("召回率：", recall)
            print("精确率：", prec)
            print("准确率:", acc)
            print("树高：", rf.depth)
            print("叶子节点数:", rf.num_node)
            print(score_arra)
            score_train.append(train_acc)
            # score_val.append(val_acc)
            list_t.append(rf.list)
            f1_all_1.append(f1_0a)
            f1_all_2.append(f1_1a)
            f1_all_3.append(f1_1b)
            roc_all.append(roc_auc)
            prec_all.append(prec)
            acc_all.append(acc)
            recall_all.append(recall)
            arr.append(score_arra)
            # av_depth = np.mean(np.array(rf.list))
            # depth.append(av_depth)
            depth.append(rf.depth)
            num_node.append(rf.num_node)
            can.append([i, j])
            describe_tree = describe_tree+rf.describe_tree

            print('根节点的分裂点为：'+str(rf.root_best))

            ff=np.array(rf.feature_sort,dtype=[('feature','S10'),('value',float),('score',float)])
            f1=np.sort(ff,order='score')
            print(f1)

    ddd = {"f1_score1": f1_all_1, "f1_score2": f1_all_2, "f1_score": f1_all_3,
               'roc_auc': roc_all, 'recall': recall_all, 'prec': prec_all, 'acc': acc_all,
               "score_arr": arr, "depth": depth, "num_node": num_node, 'can': can, "score_train": score_train}
    print('第' + str(b) + '轮的结果为：')
    print(ddd)
    dfdf = pd.DataFrame(ddd)
    dfdf.to_csv(path_result + 'list_' + str(b) + '.csv')

    dfch = pd.DataFrame(list_t)
    dfch.to_csv(path_result + 'list_trees_' + str(b) + '.csv')

    can_dt = can_dt + can
    describe_treea=describe_treea+describe_tree
    f1_all_2_dt =f1_all_2_dt+f1_all_3
    arr_dt =arr_dt+arr
    num_node_dt =num_node_dt+num_node
    depth_dt =depth_dt+depth
    roc_all_dt = roc_all_dt+roc_all
    acc_all_dt = acc_all_dt+acc_all
    prec_all_dt = prec_all_dt+prec_all
    recall_all_dt = recall_all_dt+recall_all
    score_train_dt =score_train_dt+score_train

    if len(f1_all_1a) == 0:
        f1_all_1a.append(f1_all_1)
        f1_all_2a.append(f1_all_2)
        f1_all_2b.append(f1_all_3)
        roc_alla.append(roc_all)
        prec_alla.append(prec_all)
        acc_alla.append(acc_all)
        recall_alla.append(recall_all)
        arra.append(arr)
        deptha.append(depth)
        num_nodea.append(num_node)
    else:
        f1_all_1a = np.array(f1_all_1a)
        f1_all_2a = np.array(f1_all_2a)
        f1_all_2b = np.array(f1_all_2b)
        roc_alla = np.array(roc_alla)
        prec_alla = np.array(prec_alla)
        acc_alla = np.array(acc_alla)
        recall_alla = np.array(recall_alla)
        arra = np.array(arra)
        deptha = np.array(deptha)
        num_nodea = np.array(num_nodea)
        f1_all_1a = f1_all_1a + np.array(f1_all_1)
        f1_all_2a = f1_all_2a + np.array(f1_all_2)
        f1_all_2b = f1_all_2b + np.array(f1_all_3)
        roc_alla = roc_alla + np.array(roc_all)
        prec_alla = prec_alla + np.array(prec_all)
        acc_alla = acc_alla + np.array(acc_all)
        recall_alla = recall_alla + np.array(recall_all)
        arra = arra + np.array(arr)
        deptha = deptha + np.array(depth)
        num_nodea = num_nodea + np.array(num_node)

ccc_dt={"lda_class_all": lda_class_all,"var_class_all": var_class_all,
          'dist_class_all': dist_class_all,"can": can_dt}
ccc_dtpd=pd.DataFrame(ccc_dt)
ccc_dtpd.to_csv(path_result_dt+'clss_ratio2-2newscaler.csv')

ddd_dt = {"describe_tree": describe_treea, "can": can_dt,"f1_score1": f1_all_2_dt,
          'roc_auc': roc_all_dt, 'recall': recall_all_dt, 'prec': prec_all_dt, 'acc': acc_all_dt,
          "score_arr": arr_dt, "depth": depth_dt, "num_node": num_node_dt,"score_train": score_train_dt}
ddd_dtpd=pd.DataFrame(ddd_dt)
ddd_dtpd.to_csv(path_result_dt + 'list_data2-2newscaler.csv')

ddda = {"f1_score1": (f1_all_1a / k).tolist()[0], "f1_score2": (f1_all_2a / k).tolist()[0],
        "f1_score": (f1_all_2b / k).tolist()[0],
        'roc_auc': (roc_alla / k).tolist()[0], 'recall': (recall_alla / k).tolist()[0],
        'prec': (prec_alla / k).tolist()[0], 'acc': (acc_alla / k).tolist()[0],
        "score_arr": (arra / k).tolist()[0], "depth": (deptha / k).tolist()[0],
        "num_node": (num_nodea / k).tolist()[0], 'can': can}
print('最终结果为：')
print(ddda)
dfdfa = pd.DataFrame(ddda)
dfdfa.to_csv(path_result + 'list_data2-2newscaler.csv')

print("完成")

