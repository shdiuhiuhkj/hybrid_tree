# -*- coding: utf-8 -*-
"""
@Env: Python2.7
@Time: 2019/10/24 13:31
@Author: zhaoxingfeng
@Function：Random Forest（RF），随机森林二分类
@Version: V1.2
参考文献：
[1] UCI. wine[DB/OL].https://archive.ics.uci.edu/ml/machine-learning-databases/wine.
"""
import pandas as pd
import numpy as np
import random
import math
import collections
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import pdist

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


class Tree(object):
    """定义一棵决策树"""
    def __init__(self):
        self.split_feature = None
        self.split_value = None
        self.leaf_value = None
        self.tree_left = None
        self.tree_right = None
        self.train_sample_num=None
        self.depth=None
        self.gini=None
        self.lda_ratio=None
        self.var_ratio=None
        self.class_dist=None
        self.class_lda=None
        self.class_var=None

        self.x_down=None
        self.y_down=None
        self.x_up=None
        self.y_up=None

    def calc_predict_value(self, dataset):
        """通过递归决策树找到样本所属叶子节点"""
        if self.leaf_value is not None:
            return self.leaf_value
        elif dataset[self.split_feature] <= self.split_value:
            return self.tree_left.calc_predict_value(dataset)
        else:
            return self.tree_right.calc_predict_value(dataset)

    def describe_tree(self):
        """以json形式打印决策树，方便查看树结构"""
        if not self.tree_left and not self.tree_right:
            leaf_info = "{leaf_value:" + str(self.leaf_value) +\
                         ",train_num_sample:" + str(self.train_sample_num) + \
                        ",class_lda:" + str(self.class_lda) + \
                        ",class_var:" + str(self.class_var) + \
                        ",class_dist:" + str(self.class_dist) + \
                        ",depth:" + str(self.depth) + "}"
            return leaf_info
        left_info = self.tree_left.describe_tree()
        right_info = self.tree_right.describe_tree()
        tree_structure = "{split_feature:" + str(self.split_feature) + \
                         ",split_value:" + str(self.split_value) + \
                         ",gini_value:" + str(self.gini) + \
                         ",train_num_sample:" + str(self.train_sample_num) + \
                         ",lda_ratio:" + str(self.lda_ratio) + \
                         ",var_ratio:" + str(self.var_ratio) + \
                         ",class_lda:" + str(self.class_lda) + \
                         ",class_var:" + str(self.class_var) + \
                         ",class_dist:" + str(self.class_dist) + \
                         ",depth:" + str(self.depth) + \
                         ",left_tree:" + left_info + \
                         ",right_tree:" + right_info + "}"
        return tree_structure


class RandomForestClassifier(object):
    def __init__(self, n_estimators=10, max_depth=-1, min_samples_split=2, min_samples_leaf=1,
                 min_split_gain=0.0, colsample_bytree=None, subsample=0.8, random_state=None,issort_min=True,
                 criterion='gini',istree=False,w_=0.0,w_class=0.0,w_class2=0.0,w_dist=0.0,rand_num=0):
        """
        随机森林参数
        ----------
        n_estimators:      树数量
        max_depth:         树深度，-1表示不限制深度
        min_samples_split: 节点分裂所需的最小样本数量，小于该值节点终止分裂
        min_samples_leaf:  叶子节点最少样本数量，小于该值叶子被合并
        min_split_gain:    分裂所需的最小增益，小于该值节点终止分裂
        colsample_bytree:  列采样设置，可取[sqrt、log2]。sqrt表示随机选择sqrt(n_features)个特征，
                           log2表示随机选择log(n_features)个特征，设置为其他则不进行列采样
        subsample:         行采样比例
        random_state:      随机种子，设置之后每次生成的n_estimators个样本集不会变，确保实验可重复
        """
        self.rand_num=rand_num
        self.w_=w_
        self.w_class=w_class
        self.w_class2=w_class2
        self.w_dist=w_dist
        self.n_estimators = n_estimators
        self.issort_min=issort_min
        self.max_depth = max_depth if max_depth != -1 else float('inf')
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_split_gain = min_split_gain
        self.colsample_bytree = colsample_bytree
        self.subsample = subsample
        self.random_state = random_state
        self.trees = None
        self.feature_importances_ = dict()
        self.list=[]
        self.criterion=criterion
        self.istree=istree
        self.depth=0
        self.num_node=0
        self.describe_tree=[]
        self.feature_sort=[]
        self.feature_sort1=[]
        self.feature_sort1_1=[]
        self.feature_sort2=[]
        self.feature_sort2_2=[]
        self.feature_sort2_2_2=[]
        self.feature_sort2_2_2_2=[]
        self.lda_ratio={}
        self.var_ratio={}
        self.class_dist={}
        self.var={}
        self.lda={}
        self.sort_index=[]
        self.root_best=None
        self.sec_best1=None
        self.sec_best2 = None
        self.spiltting_value={}


    def fit(self, dataset, targets):
        """模型训练入口"""
        assert targets.unique().__len__() == 2, "There must be two class for targets!"
        targets = targets.to_frame(name='label')

        if self.random_state:
            random.seed(self.random_state)
        random_state_stages = random.sample(range(self.n_estimators), self.n_estimators)

        # 两种列采样方式
        if self.colsample_bytree == "sqrt":
            self.colsample_bytree = int(len(dataset.columns) ** 0.5)
        elif self.colsample_bytree == "log2":
            self.colsample_bytree = int(math.log(len(dataset.columns)))
        else:
            self.colsample_bytree = len(dataset.columns)

        # 并行建立多棵决策树
        self.trees = Parallel(n_jobs=-1, verbose=0, backend="threading")(
            delayed(self._parallel_build_trees)(dataset, targets, random_state)
                for random_state in random_state_stages)
        
    def _parallel_build_trees(self, dataset, targets, random_state):
        """bootstrap有放回抽样生成训练样本集，建立决策树"""
        if self.istree:
            dataset_stage=dataset
            targets_stage=targets
        else:
            subcol_index = random.sample(dataset.columns.tolist(), self.colsample_bytree)
            dataset_stage = dataset.sample(n=int(self.subsample * len(dataset)), replace=True,
                                           random_state=random_state).reset_index(drop=True)
            dataset_stage = dataset_stage.loc[:, subcol_index]
            targets_stage = targets.sample(n=int(self.subsample * len(dataset)), replace=True,
                                           random_state=random_state).reset_index(drop=True)
        self.list.append(0)
        tree,_,_,_= self._build_single_tree(dataset_stage, targets_stage, depth=0)
        print(tree.describe_tree())
        if dataset_stage.shape[1]==2:
            self.tree_down_up(tree)
        self.describe_tree.append(tree.describe_tree())
        return tree

    def tree_down_up(self,tree):
        x_down = tree.x_down
        x_up = tree.x_up
        y_down = tree.y_down
        y_up = tree.y_up

        if tree.split_feature == '1':
            tree.tree_left.x_down = x_down
            tree.tree_left.x_up = tree.split_value
            tree.tree_left.y_down = y_down
            tree.tree_left.y_up = y_up
            tree.tree_right.x_down = tree.split_value
            tree.tree_right.x_up = x_up
            tree.tree_right.y_down = y_down
            tree.tree_right.y_up = y_up
            # 相当于在x轴上分割
        elif tree.split_feature == '2':
            tree.tree_left.x_down = x_down
            tree.tree_left.x_up = x_up
            tree.tree_left.y_down = y_down
            tree.tree_left.y_up = tree.split_value
            tree.tree_right.x_down = x_down
            tree.tree_right.x_up = x_up
            tree.tree_right.y_down = tree.split_value
            tree.tree_right.y_up = y_up
            # 相当于在y轴上分割
        if tree.tree_left.leaf_value==None:
            self.tree_down_up(tree.tree_left)
        if tree.tree_right.leaf_value==None:
            self.tree_down_up(tree.tree_right)


    def _build_single_tree(self, dataset, targets, depth):
        """递归建立决策树"""
        # 如果该节点的类别全都一样/样本小于分裂所需最小样本数量，则选取出现次数最多的类别。终止分裂

        class_lda,class_var,class_dist=self.calc_class_lda(dataset,targets)
        keys_dist=self.class_dist.keys()
        if depth in keys_dist:
            self.class_dist[depth].append(class_dist)
        else:
            self.class_dist[depth]=[class_dist]
        if len(targets['label'].unique()) <= 1 or dataset.__len__() <= self.min_samples_split:
            tree = Tree()
            tree.leaf_value = self.calc_leaf_value(targets['label'])
            tree.depth = depth
            tree.train_sample_num = len(dataset)
            self.num_node=self.num_node+1
            tree.class_dist=class_dist
            tree.class_lda=class_lda
            tree.class_var=class_var
            return tree,class_lda,class_var,class_dist

        if depth < self.max_depth:
            if depth>self.depth:
                self.depth=depth
            best_split_feature, best_split_value, best_split_gain = \
                self.choose_best_feature(dataset, targets,depth)
            spilt_value=(best_split_feature, best_split_value)
            keys_dist_spilt = self.spiltting_value.keys()
            if depth in keys_dist_spilt:
                self.spiltting_value[depth].append(spilt_value)
            else:
                self.spiltting_value[depth] = [spilt_value]
            # best_split_feature, best_split_value='DiabetesPedigreeFunction',0.085
            left_dataset, right_dataset, left_targets, right_targets = \
                self.split_dataset(dataset, targets, best_split_feature, best_split_value)

            tree = Tree()
            if depth == 0 and dataset.shape[1]==2 :
                tree.x_down=0
                tree.x_up=50
                tree.y_down=0
                tree.y_up=60
            # 如果父节点分裂后，左叶子节点/右叶子节点样本小于设置的叶子节点最小样本数量，则该父节点终止分裂
            if left_dataset.__len__() < self.min_samples_leaf or \
                    right_dataset.__len__() < self.min_samples_leaf or \
                    best_split_gain <= self.min_split_gain:
                tree.leaf_value = self.calc_leaf_value(targets['label'])
                tree.depth=depth
                tree.train_sample_num=len(dataset)
                self.num_node = self.num_node + 1
                tree.class_dist = class_dist
                tree.class_lda = class_lda
                tree.class_var = class_var
                return tree,class_lda,class_var,class_dist
            else:
            # if True:
                # 如果分裂的时候用到该特征，则该特征的importance加1
                self.feature_importances_[best_split_feature] = \
                    self.feature_importances_.get(best_split_feature, 0) + 1

                # x_down=tree.x_down
                # x_up=tree.x_up
                # y_down=tree.y_down
                # y_up=tree.y_up

                tree.split_feature = best_split_feature
                tree.split_value = best_split_value
                tree.gini=best_split_gain
                tree.train_sample_num=len(dataset)
                tree.depth=depth
                tree.tree_left,cl_class_lda,cl_class_var,cl_class_dist = self._build_single_tree(left_dataset, left_targets, depth+1)
                tree.tree_right,cr_class_lda,cr_class_var,cr_class_dist = self._build_single_tree(right_dataset, right_targets, depth+1)
                # #记录二维数据的非叶节点的范围
                # if dataset.shape[1]==2:
                #     if best_split_feature=='1':
                #         tree.tree_left.x_down=x_down
                #         tree.tree_left.x_up=best_split_value
                #         tree.tree_left.y_down=y_down
                #         tree.tree_left.y_up=y_up
                #         tree.tree_right.x_down=best_split_value
                #         tree.tree_right.x_up=x_up
                #         tree.tree_right.y_down=y_down
                #         tree.tree_right.y_up=y_up
                #         #相当于在x轴上分割
                #     elif best_split_feature=='2':
                #         tree.tree_left.x_down=x_down
                #         tree.tree_left.x_up=x_up
                #         tree.tree_left.y_down=y_down
                #         tree.tree_left.y_up=best_split_value
                #         tree.tree_right.x_down=x_down
                #         tree.tree_right.x_up=x_up
                #         tree.tree_right.y_down=best_split_value
                #         tree.tree_right.y_up=y_up
                #         #相当于在y轴上分割

                # 记录当前节点的类紧性，得到子节点的类紧性，输出父节点和子节点的类紧性比例
                cl_class_lda,cl_class_var,cl_class_dist,cr_class_lda,cr_class_var,cr_class_dist\
                    =self.calc_class_lda2(left_targets, right_targets, left_dataset, right_dataset)
                # class_lda_all,class_var_all,dist_all\
                # #测试
                # lda_ratio1=(cl_class_lda+cr_class_lda)/class_lda_all
                # var_ratio1=(cl_class_var+cr_class_var)/class_var_all

                #子节点除以父节点
                left_ratio=len(left_targets) * 1.0 / (len(left_targets) + len(right_targets))
                right_ratio=len(right_targets) * 1.0 / (len(left_targets) + len(right_targets))
                lda_ratio=(left_ratio*cl_class_lda+right_ratio*cr_class_lda)/class_lda
                var_ratio=(left_ratio*cl_class_var+right_ratio*cr_class_var)/class_var
                keys=self.lda_ratio.keys()
                if depth in keys:
                    self.lda_ratio[depth].append(lda_ratio)
                    self.var_ratio[depth].append(var_ratio)
                else:
                    list1=[lda_ratio]
                    list2=[var_ratio]
                    self.lda_ratio[depth]=list1
                    self.var_ratio[depth]=list2
                tree.lda_ratio=lda_ratio
                tree.var_ratio=var_ratio
                tree.class_dist = class_dist
                tree.class_lda = class_lda
                tree.class_var = class_var
                return tree,class_lda,class_var,class_dist
        # 如果树的深度超过预设值，则终止分裂
        else:
            tree = Tree()
            tree.leaf_value = self.calc_leaf_value(targets['label'])
            tree.depth = depth
            tree.train_sample_num = len(dataset)
            self.num_node=self.num_node+1
            tree.class_dist=class_dist
            tree.class_lda=class_lda
            tree.class_var=class_var
            return tree,class_lda,class_var,class_dist

    def choose_best_feature(self, dataset, targets,depth):
        """寻找最好的数据集划分方式，找到最优分裂特征、分裂阈值、分裂增益"""
        best_split_gain = 1
        best_split_feature = None
        best_split_value = None
        feature_list=[]

        for feature in dataset.columns:
            if dataset[feature].unique().__len__() <= 1000:
                unique_values = sorted(dataset[feature].unique().tolist())
            # 如果该维度特征取值太多，则选择100个百分位值作为待选分裂阈值
            else:
                # unique_values=dataset[feature].to_numpy()
                unique_values = np.unique([np.percentile(dataset[feature], x)
                                           for x in np.linspace(0, 100, 100)])
            # #取中位数
            # if dataset[feature].unique().__len__() <= 100:
            #     unique_values = sorted(dataset[feature].unique().tolist())
            #     unique_values1=np.array(unique_values[0:-1])
            #     unique_values2=np.array(unique_values[1:])
            #     unique_values=(unique_values1+unique_values2)/2.0
            # # 如果该维度特征取值太多，则选择100个百分位值作为待选分裂阈值
            # else:
            #     unique_values = np.unique([np.percentile(dataset[feature], x)
            #                                for x in np.linspace(0, 100, 100)])
            #     unique_values=unique_values[1:-1]

            # 对可能的分裂阈值求分裂增益，选取增益最大的阈值
            for split_value in unique_values:
                # feature,split_value='pox',0.5
                left_targets = targets[dataset[feature] <= split_value]
                right_targets = targets[dataset[feature] > split_value]
                left_data = dataset[dataset[feature] <= split_value]
                right_data = dataset[dataset[feature] > split_value]
                if self.criterion=='gini_var':
                    split_gain,score = self.calc_gini_var(left_targets['label'], right_targets['label'],
                                                    left_data,right_data,w_=self.w_)
                    if depth<=0:
                        list1 = [(feature, split_value, score)]
                        self.feature_sort = self.feature_sort + list1
                    if score < best_split_gain:
                        best_split_feature = feature
                        best_split_value = split_value
                        best_split_gain = score
                elif self.criterion=='gini_class2':
                    split_gain,score = self.calc_gini_class2(left_targets['label'], right_targets['label'],
                                                    left_data,right_data,feature,split_value,
                                                  w_class=self.w_class,w_class2=self.w_class2)
                    if depth<=0:
                        list1 = [(feature, split_value, score)]
                        self.feature_sort = self.feature_sort + list1
                    if split_gain==0:
                        best_split_feature = feature
                        best_split_value = split_value
                        best_split_gain = score
                        break
                    if score < best_split_gain:
                        best_split_feature = feature
                        best_split_value = split_value
                        best_split_gain = score
                elif self.criterion=='gini_var_class':
                    split_gain,score = self.calc_gini_var_class(left_targets['label'], right_targets['label'],
                                                    left_data,right_data,feature,split_value,
                                                w_=self.w_,w_class=self.w_class,w_class2=self.w_class2)
                    if depth <= 0:
                        list1 = [(feature, split_value, score)]
                        self.feature_sort = self.feature_sort + list1
                    if score < best_split_gain:
                        best_split_feature = feature
                        best_split_value = split_value
                        best_split_gain = score
                elif self.criterion=='gini_var3':
                    split_gain,score = self.calc_gini_var3(left_targets['label'], right_targets['label'],
                                                    left_data,right_data,feature,split_value)
                    list1 = [(feature, split_value,split_gain, score)]
                    feature_list=feature_list+list1
                    if score < best_split_gain:
                        best_split_feature = feature
                        best_split_value = split_value
                        best_split_gain = score
                elif self.criterion=='gini_var3_minmax':
                    split_gain,score = self.calc_gini_var3_minmax(left_targets['label'], right_targets['label'],
                                                    left_data,right_data,feature,split_value)
                    list1 = [(feature, split_value,split_gain, score)]
                    feature_list=feature_list+list1
                    if score < best_split_gain:
                        best_split_feature = feature
                        best_split_value = split_value
                        best_split_gain = score
                elif self.criterion=='gini_var3_scaler':
                    split_gain,score = self.calc_gini_var3(left_targets['label'], right_targets['label'],
                                                    left_data,right_data,feature,split_value)
                    list1 = [(feature, split_value,split_gain, score)]
                    feature_list=feature_list+list1
                elif self.criterion=='gini_var_class_scaler':
                    split_gain,score_class,score_onlyvar = self.calc_gini_var_class_scaler(left_targets['label'], right_targets['label'],
                                                    left_data,right_data,feature,split_value,
                                                w_class=self.w_class,w_class2=self.w_class2)
                    list1 = [(feature, split_value, score_class, score_onlyvar)]
                    feature_list = feature_list + list1
                elif self.criterion=='gini_class2_var3':
                    split_gain,score = self.calc_gini_class3_no(left_targets['label'], right_targets['label'],
                                                    left_data,right_data,feature,split_value,
                                                  w_class=self.w_class,w_class2=self.w_class2)
                    list1 = [(feature, split_value, split_gain, score)]
                    feature_list = feature_list + list1
                    if depth<=0:
                        list1 =[(feature, split_value, split_gain, score)]
                        self.feature_sort = self.feature_sort + list1
                elif self.criterion=='gini_var_class_sort':
                    split_gain,score_class,score_onlyvar = self.calc_gini_var_class_no(left_targets['label'], right_targets['label'],
                                                    left_data,right_data,feature,split_value,
                                                w_class=self.w_class,w_class2=self.w_class2)
                    list1 = [(feature, split_value, split_gain,score_class, score_onlyvar)]
                    feature_list = feature_list + list1
                    if depth <= 0:
                        list1 = [(feature, split_value, score_class,score_onlyvar)]
                        self.feature_sort = self.feature_sort + list1
                    if score_class < best_split_gain:
                        best_split_feature = feature
                        best_split_value = split_value
                        best_split_gain = score_class
                elif self.criterion=='gini_var4':
                    split_gain,score = self.calc_gini_var4(left_targets['label'], right_targets['label'],
                                                    left_data,right_data,w_=self.w_,w_dist=self.w_dist)
                    if depth<=0:
                        list1 = [(feature, split_value,score)]
                        self.feature_sort = self.feature_sort + list1
                    if score < best_split_gain:
                        best_split_feature = feature
                        best_split_value = split_value
                        best_split_gain = score
                elif self.criterion == 'gini':
                    split_gain = self.calc_gini(left_targets['label'], right_targets['label'])
                    if depth<=0:
                        list1 = [(feature, split_value, split_gain)]
                        self.feature_sort = self.feature_sort + list1
                    if split_gain < best_split_gain:
                        best_split_feature = feature
                        best_split_value = split_value
                        best_split_gain = split_gain
                elif self.criterion == 'gini_random':
                    split_gain = self.calc_gini(left_targets['label'], right_targets['label'])
                    if depth<=0:
                        list1 = [(feature, split_value, split_gain)]
                        self.feature_sort = self.feature_sort + list1
                    if split_gain < best_split_gain:
                        best_split_feature = feature
                        best_split_value = split_value
                        best_split_gain = split_gain
                else:
                    split_gain = self.calc_gini(left_targets['label'], right_targets['label'])
                    if split_gain < best_split_gain:
                        best_split_feature = feature
                        best_split_value = split_value
                        best_split_gain = split_gain
                    # split_gain,score = self.calc_gini_var(left_targets['label'], right_targets['label'],
                    #                                 left_data,right_data)
                    # if score < best_split_gain:
                    #     best_split_feature = feature
                    #     best_split_value = split_value
                    #     best_split_gain = score
            else:
                continue
            break
        if self.criterion == 'gini_var3'or \
                self.criterion == 'gini_var3_minmax':
            ff=np.array(feature_list,dtype=[('feature','S40'),('value',float),
                                            ('split_gain',float),('score',float)])
            f1=np.sort(ff,order='split_gain')
            # f2=f1[range(self.rand_num)]
            # ll=f1.shape[0]
            if f1.shape[0]>=self.rand_num:
                f2=f1[range(self.rand_num)]
            else:
                f2=f1
            f3=np.sort(f2,order='score')
            # best_split_feature, best_split_value, best_split_gain,best_score = f3[-1]
            if self.issort_min==True:
                best_split_feature, best_split_value, best_split_gain,best_score = f3[0]
                index_min = np.argmin(f2['score'])
                self.sort_index.append(index_min)
                print('第' + str(depth) + '层，最佳分裂点为：' + str(f3[0]))
                print('候选值为：' + str(f2))
                print()
            else:
                best_split_feature, best_split_value, best_split_gain, best_score = f3[-1]
                index_max = np.argmax(f2['score'])
                self.sort_index.append(index_max)
                print('第' + str(depth) + '层，最佳分裂点为：' + str(f3[-1]))
                print('候选值为：' + str(f2))
                print()
            if f1[0]['split_gain']==0.0 and self.criterion=='gini_var3':
                best_split_feature, best_split_value, best_split_gain, best_score=f1[0]
            best_split_feature = str(best_split_feature, encoding='utf-8')

        if self.criterion=='gini_var_class_sort':
            ff=np.array(feature_list,dtype=[('feature','S40'),('value',float),
                                            ('split_gain',float),('score_class',float),('score',float)])
            f1=np.sort(ff,order='score_class')
            f4 = np.sort(ff, order='split_gain')
            # f2=f1[range(self.rand_num)]
            # ll=f1.shape[0]
            if f1.shape[0]>=self.rand_num:
                f2=f1[range(self.rand_num)]
            else:
                f2=f1
            f3=np.sort(f2,order='score')
            # best_split_feature, best_split_value, best_split_gain,best_score = f3[-1]
            if self.issort_min==True:
                best_split_feature, best_split_value,_, best_split_gain,best_score = f3[0]
                index_min = np.argmin(f2['score'])
                self.sort_index.append(index_min)
                print('第' + str(depth) + '层，最佳分裂点为：' + str(f3[0]))
                print('候选值为：' + str(f2))
                print()
            else:
                best_split_feature, best_split_value,_, best_split_gain, best_score = f3[-1]
                index_max = np.argmax(f2['score'])
                self.sort_index.append(index_max)
                print('第' + str(depth) + '层，最佳分裂点为：' + str(f3[-1]))
                print('候选值为：' + str(f2))
                print()
            if f4[0]['split_gain']==0.0:
                best_split_feature, best_split_value,_, best_split_gain, best_score=f4[0]
            best_split_feature = str(best_split_feature, encoding='utf-8')

        if self.criterion == 'gini_var3_scaler' or self.criterion == 'gini_var_class_scaler' :
            ff=np.array(feature_list,dtype=[('feature','S40'),('value',float),
                                            ('split_gain',float),('score',float)])
            scaler = MinMaxScaler()
            f1=ff['split_gain'].reshape(-1,1)
            f2 = ff['score'].reshape(-1,1)
            f1_s=scaler.fit_transform(f1)
            f2_s=scaler.fit_transform(f2)
            f3=f1_s+self.w_*f2_s
            index_min=np.argmin(f3)
            best_split_feature, best_split_value, best_split_gain,best_score = ff[index_min]
            best_split_feature = str(best_split_feature, encoding='utf-8')
        if self.criterion == 'gini_random':
            ff=np.array(self.feature_sort,dtype=[('feature','S40'),('value',float),('score',float)])
            f1=np.sort(ff,order='score')
            index=random.randint(0,self.rand_num)
            if depth<=0:
                index=0
            best_split_feature, best_split_value, best_split_gain=f1[index]
            best_split_feature=str(best_split_feature,encoding='utf-8')
        if depth<=0:
            self.root_best=[best_split_feature, best_split_value, best_split_gain]
        if depth==1:
            if self.sec_best1==None:
                self.sec_best1=[best_split_feature, best_split_value, best_split_gain]
            else:
                self.sec_best2=[best_split_feature, best_split_value, best_split_gain]
        return best_split_feature, best_split_value, best_split_gain

    @staticmethod
    def calc_class_lda2(left_targets, right_targets,left_data,right_data):#减少小簇偏向
        """分类树采用基尼指数作为指标来选择最优分裂点"""
        left_targets = left_targets.values.flatten()
        right_targets = right_targets.values.flatten()
        left_data = left_data.values
        right_data = right_data.values
        scaler = MinMaxScaler()
        if len(left_data) != 0 and len(right_data) != 0:
            data = np.concatenate((left_data, right_data), axis=0)
            scaler.fit(data)
            left_data = scaler.transform(left_data)
            right_data = scaler.transform(right_data)
            # data=scaler.transform(data)
            # all_data = np.concatenate((left_targets, right_targets), axis=0)
        elif len(left_data) != 0:
            left_data = scaler.fit_transform(left_data)
            # data=left_data
            # all_data=left_targets
        elif len(left_data) != 0:
            right_data = scaler.fit_transform(right_data)
            # data=right_data
            # all_data=right_targets
        else:
            return None
        left_label_class = np.unique(left_targets)
        right_label_class = np.unique(right_targets)

        # all_class = np.unique(all_data)
        # all_class = np.array(all_class, dtype=int)
        # var_all = 0
        # dist_all = 0
        # aver_all = np.zeros((2, data.shape[1]))
        # for j in all_class:
        #     index_ = np.where(all_data == j)
        #     set_ = np.array(data[index_])
        #     v = np.var(set_, axis=0)
        #     v1 = np.sum(v) / len(v)
        #     var_all = var_all + v1
        #     aver_all[j] = np.mean(set_, axis=0)
        # if len(all_class) == 2:
        #     d = aver_all[0] - aver_all[1]
        #     dist_all = np.dot(d, d.T)
        # if dist_all == 0:
        #     score = (var_all)
        # else:
        #     score = (var_all) / (dist_all)

        left_label_class = np.array(left_label_class, dtype=int)
        right_label_class = np.array(right_label_class, dtype=int)
        var_all_left = 0
        var_all_right = 0
        dist_l = 0
        dist_r = 0
        aver_l = np.zeros((2, left_data.shape[1]))
        aver_r = np.zeros((2, right_data.shape[1]))
        for i in left_label_class:
            index_ = np.where(left_targets == i)
            set_ = np.array(left_data[index_])
            v = np.var(set_, axis=0)
            v1 = np.sum(v) / len(v)
            var_all_left = var_all_left + v1
            aver_l[i] = np.mean(set_, axis=0)
        for j in right_label_class:
            index_ = np.where(right_targets == j)
            set_ = np.array(right_data[index_])
            v = np.var(set_, axis=0)
            v1 = np.sum(v) / len(v)
            var_all_right = var_all_right + v1
            aver_r[j] = np.mean(set_, axis=0)
        if len(left_label_class) == 2:
            d = aver_l[0] - aver_l[1]
            dist_l = np.dot(d, d.T)/left_data.shape[1]
        if len(right_label_class) == 2:
            dd = aver_r[0] - aver_r[1]
            dist_r = np.dot(dd, dd.T)/right_data.shape[1]
        if dist_l == 0:
            left_score = (var_all_left)
        else:
            left_score = (var_all_left) / (dist_l)

        if dist_r == 0:
            right_score = (var_all_right)
        else:
            right_score = (var_all_right) / (dist_r)
        return left_score, var_all_left, dist_l,right_score,var_all_right,dist_r
            # ,score,var_all,dist_all

    @staticmethod
    def calc_class_lda(left_data, left_targets):
        # return 1,1,1
        left_targets = left_targets.values.flatten()
        left_data = left_data.values
        scaler = MinMaxScaler()
        left_data = scaler.fit_transform(left_data)
        left_label_class = np.unique(left_targets)
        left_label_class = np.array(left_label_class, dtype=int)
        var_all_left = 0
        dist_l = 0
        aver_l = np.zeros((2, left_data.shape[1]))
        for i in left_label_class:
            index_ = np.where(left_targets == i)
            # aa=left_data[index_,:]
            set_ = np.array(left_data[index_])
            v= np.var(set_,axis=0)
            v1=np.sum(v)/len(v)
            var_all_left = var_all_left + v1
            aver_l[i] = np.mean(set_, axis=0)
        if len(left_label_class) == 2:
            d = aver_l[0] - aver_l[1]
            dist_l = np.dot(d, d.T)/left_data.shape[1]
        if dist_l == 0:
            left_score = (var_all_left)
        else:
            left_score = (var_all_left) / (dist_l)
        score =left_score
        return score,var_all_left,dist_l

    @staticmethod
    def calc_leaf_value(targets):
        """选择样本中出现次数最多的类别作为叶子节点取值"""
        label_counts = collections.Counter(targets)
        major_label = max(zip(label_counts.values(), label_counts.keys()))
        return major_label[1]

    @staticmethod
    def calc_gini(left_targets, right_targets):
        """分类树采用基尼指数作为指标来选择最优分裂点"""
        split_gain = 0
        for targets in [left_targets, right_targets]:
            gini = 1
            # 统计每个类别有多少样本，然后计算gini
            label_counts = collections.Counter(targets)
            for key in label_counts:
                prob = label_counts[key] * 1.0 / len(targets)
                gini -= prob ** 2
            split_gain += len(targets) * 1.0 / (len(left_targets) + len(right_targets)) * gini
        return split_gain

    @staticmethod
    def calc_var_class(left_targets, right_targets,left_data,right_data,
                            feature,split_value,w_class,w_,w_class2):#减少小簇偏向
        """分类树采用基尼指数作为指标来选择最优分裂点"""
        w=w_
        w1=1
        w2=w_
        w3=w_class2
        gini_left=1
        gini_right=1
        left_targets=left_targets.values
        right_targets = right_targets.values
        left_data = left_data.values
        right_data = right_data.values
        scaler = MinMaxScaler()
        if len(left_data)!=0:
            left_data = scaler.fit_transform(left_data)
        if len(right_data) != 0:
            right_data = scaler.fit_transform(right_data)
        left_label_class = np.unique(left_targets)
        right_label_class = np.unique(right_targets)
        var_all_left = 0
        var_all_right = 0
        for i in left_label_class:
            index_ = np.where(left_targets == i)
            set_ = np.array(left_data[index_])
            v= np.var(set_,axis=0)
            v1=np.sum(v)/len(v)
            var_all_left=var_all_left+v1
        for j in right_label_class:
            index_ = np.where(right_targets == j)
            set_ = np.array(right_data[index_])
            v= np.var(set_,axis=0)
            v1 = np.sum(v) / len(v)
            var_all_right=var_all_right+v1
        p1=len(left_label_class)/2
        p2 = len(right_label_class) / 2
        # p1=1
        # p2=1
        left_score=w1*gini_left+w*var_all_left*p1
        right_score = w1*gini_right + w * var_all_right*p2
        CSN_score=len(left_targets) * 1.0 / (len(left_targets) + len(right_targets)) * left_score\
                     +len(right_targets) * 1.0 / (len(left_targets) + len(right_targets)) * right_score

        left_item=left_data[feature].values.reshape(-1,1)
        right_item=right_data[feature].values.reshape(-1,1)
        all_item=np.concatenate((left_item,right_item),axis=0)
        max_item=np.max(all_item)
        min_item = np.min(all_item)
        # split_value=np.array(split_value).reshape(-1,1)
        left_targets=left_targets.values
        right_targets = right_targets.values
        left_data = left_data.values
        right_data = right_data.values
        scaler = MinMaxScaler()
        if len(left_data)!=0 and len(right_data)!= 0 :
            data = np.concatenate((left_data, right_data), axis=0)
            scaler.fit(data)
            left_data = scaler.transform(left_data)
            right_data = scaler.transform(right_data)
        elif len(left_data)!=0:
            left_data = scaler.fit_transform(left_data)
        elif len(left_data) != 0:
            right_data = scaler.fit_transform(right_data)
        left_label_class = np.unique(left_targets)
        right_label_class = np.unique(right_targets)
        var_all_left = 0
        var_all_right = 0
        for i in left_label_class:
            index_ = np.where(left_targets == i)
            set_ = np.array(left_data[index_])
            v= np.var(set_,axis=0)
            v1=np.sum(v)/len(v)
            var_all_left=var_all_left+v1
        for j in right_label_class:
            index_ = np.where(right_targets == j)
            set_ = np.array(right_data[index_])
            v= np.var(set_,axis=0)
            v1 = np.sum(v) / len(v)
            var_all_right=var_all_right+v1
        p1=len(left_label_class)/2
        p2 = len(right_label_class) / 2
        left_score=w1*gini_left+w2*var_all_left*p1
        right_score = w1*gini_right + w2* var_all_right*p2
        score1 = len(left_targets) * 1.0 / (len(left_targets) + len(right_targets)) * left_score \
                + len(right_targets) * 1.0 / (len(left_targets) + len(right_targets)) * right_score

        all_data=np.concatenate((left_targets,right_targets),axis=0)
        all_class = np.unique(all_data)
        all_class=np.array(all_class,dtype=int)
        all_class_num=np.max(np.array(all_class,dtype=int))+1
        left_mean=np.zeros((all_class_num,left_data.shape[1]))
        right_mean = np.zeros((all_class_num, right_data.shape[1]))
        dist=np.zeros((all_class_num,2))
        num_p=np.zeros((all_class_num,2))#记录左右两节点分别每类的个数（左，右）第一列为左节点，第二列为右节点
        for i in all_class:
            index_ = np.where(left_targets == i)
            set_ = np.array(left_data[index_])
            if len(set_)!=0:
                if max_item - min_item==0:
                    dist1 = np.min(np.absolute(left_item[index_] - split_value))
                else:
                    dist1 = np.min(np.absolute(left_item[index_] - split_value) / (max_item - min_item))
                dist[i, 0] = dist1
                left_mean[i] = np.mean(set_, axis=0)
                num_p[i,0]=set_.shape[0]
        for j in all_class:
            index_ = np.where(right_targets == j)
            set_ = np.array(right_data[index_])
            if len(set_) != 0:
                if max_item - min_item==0:
                    dist1 = np.min(np.absolute(right_item[index_] - split_value))
                else:
                    dist1 = np.min(np.absolute(right_item[index_] - split_value) / (max_item - min_item))
                dist[j, 1] = dist1
                right_mean[j] = np.mean(set_, axis=0)
                num_p[j, 1] = set_.shape[0]
        class_dist_l_r = 0
        is_same_class = 0
        dist_p=0
        dist_num=0
        for p in all_class:
            if (num_p[p,0]!=0).all() and (num_p[0-p+1,1]!=0).all():
                dist_p=dist[p,0]+dist[0-p+1,1]+dist_p
                dist_num=dist_num+1
            if (num_p[p,0]!=0).all() and (num_p[p,1]!=0).all():
                vector1 = left_mean[p]
                vector2 = right_mean[p]
                # w2 = np.absolute(num_p[p, 0] - num_p[p, 1]) / np.sum(num_p[p, 0] + num_p[p, 1])
                w2 = 1
                class_dist_l_r = np.sqrt(np.sum(np.square(vector1 - vector2))) * w2
                is_same_class = is_same_class + 1
        if is_same_class==0:
            is_same_class=1
        if dist_num == 0:
            dist_num = 1
        class_dist_l_r = class_dist_l_r /is_same_class
        # class_dist_l_r = class_dist_l_r / is_same_class
        BNM_score = w1 * score1 - w * class_dist_l_r+w3*dist_p/dist_num
        score=CSN_score-w_*BNM_score
        return BNM_score, CSN_score,score

    @staticmethod
    def calc_gini_var(left_targets, right_targets,left_data,right_data,w_):
        """分类树采用基尼指数作为指标来选择最优分裂点"""
        split_gain = 0
        w=w_
        w1=1
        gini_left=1
        gini_right=1
        label_counts_left = collections.Counter(left_targets)
        for key in label_counts_left:
            prob = label_counts_left[key] * 1.0 / len(left_targets)
            gini_left -= prob ** 2
        label_counts_right = collections.Counter(right_targets)
        for key in label_counts_right:
            prob = label_counts_right[key] * 1.0 / len(right_targets)
            gini_right -= prob ** 2
        split_gain = len(left_targets) * 1.0 / (len(left_targets) + len(right_targets)) * gini_left\
                     +len(right_targets) * 1.0 / (len(left_targets) + len(right_targets)) * gini_right
        # for targets in [left_targets, right_targets]:
        #     gini = 1
        #     # 统计每个类别有多少样本，然后计算gini
        #     label_counts = collections.Counter(targets)
        #     for key in label_counts:
        #         prob = label_counts[key] * 1.0 / len(targets)
        #         gini -= prob ** 2
        #     split_gain += len(targets) * 1.0 / (len(left_targets) + len(right_targets)) * gini
        left_targets=left_targets.values
        right_targets = right_targets.values
        left_data = left_data.values
        right_data = right_data.values
        scaler = MinMaxScaler()
        if len(left_data)!=0:
            left_data = scaler.fit_transform(left_data)
        if len(right_data) != 0:
            right_data = scaler.fit_transform(right_data)
        left_label_class = np.unique(left_targets)
        right_label_class = np.unique(right_targets)
        var_all_left = 0
        var_all_right = 0
        for i in left_label_class:
            index_ = np.where(left_targets == i)
            set_ = np.array(left_data[index_])
            v= np.var(set_,axis=0)
            v1=np.sum(v)/len(v)
            var_all_left=var_all_left+v1
        for j in right_label_class:
            index_ = np.where(right_targets == j)
            set_ = np.array(right_data[index_])
            v= np.var(set_,axis=0)
            v1 = np.sum(v) / len(v)
            var_all_right=var_all_right+v1
        p1=len(left_label_class)/2
        p2 = len(right_label_class) / 2
        # p1=1
        # p2=1
        left_score=w1*gini_left+w*var_all_left*p1
        right_score = w1*gini_right + w * var_all_right*p2
        score=len(left_targets) * 1.0 / (len(left_targets) + len(right_targets)) * left_score\
                     +len(right_targets) * 1.0 / (len(left_targets) + len(right_targets)) * right_score
        return split_gain,score

    @staticmethod
    def calc_gini_var4(left_targets, right_targets,left_data,right_data,w_,w_dist):#对方差加上类间距的影响
        """分类树采用基尼指数作为指标来选择最优分裂点"""
        #二分类问题
        split_gain = 0
        w=w_
        w1=1
        w4=w_dist
        gini_left=1
        gini_right=1
        label_counts_left = collections.Counter(left_targets)
        for key in label_counts_left:
            prob = label_counts_left[key] * 1.0 / len(left_targets)
            gini_left -= prob ** 2
        label_counts_right = collections.Counter(right_targets)
        for key in label_counts_right:
            prob = label_counts_right[key] * 1.0 / len(right_targets)
            gini_right -= prob ** 2
        split_gain = len(left_targets) * 1.0 / (len(left_targets) + len(right_targets)) * gini_left\
                     +len(right_targets) * 1.0 / (len(left_targets) + len(right_targets)) * gini_right
        left_targets = left_targets.values
        right_targets = right_targets.values
        left_data = left_data.values
        right_data = right_data.values
        scaler = MinMaxScaler()
        if len(left_data) != 0:
            left_data = scaler.fit_transform(left_data)
        if len(right_data) != 0:
            right_data = scaler.fit_transform(right_data)
        left_label_class = np.unique(left_targets)
        right_label_class = np.unique(right_targets)
        left_label_class=np.array(left_label_class,dtype=int)
        right_label_class=np.array(right_label_class,dtype=int)
        var_all_left = 0
        var_all_right = 0
        dist_l=0
        dist_r=0
        aver_l = np.zeros((2,left_data.shape[1]))
        aver_r = np.zeros((2, right_data.shape[1]))
        for i in left_label_class:
            index_ = np.where(left_targets == i)
            set_ = np.array(left_data[index_])
            v = np.var(set_, axis=0)
            v1 = np.sum(v) / len(v)
            var_all_left = var_all_left + v1
            aver_l[i]=np.mean(set_,axis=0)
        for j in right_label_class:
            index_ = np.where(right_targets == j)
            set_ = np.array(right_data[index_])
            v = np.var(set_, axis=0)
            v1 = np.sum(v) / len(v)
            var_all_right = var_all_right + v1
            aver_r[j] = np.mean(set_, axis=0)
        if len(left_label_class) == 2:
            d=aver_l[0]-aver_l[1]
            dist_l=np.dot(d,d.T)
        if len(right_label_class) == 2:
            dd=aver_r[0]-aver_r[1]
            dist_r=np.dot(dd,dd.T)
        p1 = len(left_label_class) / 2
        p2 = len(right_label_class) / 2
        # p1=1
        # p2=1
        left_score = w1 * gini_left + w * var_all_left * p1-w4*dist_l
        right_score = w1 * gini_right + w * var_all_right * p2-w4*dist_r
        score = len(left_targets) * 1.0 / (len(left_targets) + len(right_targets)) * left_score \
                + len(right_targets) * 1.0 / (len(left_targets) + len(right_targets)) * right_score
        return split_gain, score

    @staticmethod
    def calc_gini_var_class(left_targets, right_targets,left_data,right_data,
                            feature,split_value,w_class,w_,w_class2):#减少小簇偏向
        """分类树采用基尼指数作为指标来选择最优分裂点"""
        split_gain = 0
        w=w_class
        w1=1
        w2=w_
        w3=w_class2
        gini_left=1
        gini_right=1
        label_counts_left = collections.Counter(left_targets)
        for key in label_counts_left:
            prob = label_counts_left[key] * 1.0 / len(left_targets)
            gini_left -= prob ** 2
        label_counts_right = collections.Counter(right_targets)
        for key in label_counts_right:
            prob = label_counts_right[key] * 1.0 / len(right_targets)
            gini_right -= prob ** 2
        split_gain = len(left_targets) * 1.0 / (len(left_targets) + len(right_targets)) * gini_left\
                     +len(right_targets) * 1.0 / (len(left_targets) + len(right_targets)) * gini_right
        left_item=left_data[feature].values.reshape(-1,1)
        right_item=right_data[feature].values.reshape(-1,1)
        all_item=np.concatenate((left_item,right_item),axis=0)
        max_item=np.max(all_item)
        min_item = np.min(all_item)
        # split_value=np.array(split_value).reshape(-1,1)
        left_targets=left_targets.values
        right_targets = right_targets.values
        left_data = left_data.values
        right_data = right_data.values
        scaler = MinMaxScaler()
        if len(left_data)!=0 and len(right_data)!= 0 :
            data = np.concatenate((left_data, right_data), axis=0)
            scaler.fit(data)
            left_data = scaler.transform(left_data)
            right_data = scaler.transform(right_data)
        elif len(left_data)!=0:
            left_data = scaler.fit_transform(left_data)
        elif len(left_data) != 0:
            right_data = scaler.fit_transform(right_data)
        else:
            score =split_gain
            return split_gain, score

        left_label_class = np.unique(left_targets)
        right_label_class = np.unique(right_targets)
        var_all_left = 0
        var_all_right = 0
        for i in left_label_class:
            index_ = np.where(left_targets == i)
            set_ = np.array(left_data[index_])
            v= np.var(set_,axis=0)
            v1=np.sum(v)/len(v)
            var_all_left=var_all_left+v1
        for j in right_label_class:
            index_ = np.where(right_targets == j)
            set_ = np.array(right_data[index_])
            v= np.var(set_,axis=0)
            v1 = np.sum(v) / len(v)
            var_all_right=var_all_right+v1
        p1=len(left_label_class)/2
        p2 = len(right_label_class) / 2
        left_score=w1*gini_left+w2*var_all_left*p1
        right_score = w1*gini_right + w2* var_all_right*p2
        score1 = len(left_targets) * 1.0 / (len(left_targets) + len(right_targets)) * left_score \
                + len(right_targets) * 1.0 / (len(left_targets) + len(right_targets)) * right_score

        all_data=np.concatenate((left_targets,right_targets),axis=0)
        all_class = np.unique(all_data)
        all_class=np.array(all_class,dtype=int)
        all_class_num=np.max(np.array(all_class,dtype=int))+1
        left_mean=np.zeros((all_class_num,left_data.shape[1]))
        right_mean = np.zeros((all_class_num, right_data.shape[1]))
        dist=np.zeros((all_class_num,2))
        num_p=np.zeros((all_class_num,2))#记录左右两节点分别每类的个数（左，右）第一列为左节点，第二列为右节点
        for i in all_class:
            index_ = np.where(left_targets == i)
            set_ = np.array(left_data[index_])
            if len(set_)!=0:
                if max_item - min_item==0:
                    dist1 = np.min(np.absolute(left_item[index_] - split_value))
                else:
                    dist1 = np.min(np.absolute(left_item[index_] - split_value) / (max_item - min_item))
                dist[i, 0] = dist1
                left_mean[i] = np.mean(set_, axis=0)
                num_p[i,0]=set_.shape[0]
        for j in all_class:
            index_ = np.where(right_targets == j)
            set_ = np.array(right_data[index_])
            if len(set_) != 0:
                if max_item - min_item==0:
                    dist1 = np.min(np.absolute(right_item[index_] - split_value))
                else:
                    dist1 = np.min(np.absolute(right_item[index_] - split_value) / (max_item - min_item))
                dist[j, 1] = dist1
                right_mean[j] = np.mean(set_, axis=0)
                num_p[j, 1] = set_.shape[0]
        class_dist_l_r = 0
        is_same_class = 0
        dist_p=0
        dist_num=0
        for p in all_class:
            if (num_p[p,0]!=0).all() and (num_p[0-p+1,1]!=0).all():
                dist_p=dist[p,0]+dist[0-p+1,1]+dist_p
                dist_num=dist_num+1
            if (num_p[p,0]!=0).all() and (num_p[p,1]!=0).all():
                vector1 = left_mean[p]
                vector2 = right_mean[p]
                # w2 = np.absolute(num_p[p, 0] - num_p[p, 1]) / np.sum(num_p[p, 0] + num_p[p, 1])
                w2 = 1
                class_dist_l_r = np.sqrt(np.sum(np.square(vector1 - vector2))) * w2
                is_same_class = is_same_class + 1
        if is_same_class==0:
            is_same_class=1
        if dist_num == 0:
            dist_num = 1
        class_dist_l_r = class_dist_l_r /is_same_class
        # class_dist_l_r = class_dist_l_r / is_same_class
        score = w1 * score1 - w * class_dist_l_r+w3*dist_p/dist_num
        return split_gain, score

    @staticmethod
    def calc_gini_class2(left_targets, right_targets,left_data,right_data,
                         feature,split_value,w_class,w_class2):#减少小簇偏向
        """分类树采用基尼指数作为指标来选择最优分裂点"""
        split_gain = 0
        w=w_class
        w1=1
        w3=w_class2
        gini_left=1
        gini_right=1
        label_counts_left = collections.Counter(left_targets)
        for key in label_counts_left:
            prob = label_counts_left[key] * 1.0 / len(left_targets)
            gini_left -= prob ** 2
        label_counts_right = collections.Counter(right_targets)
        for key in label_counts_right:
            prob = label_counts_right[key] * 1.0 / len(right_targets)
            gini_right -= prob ** 2
        split_gain = len(left_targets) * 1.0 / (len(left_targets) + len(right_targets)) * gini_left\
                     +len(right_targets) * 1.0 / (len(left_targets) + len(right_targets)) * gini_right
        left_item=left_data[feature].values.reshape(-1,1)
        right_item=right_data[feature].values.reshape(-1,1)
        all_item=np.concatenate((left_item,right_item),axis=0)
        max_item=np.max(all_item)
        min_item = np.min(all_item)
        # split_value=np.array(split_value).reshape(-1,1)
        left_targets=left_targets.values
        right_targets = right_targets.values
        left_data = left_data.values
        right_data = right_data.values
        scaler = MinMaxScaler()
        if len(left_data)!=0 and len(right_data) != 0 :
            data = np.concatenate((left_data, right_data), axis=0)
            scaler.fit(data)
            left_data = scaler.transform(left_data)
            right_data = scaler.transform(right_data)
        else:
            score =split_gain
            return split_gain, score
        all_data=np.concatenate((left_targets,right_targets),axis=0)
        all_class = np.unique(all_data)
        all_class=np.array(all_class,dtype=int)
        all_class_num=np.max(np.array(all_class,dtype=int))+1
        left_mean=np.zeros((all_class_num,left_data.shape[1]))
        right_mean = np.zeros((all_class_num, right_data.shape[1]))
        dist=np.zeros((all_class_num,2))
        num_p=np.zeros((all_class_num,2))#记录左右两节点分别每类的个数（左，右）第一列为左节点，第二列为右节点
        for i in all_class:
            index_ = np.where(left_targets == i)
            set_ = np.array(left_data[index_])
            if len(set_)!=0:
                dist1 = np.min(np.absolute(left_item[index_] - split_value) / (max_item - min_item))
                dist[i, 0] = dist1
                left_mean[i] = np.mean(set_, axis=0)
                num_p[i,0]=set_.shape[0]
        for j in all_class:
            index_ = np.where(right_targets == j)
            set_ = np.array(right_data[index_])
            if len(set_) != 0:
                dist1 = np.min(np.absolute(right_item[index_] - split_value) / (max_item - min_item))
                dist[j, 1] = dist1
                right_mean[j] = np.mean(set_, axis=0)
                num_p[j, 1] = set_.shape[0]
        class_dist_l_r = 0
        is_same_class = 0
        dist_p=0
        dist_num=0
        for p in all_class:
            if (num_p[p,0]!=0).all() and (num_p[0-p+1,1]!=0).all():
                dist_p=dist[p,0]+dist[0-p+1,1]+dist_p
                dist_num=dist_num+1
            if (num_p[p,0]!=0).all() and (num_p[p,1]!=0).all():
                vector1 = left_mean[p]
                vector2 = right_mean[p]
                # w2 = np.absolute(num_p[p, 0] - num_p[p, 1]) / np.sum(num_p[p, 0] + num_p[p, 1])
                w2 = 1
                class_dist_l_r = np.sqrt(np.sum(np.square(vector1 - vector2))) * w2
                is_same_class = is_same_class + 1
        if is_same_class==0:
            is_same_class=1
        if dist_num == 0:
            dist_num = 1
        class_dist_l_r = class_dist_l_r /is_same_class
        # class_dist_l_r = class_dist_l_r / is_same_class
        score = w1 * split_gain - w * class_dist_l_r+w3*dist_p/dist_num
        return split_gain, score


    @staticmethod
    def calc_gini_class3_no(left_targets, right_targets,left_data,right_data,
                         feature,split_value,w_class,w_class2):#减少小簇偏向
        """分类树采用基尼指数作为指标来选择最优分裂点"""
        split_gain = 0
        w=1
        w1=0
        w3=1
        gini_left=1
        gini_right=1
        label_counts_left = collections.Counter(left_targets)
        for key in label_counts_left:
            prob = label_counts_left[key] * 1.0 / len(left_targets)
            gini_left -= prob ** 2
        label_counts_right = collections.Counter(right_targets)
        for key in label_counts_right:
            prob = label_counts_right[key] * 1.0 / len(right_targets)
            gini_right -= prob ** 2
        split_gain = len(left_targets) * 1.0 / (len(left_targets) + len(right_targets)) * gini_left\
                     +len(right_targets) * 1.0 / (len(left_targets) + len(right_targets)) * gini_right
        left_item=left_data[feature].values.reshape(-1,1)
        right_item=right_data[feature].values.reshape(-1,1)
        all_item=np.concatenate((left_item,right_item),axis=0)
        max_item=np.max(all_item)
        min_item = np.min(all_item)
        # split_value=np.array(split_value).reshape(-1,1)
        left_targets=left_targets.values
        right_targets = right_targets.values
        left_data = left_data.values
        right_data = right_data.values
        scaler = MinMaxScaler()
        if len(left_data)!=0 and len(right_data) != 0 :
            data = np.concatenate((left_data, right_data), axis=0)
            scaler.fit(data)
            left_data = scaler.transform(left_data)
            right_data = scaler.transform(right_data)
        else:
            score =split_gain
            return split_gain, score
        all_data=np.concatenate((left_targets,right_targets),axis=0)
        all_class = np.unique(all_data)
        all_class=np.array(all_class,dtype=int)
        all_class_num=np.max(np.array(all_class,dtype=int))+1
        left_mean=np.zeros((all_class_num,left_data.shape[1]))
        right_mean = np.zeros((all_class_num, right_data.shape[1]))
        dist=np.zeros((all_class_num,2))
        num_p=np.zeros((all_class_num,2))#记录左右两节点分别每类的个数（左，右）第一列为左节点，第二列为右节点
        for i in all_class:
            index_ = np.where(left_targets == i)
            set_ = np.array(left_data[index_])
            if len(set_)!=0:
                dist1 = np.min(np.absolute(left_item[index_] - split_value) / (max_item - min_item))
                dist[i, 0] = dist1
                left_mean[i] = np.mean(set_, axis=0)
                num_p[i,0]=set_.shape[0]
        for j in all_class:
            index_ = np.where(right_targets == j)
            set_ = np.array(right_data[index_])
            if len(set_) != 0:
                dist1 = np.min(np.absolute(right_item[index_] - split_value) / (max_item - min_item))
                dist[j, 1] = dist1
                right_mean[j] = np.mean(set_, axis=0)
                num_p[j, 1] = set_.shape[0]
        class_dist_l_r = 0
        is_same_class = 0
        dist_p=0
        dist_num=0
        for p in all_class:
            if (num_p[p,0]!=0).all() and (num_p[0-p+1,1]!=0).all():
                dist_p=dist[p,0]+dist[0-p+1,1]+dist_p
                dist_num=dist_num+1
            if (num_p[p,0]!=0).all() and (num_p[p,1]!=0).all():
                vector1 = left_mean[p]
                vector2 = right_mean[p]
                # w2 = np.absolute(num_p[p, 0] - num_p[p, 1]) / np.sum(num_p[p, 0] + num_p[p, 1])
                w2 = 1
                class_dist_l_r = np.sqrt(np.sum(np.square(vector1 - vector2))) * w2
                is_same_class = is_same_class + 1
        if is_same_class==0:
            is_same_class=1
        if dist_num == 0:
            dist_num = 1
        class_dist_l_r = class_dist_l_r /is_same_class
        # class_dist_l_r = class_dist_l_r / is_same_class
        score = w1 * split_gain - w * class_dist_l_r+w3*dist_p/dist_num
        return split_gain, score

    @staticmethod
    def calc_gini_var3(left_targets, right_targets,left_data,right_data,w_,w_dist):#减少小簇偏向
        """分类树采用基尼指数作为指标来选择最优分裂点"""
        #二分类问题
        split_gain = 0
        w=w_
        w1=0
        w4=w_dist
        gini_left=1
        gini_right=1
        label_counts_left = collections.Counter(left_targets)
        for key in label_counts_left:
            prob = label_counts_left[key] * 1.0 / len(left_targets)
            gini_left -= prob ** 2
        label_counts_right = collections.Counter(right_targets)
        for key in label_counts_right:
            prob = label_counts_right[key] * 1.0 / len(right_targets)
            gini_right -= prob ** 2
        split_gain = len(left_targets) * 1.0 / (len(left_targets) + len(right_targets)) * gini_left\
                     +len(right_targets) * 1.0 / (len(left_targets) + len(right_targets)) * gini_right
        left_targets = left_targets.values
        right_targets = right_targets.values
        left_data = left_data.values
        right_data = right_data.values
        scaler = MinMaxScaler()
        if len(left_data) != 0:
            left_data = scaler.fit_transform(left_data)
        if len(right_data) != 0:
            right_data = scaler.fit_transform(right_data)
        left_label_class = np.unique(left_targets)
        right_label_class = np.unique(right_targets)
        left_label_class=np.array(left_label_class,dtype=int)
        right_label_class=np.array(right_label_class,dtype=int)
        var_all_left = 0
        var_all_right = 0
        dist_l=0
        dist_r=0
        aver_l = np.zeros((2,left_data.shape[1]))
        aver_r = np.zeros((2, right_data.shape[1]))
        for i in left_label_class:
            index_ = np.where(left_targets == i)
            set_ = np.array(left_data[index_])
            v = np.var(set_, axis=0)
            v1 = np.sum(v) / len(v)
            # if len(set_)==1:
            #     v1=0.00001
            var_all_left = var_all_left + v1
            aver_l[i]=np.mean(set_,axis=0)
        for j in right_label_class:
            index_ = np.where(right_targets == j)
            set_ = np.array(right_data[index_])
            v = np.var(set_, axis=0)
            v1 = np.sum(v) / len(v)
            var_all_right = var_all_right + v1
            aver_r[j] = np.mean(set_, axis=0)
        if len(left_label_class) == 2:
            d=aver_l[0]-aver_l[1]
            dist_l=np.dot(d,d.T)
        if len(right_label_class) == 2:
            dd=aver_r[0]-aver_r[1]
            dist_r=np.dot(dd,dd.T)
        # p1 = len(left_label_class) / 2
        # p2 = len(right_label_class) / 2
        p1=1
        p2=1
        if dist_l==0:#说明此时纯度最大
            # left_score = (var_all_left)
            left_score = 0
        else:
            left_score = (var_all_left)/(dist_l)

        if dist_r==0:
            # right_score = (var_all_right)
            right_score =0

        else:
            right_score = (var_all_right)/(dist_r)
        score = len(left_targets) * 1.0 / (len(left_targets) + len(right_targets)) * left_score \
                + len(right_targets) * 1.0 / (len(left_targets) + len(right_targets)) * right_score
        return split_gain, score

    @staticmethod
    def calc_gini_var3_minmax(left_targets, right_targets,left_data,right_data,w_,w_dist):#减少小簇偏向
        """分类树采用基尼指数作为指标来选择最优分裂点"""
        #二分类问题
        split_gain = 0
        w=w_
        w1=0
        w4=w_dist
        gini_left=1
        gini_right=1
        label_counts_left = collections.Counter(left_targets)
        for key in label_counts_left:
            prob = label_counts_left[key] * 1.0 / len(left_targets)
            gini_left -= prob ** 2
        label_counts_right = collections.Counter(right_targets)
        for key in label_counts_right:
            prob = label_counts_right[key] * 1.0 / len(right_targets)
            gini_right -= prob ** 2
        split_gain = len(left_targets) * 1.0 / (len(left_targets) + len(right_targets)) * gini_left\
                     +len(right_targets) * 1.0 / (len(left_targets) + len(right_targets)) * gini_right
        left_targets = left_targets.values
        right_targets = right_targets.values
        left_data = left_data.values
        right_data = right_data.values
        scaler = MinMaxScaler()
        if len(left_data)!=0 and len(right_data)!= 0 :
            data = np.concatenate((left_data, right_data), axis=0)
            scaler.fit(data)
            left_data = scaler.transform(left_data)
            right_data = scaler.transform(right_data)
        elif len(left_data)!=0:
            left_data = scaler.fit_transform(left_data)
        elif len(left_data) != 0:
            right_data = scaler.fit_transform(right_data)
        else:
            score =split_gain
            return split_gain, score
        left_label_class = np.unique(left_targets)
        right_label_class = np.unique(right_targets)
        left_label_class=np.array(left_label_class,dtype=int)
        right_label_class=np.array(right_label_class,dtype=int)
        var_all_left = 0
        var_all_right = 0
        dist_l=0
        dist_r=0
        aver_l = np.zeros((2,left_data.shape[1]))
        aver_r = np.zeros((2, right_data.shape[1]))
        for i in left_label_class:
            index_ = np.where(left_targets == i)
            set_ = np.array(left_data[index_])
            v = np.var(set_, axis=0)
            v1 = np.sum(v) / len(v)
            var_all_left = var_all_left + v1
            aver_l[i]=np.mean(set_,axis=0)
        for j in right_label_class:
            index_ = np.where(right_targets == j)
            set_ = np.array(right_data[index_])
            v = np.var(set_, axis=0)
            v1 = np.sum(v) / len(v)
            var_all_right = var_all_right + v1
            aver_r[j] = np.mean(set_, axis=0)
        if len(left_label_class) == 2:
            d=aver_l[0]-aver_l[1]
            dist_l=np.dot(d,d.T)
        if len(right_label_class) == 2:
            dd=aver_r[0]-aver_r[1]
            dist_r=np.dot(dd,dd.T)
        # p1 = len(left_label_class) / 2
        # p2 = len(right_label_class) / 2
        p1=1
        p2=1
        if dist_l==0:
            left_score = (var_all_left)
        else:
            left_score = (var_all_left)/(dist_l)

        if dist_r==0:
            right_score = (var_all_right)
        else:
            right_score = (var_all_right)/(dist_r)
        score = len(left_targets) * 1.0 / (len(left_targets) + len(right_targets)) * left_score \
                + len(right_targets) * 1.0 / (len(left_targets) + len(right_targets)) * right_score
        return split_gain, score

    @staticmethod
    def calc_gini_var3_scaler(left_targets, right_targets,left_data,right_data,w_,w_dist):#减少小簇偏向
        """分类树采用基尼指数作为指标来选择最优分裂点"""
        #二分类问题
        split_gain = 0
        w=w_
        w1=0
        w4=w_dist
        gini_left=1
        gini_right=1
        label_counts_left = collections.Counter(left_targets)
        for key in label_counts_left:
            prob = label_counts_left[key] * 1.0 / len(left_targets)
            gini_left -= prob ** 2
        label_counts_right = collections.Counter(right_targets)
        for key in label_counts_right:
            prob = label_counts_right[key] * 1.0 / len(right_targets)
            gini_right -= prob ** 2
        split_gain = len(left_targets) * 1.0 / (len(left_targets) + len(right_targets)) * gini_left\
                     +len(right_targets) * 1.0 / (len(left_targets) + len(right_targets)) * gini_right
        left_targets = left_targets.values
        right_targets = right_targets.values
        left_data = left_data.values
        right_data = right_data.values
        scaler = MinMaxScaler()
        if len(left_data) != 0:
            left_data = scaler.fit_transform(left_data)
        if len(right_data) != 0:
            right_data = scaler.fit_transform(right_data)
        left_label_class = np.unique(left_targets)
        right_label_class = np.unique(right_targets)
        left_label_class=np.array(left_label_class,dtype=int)
        right_label_class=np.array(right_label_class,dtype=int)
        var_all_left = 0
        var_all_right = 0
        dist_l=0
        dist_r=0
        aver_l = np.zeros((2,left_data.shape[1]))
        aver_r = np.zeros((2, right_data.shape[1]))
        for i in left_label_class:
            index_ = np.where(left_targets == i)
            set_ = np.array(left_data[index_])
            v = np.var(set_, axis=0)
            v1 = np.sum(v) / len(v)
            var_all_left = var_all_left + v1
            aver_l[i]=np.mean(set_,axis=0)
        for j in right_label_class:
            index_ = np.where(right_targets == j)
            set_ = np.array(right_data[index_])
            v = np.var(set_, axis=0)
            v1 = np.sum(v) / len(v)
            var_all_right = var_all_right + v1
            aver_r[j] = np.mean(set_, axis=0)
        if len(left_label_class) == 2:
            d=aver_l[0]-aver_l[1]
            dist_l=np.dot(d,d.T)
        if len(right_label_class) == 2:
            dd=aver_r[0]-aver_r[1]
            dist_r=np.dot(dd,dd.T)
        # p1 = len(left_label_class) / 2
        # p2 = len(right_label_class) / 2
        p1=1
        p2=1
        if dist_l==0:
            left_score = (var_all_left)
        else:
            left_score = (var_all_left)/(dist_l)

        if dist_r==0:
            right_score = (var_all_right)
        else:
            right_score = (var_all_right)/(dist_r)
        score = len(left_targets) * 1.0 / (len(left_targets) + len(right_targets)) * left_score \
                + len(right_targets) * 1.0 / (len(left_targets) + len(right_targets)) * right_score
        return split_gain, score

    @staticmethod
    def calc_gini_var_class_no(left_targets, right_targets,left_data,right_data,
                            feature,split_value,w_class,w_class2):#减少小簇偏向
        """分类树采用基尼指数作为指标来选择最优分裂点"""
        split_gain = 0
        w=w_class
        w1=1
        # w2=w_
        w3=w_class2
        gini_left=1
        gini_right=1
        label_counts_left = collections.Counter(left_targets)
        for key in label_counts_left:
            prob = label_counts_left[key] * 1.0 / len(left_targets)
            gini_left -= prob ** 2
        label_counts_right = collections.Counter(right_targets)
        for key in label_counts_right:
            prob = label_counts_right[key] * 1.0 / len(right_targets)
            gini_right -= prob ** 2
        split_gain = len(left_targets) * 1.0 / (len(left_targets) + len(right_targets)) * gini_left\
                     +len(right_targets) * 1.0 / (len(left_targets) + len(right_targets)) * gini_right
        left_item=left_data[feature].values.reshape(-1,1)
        right_item=right_data[feature].values.reshape(-1,1)
        all_item=np.concatenate((left_item,right_item),axis=0)
        max_item=np.max(all_item)
        min_item = np.min(all_item)
        # split_value=np.array(split_value).reshape(-1,1)
        left_targets=left_targets.values
        right_targets = right_targets.values
        left_data = left_data.values
        right_data = right_data.values
        scaler = MinMaxScaler()
        if len(left_data)!=0 and len(right_data)!= 0 :
            data = np.concatenate((left_data, right_data), axis=0)
            scaler.fit(data)
            left_data = scaler.transform(left_data)
            right_data = scaler.transform(right_data)
        elif len(left_data)!=0:
            left_data = scaler.fit_transform(left_data)
        elif len(left_data) != 0:
            right_data = scaler.fit_transform(right_data)
        else:
            score =split_gain
            return split_gain, score

        left_label_class = np.unique(left_targets)
        right_label_class = np.unique(right_targets)
        left_label_class=np.array(left_label_class,dtype=int)
        right_label_class=np.array(right_label_class,dtype=int)
        var_all_left = 0
        var_all_right = 0
        dist_l=0
        dist_r=0
        aver_l = np.zeros((2,left_data.shape[1]))
        aver_r = np.zeros((2, right_data.shape[1]))
        for i in left_label_class:
            index_ = np.where(left_targets == i)
            set_ = np.array(left_data[index_])
            v = np.var(set_, axis=0)
            v1 = np.sum(v) / len(v)
            var_all_left = var_all_left + v1
            aver_l[i]=np.mean(set_,axis=0)
        for j in right_label_class:
            index_ = np.where(right_targets == j)
            set_ = np.array(right_data[index_])
            v = np.var(set_, axis=0)
            v1 = np.sum(v) / len(v)
            var_all_right = var_all_right + v1
            aver_r[j] = np.mean(set_, axis=0)
        if len(left_label_class) == 2:
            d=aver_l[0]-aver_l[1]
            dist_l=np.dot(d,d.T)
        if len(right_label_class) == 2:
            dd=aver_r[0]-aver_r[1]
            dist_r=np.dot(dd,dd.T)
        if dist_l==0:
            # left_score = (var_all_left)
            left_score = 0
        else:
            left_score = (var_all_left)/(dist_l)

        if dist_r==0:
            # right_score = (var_all_right)
            right_score =0
        else:
            right_score = (var_all_right)/(dist_r)
        # p1=len(left_label_class)/2
        # p2 = len(right_label_class) / 2
        # left_score=w1*gini_left+w2*var_all_left*p1
        # right_score = w1*gini_right + w2* var_all_right*p2
        score_onlyvar = len(left_targets) * 1.0 / (len(left_targets) + len(right_targets)) * left_score \
                + len(right_targets) * 1.0 / (len(left_targets) + len(right_targets)) * right_score

        all_data=np.concatenate((left_targets,right_targets),axis=0)
        all_class = np.unique(all_data)
        all_class=np.array(all_class,dtype=int)
        all_class_num=np.max(np.array(all_class,dtype=int))+1
        left_mean=np.zeros((all_class_num,left_data.shape[1]))
        right_mean = np.zeros((all_class_num, right_data.shape[1]))
        dist=np.zeros((all_class_num,2))
        num_p=np.zeros((all_class_num,2))#记录左右两节点分别每类的个数（左，右）第一列为左节点，第二列为右节点
        for i in all_class:
            index_ = np.where(left_targets == i)
            set_ = np.array(left_data[index_])
            if len(set_)!=0:
                if max_item - min_item==0:
                    dist1 = np.min(np.absolute(left_item[index_] - split_value))
                else:
                    dist1 = np.min(np.absolute(left_item[index_] - split_value) / (max_item - min_item))
                dist[i, 0] = dist1
                left_mean[i] = np.mean(set_, axis=0)
                num_p[i,0]=set_.shape[0]
        for j in all_class:
            index_ = np.where(right_targets == j)
            set_ = np.array(right_data[index_])
            if len(set_) != 0:
                if max_item - min_item==0:
                    dist1 = np.min(np.absolute(right_item[index_] - split_value))
                else:
                    dist1 = np.min(np.absolute(right_item[index_] - split_value) / (max_item - min_item))
                dist[j, 1] = dist1
                right_mean[j] = np.mean(set_, axis=0)
                num_p[j, 1] = set_.shape[0]
        class_dist_l_r = 0
        is_same_class = 0
        dist_p=0
        dist_num=0
        for p in all_class:
            if (num_p[p,0]!=0).all() and (num_p[0-p+1,1]!=0).all():
                dist_p=dist[p,0]+dist[0-p+1,1]+dist_p
                dist_num=dist_num+1
            if (num_p[p,0]!=0).all() and (num_p[p,1]!=0).all():
                vector1 = left_mean[p]
                vector2 = right_mean[p]
                # w2 = np.absolute(num_p[p, 0] - num_p[p, 1]) / np.sum(num_p[p, 0] + num_p[p, 1])
                w2 = 1
                class_dist_l_r = np.sqrt(np.sum(np.square(vector1 - vector2))) * w2
                is_same_class = is_same_class + 1
        if is_same_class==0:
            is_same_class=1
        if dist_num == 0:
            dist_num = 1
        class_dist_l_r = class_dist_l_r /is_same_class
        # class_dist_l_r = class_dist_l_r / is_same_class
        score_class = w1 * split_gain - w * class_dist_l_r+w3*dist_p/dist_num
        return split_gain, score_class,score_onlyvar

    @staticmethod
    def calc_gini_var_class_scaler(left_targets, right_targets,left_data,right_data,
                            feature,split_value,w_class,w_class2):#减少小簇偏向
        """分类树采用基尼指数作为指标来选择最优分裂点"""
        split_gain = 0
        w=w_class
        w1=1
        # w2=w_
        w3=w_class2
        gini_left=1
        gini_right=1
        label_counts_left = collections.Counter(left_targets)
        for key in label_counts_left:
            prob = label_counts_left[key] * 1.0 / len(left_targets)
            gini_left -= prob ** 2
        label_counts_right = collections.Counter(right_targets)
        for key in label_counts_right:
            prob = label_counts_right[key] * 1.0 / len(right_targets)
            gini_right -= prob ** 2
        split_gain = len(left_targets) * 1.0 / (len(left_targets) + len(right_targets)) * gini_left\
                     +len(right_targets) * 1.0 / (len(left_targets) + len(right_targets)) * gini_right
        left_item=left_data[feature].values.reshape(-1,1)
        right_item=right_data[feature].values.reshape(-1,1)
        all_item=np.concatenate((left_item,right_item),axis=0)
        max_item=np.max(all_item)
        min_item = np.min(all_item)
        # split_value=np.array(split_value).reshape(-1,1)
        left_targets=left_targets.values
        right_targets = right_targets.values
        left_data = left_data.values
        right_data = right_data.values
        scaler = MinMaxScaler()
        if len(left_data)!=0 and len(right_data)!= 0 :
            data = np.concatenate((left_data, right_data), axis=0)
            scaler.fit(data)
            left_data = scaler.transform(left_data)
            right_data = scaler.transform(right_data)
        elif len(left_data)!=0:
            left_data = scaler.fit_transform(left_data)
        elif len(left_data) != 0:
            right_data = scaler.fit_transform(right_data)
        else:
            score =split_gain
            return split_gain, score

        left_label_class = np.unique(left_targets)
        right_label_class = np.unique(right_targets)
        left_label_class=np.array(left_label_class,dtype=int)
        right_label_class=np.array(right_label_class,dtype=int)
        var_all_left = 0
        var_all_right = 0
        dist_l=0
        dist_r=0
        aver_l = np.zeros((2,left_data.shape[1]))
        aver_r = np.zeros((2, right_data.shape[1]))
        for i in left_label_class:
            index_ = np.where(left_targets == i)
            set_ = np.array(left_data[index_])
            v = np.var(set_, axis=0)
            v1 = np.sum(v) / len(v)
            var_all_left = var_all_left + v1
            aver_l[i]=np.mean(set_,axis=0)
        for j in right_label_class:
            index_ = np.where(right_targets == j)
            set_ = np.array(right_data[index_])
            v = np.var(set_, axis=0)
            v1 = np.sum(v) / len(v)
            var_all_right = var_all_right + v1
            aver_r[j] = np.mean(set_, axis=0)
        if len(left_label_class) == 2:
            d=aver_l[0]-aver_l[1]
            dist_l=np.dot(d,d.T)
        if len(right_label_class) == 2:
            dd=aver_r[0]-aver_r[1]
            dist_r=np.dot(dd,dd.T)
        if dist_l==0:
            left_score = (var_all_left)
        else:
            left_score = (var_all_left)/(dist_l)

        if dist_r==0:
            right_score = (var_all_right)
        else:
            right_score = (var_all_right)/(dist_r)
        # p1=len(left_label_class)/2
        # p2 = len(right_label_class) / 2
        # left_score=w1*gini_left+w2*var_all_left*p1
        # right_score = w1*gini_right + w2* var_all_right*p2
        score_onlyvar = len(left_targets) * 1.0 / (len(left_targets) + len(right_targets)) * left_score \
                + len(right_targets) * 1.0 / (len(left_targets) + len(right_targets)) * right_score

        all_data=np.concatenate((left_targets,right_targets),axis=0)
        all_class = np.unique(all_data)
        all_class=np.array(all_class,dtype=int)
        all_class_num=np.max(np.array(all_class,dtype=int))+1
        left_mean=np.zeros((all_class_num,left_data.shape[1]))
        right_mean = np.zeros((all_class_num, right_data.shape[1]))
        dist=np.zeros((all_class_num,2))
        num_p=np.zeros((all_class_num,2))#记录左右两节点分别每类的个数（左，右）第一列为左节点，第二列为右节点
        for i in all_class:
            index_ = np.where(left_targets == i)
            set_ = np.array(left_data[index_])
            if len(set_)!=0:
                if max_item - min_item==0:
                    dist1 = np.min(np.absolute(left_item[index_] - split_value))
                else:
                    dist1 = np.min(np.absolute(left_item[index_] - split_value) / (max_item - min_item))
                dist[i, 0] = dist1
                left_mean[i] = np.mean(set_, axis=0)
                num_p[i,0]=set_.shape[0]
        for j in all_class:
            index_ = np.where(right_targets == j)
            set_ = np.array(right_data[index_])
            if len(set_) != 0:
                if max_item - min_item==0:
                    dist1 = np.min(np.absolute(right_item[index_] - split_value))
                else:
                    dist1 = np.min(np.absolute(right_item[index_] - split_value) / (max_item - min_item))
                dist[j, 1] = dist1
                right_mean[j] = np.mean(set_, axis=0)
                num_p[j, 1] = set_.shape[0]
        class_dist_l_r = 0
        is_same_class = 0
        dist_p=0
        dist_num=0
        for p in all_class:
            if (num_p[p,0]!=0).all() and (num_p[0-p+1,1]!=0).all():
                dist_p=dist[p,0]+dist[0-p+1,1]+dist_p
                dist_num=dist_num+1
            if (num_p[p,0]!=0).all() and (num_p[p,1]!=0).all():
                vector1 = left_mean[p]
                vector2 = right_mean[p]
                # w2 = np.absolute(num_p[p, 0] - num_p[p, 1]) / np.sum(num_p[p, 0] + num_p[p, 1])
                w2 = 1
                class_dist_l_r = np.sqrt(np.sum(np.square(vector1 - vector2))) * w2
                is_same_class = is_same_class + 1
        if is_same_class==0:
            is_same_class=1
        if dist_num == 0:
            dist_num = 1
        class_dist_l_r = class_dist_l_r /is_same_class
        # class_dist_l_r = class_dist_l_r / is_same_class
        score_class = w1 * split_gain - w * class_dist_l_r+w3*dist_p/dist_num
        return split_gain, score_class,score_onlyvar

    @staticmethod
    def split_dataset(dataset, targets, split_feature, split_value):
        """根据特征和阈值将样本划分成左右两份，左边小于等于阈值，右边大于阈值"""
        left_dataset = dataset[dataset[split_feature] <= split_value]
        left_targets = targets[dataset[split_feature] <= split_value]
        right_dataset = dataset[dataset[split_feature] > split_value]
        right_targets = targets[dataset[split_feature] > split_value]
        return left_dataset, right_dataset, left_targets, right_targets

    def predict(self, dataset):
        """输入样本，预测所属类别"""
        res = []
        for _, row in dataset.iterrows():
            pred_list = []
            # 统计每棵树的预测结果，选取出现次数最多的结果作为最终类别
            for tree in self.trees:
                pred_list.append(tree.calc_predict_value(row))

            pred_label_counts = collections.Counter(pred_list)
            pred_label = max(zip(pred_label_counts.values(), pred_label_counts.keys()))
            res.append(pred_label[1])
        return np.array(res)


if __name__ == '__main__':
    df = pd.read_csv("source/wine.txt")
    df = df[df['label'].isin([1, 2])].sample(frac=1, random_state=66).reset_index(drop=True)
    clf = RandomForestClassifier(n_estimators=5,
                                 max_depth=5,
                                 min_samples_split=6,
                                 min_samples_leaf=2,
                                 min_split_gain=0.0,
                                 colsample_bytree="sqrt",
                                 subsample=0.8,
                                 random_state=66)
    train_count = int(0.7 * len(df))
    feature_list = ["Alcohol", "Malic acid", "Ash", "Alcalinity of ash", "Magnesium", "Total phenols", 
                    "Flavanoids", "Nonflavanoid phenols", "Proanthocyanins", "Color intensity", "Hue", 
                    "OD280/OD315 of diluted wines", "Proline"]
    clf.fit(df.loc[:train_count, feature_list], df.loc[:train_count, 'label'])

    from sklearn import metrics
    print(metrics.accuracy_score(df.loc[:train_count, 'label'], clf.predict(df.loc[:train_count, feature_list])))
    print(metrics.accuracy_score(df.loc[train_count:, 'label'], clf.predict(df.loc[train_count:, feature_list])))
