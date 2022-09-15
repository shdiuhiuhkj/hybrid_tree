# hybrid_tree
决策树是目前最流行的机器学习算法之一，由于其具有可解释性强和训练时间短等优点，得到了广泛应用。作为树生长关键步骤的分裂或分叉准则通常根据纯度和误分类误差等实现，分裂生长分为轴平行和非轴平行方式。这些分裂准则一般与数据内在结构（如类别是否是多簇或单簇组成）无关。为了弥补这一缺失，本文提出了两种混合分裂准则，分别用加权和两步法将同类内的节点间距（Between-node margin within the same class，BNM）和同一节点内的类紧性（Within-class compactness and between-class separation in the same inner node，CSN）与纯度度量相结合。

RandomForestClassification.py #混合分裂准则分类树的具体实现
val_cv_gini.py #调用cart树
val_cv_class.py #调用gini+BNM
val_cv_var3_minmax.py #调用gini+CSN
val_cv_var_class_sort.py #调用gini+CSN+BNM
toy.py #一个简单的异或问题
german，sonar，banana #三个数据集：五折交叉验证法
