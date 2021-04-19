# npml
Npml(Numpy machine learning)，是一个基于numpy实现常见的机器学习算法的python包。

## 项目结构
- data: 存放一些测试用到的数据
- docs: 各个算法的理论介绍
- npml: npml的所有代码
- tests: npml单元测试目录

## 算法清单

- npml

  - 回归
  - 分类
    - 
  - 聚类
  - 降维
  - 时间序列
  - 
  - 数据挖掘

- [ ] [关联规则(association_rules)](association_rules)
  - [ ] [Aproior](association_rules/Aproior.py)
  - [ ] [FPGrowth](association_rules/FPGrowth.py)
- [ ] [聚类(cluster)](cluster)
  - [ ] [K 均值聚类(KMeans)](cluster/KMeans.py)
  - [ ] [密度聚类(DBSCAN)](cluster/DBSCAN.py)
  - [ ] [层次聚类(HierarchicalClustering)](cluster/HierarchicalClustering.py)
- [ ] [降维(dimensionality_reduction)](dimensionality_reduction)
  - [ ] [主成分分析(PCA)](dimensionality_reduction/PCA.py)
  - [ ] [LDA](dimensionality_reduction/LDA.py)
- [ ] [集成模型(ensemble)](ensemble)
  - [ ] [Bagging](ensemble/Bagging.py)
  - [ ] [Boost](ensemble/Boost.py)
  - [ ] [Stacking](ensemble/Stacking.py)
- [ ] [线性模型(linear_model)](linear_model)
  - [x] [普通最小二乘法(OLS)](linear_model/OrdinaryLeastSquare.py)
  - [ ] [岭回归(Ridge)](linear_model/Ridge.py)
  - [ ] [Lasso 回归(Lasso)](linear_model/Lasso.py)
- [ ] [自然语言处理(natural_language_processing)](natural_language_processing)
- [ ] [近邻(neighbors)](neighbors)
- [ ] [神经网络(neural_network)](neural_network)
- [ ] [推荐系统(recommended_system)](recommended_system)
- [ ] [支持向量机(svm)](svm)
- [ ] [树模型(tree)](tree)
  - [ ] [决策树](tree/DecisionTree.py)
  - [ ] [GBDT](tree/GradientBoostedDecisionTree.py) 
- [ ] [时间序列模型(time_series)](time_series)
    - [ ] [ARIMA](time_series/ARIMA.py)
    - [ ] [指数平滑法](time_series/HoltWinters.py)
- [ ] [异常检测(anomaly_detect)](anomaly_detection)
    - [ ] [KSigma](anomaly_detection/KSigma.py)
    - [ ] [Boxplot](anomaly_detection/Boxplot.py)
    - [ ] [MAD](anomaly_detection/MAD.py)
    - [ ] [PAD](anomaly_detection/PAD.py)
    - [ ] [移动平均]()