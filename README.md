# npml
Npml(Numpy machine learning)，是一个基于numpy实现常见的机器学习算法的python包。

> 配套算法理论知识详解 👉 [https://github.com/xiligey/npml_theories](https://github.com/xiligey/npml_theories)

## 项目结构
- data: 存放一些测试用到的数据
- docs: 各个算法的理论介绍
- npml: npml的所有代码
- tests: npml单元测试目录

## 算法清单

- npml

  - 回归
    - [x] [普通最小二乘法](npml/regress/ordinary_least_squares.py)
    - [ ] [Lasso回归](npml/regress/lasso.py)
    - [x] [岭回归](npml/regress/ridge.py)
    - [ ] [多项式回归](npml/regress/polynomial.py)
    - [ ] [弹性网络](npml/regress/elastic_network.py)
  - 分类
    - [x] [k最近邻](npml/classify/k_nearest_neighbors.py)
    - [ ] [决策树](npml/classify/decision_tree.py)
    - [ ] [朴素贝叶斯](npml/classify/naive_bayes.py)
    - [ ] [支持向量机](npml/classify/svm.py)
    - [ ] [逻辑回归](npml/classify/logistic.py)
    - 集成模型
      - [ ] [随机森林](npml/classify/ensemble/random_forest.py)
      - [ ] [bagging](npml/classify/ensemble/bagging.py)
      - [ ] [boost](npml/classify/ensemble/boost.py)
      - [ ] [adaboost](npml/classify/ensemble/adaboost.py)
      - [ ] [stacking](npml/classify/ensemble/stacking.py)
  - 聚类
    - [x] [kmeans聚类](npml/cluster/kmeans.py)
    - [ ] [dbscan](npml/cluster/dbscan.py)
    - [ ] [层次聚类](npml/cluster/hierarchical.py)
  - 降维
    - [ ] [主成分分析](npml/dimension_reduct/pca.py)
    - [ ] [LDA](npml/dimension_reduct/lda.py)
  - 时间序列
  - 数据挖掘
