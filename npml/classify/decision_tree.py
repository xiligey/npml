"""决策树
相关理论 👉 https://github.com/xiligey/npml_theories/blob/master/classify/decision_tree.md
"""
from __future__ import annotations

from typing import Optional

from numpy import ndarray

from npml.model import Classifier


class Node:
    def __init__(self, left: Optional[Node] = None, right: Optional[Node] = None):
        self.left = None
        self.right = None


class DecisionTree(Classifier):
    def __init__(self):
        super().__init__()

    def fit(self) -> None:
        """训练"""

    def prune(self):
        """剪枝"""

    def predict(self) -> ndarray:
        """预测"""
