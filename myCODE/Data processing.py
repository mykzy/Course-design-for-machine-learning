#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/6/26 3:28
# @Author  : j'x   like  kdqs
# @File    : Data processing.py
# @Description :机器学习的课设

import pickle

import implicit
import numpy as np
import pandas as pd
from scipy import sparse

model_file = 'als_model.pkl'

# 加载数据集
data = pd.read_csv('../data/train_dataset.csv', names=['user_id', 'item_id'])

# 转换成稀疏矩阵
interaction_matrix = pd.crosstab(index=data['user_id'], columns=data['item_id'])
sparse_data = sparse.csr_matrix(interaction_matrix)

# 计算物品之间的相似度，使用ALS算法
model = implicit.als.AlternatingLeastSquares(factors=128, regularization=0.1, iterations=10)
model.fit(sparse_data)

with open(model_file, 'wb') as f:
    pickle.dump(model, f)


def load_model(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)


# 加载模型
model = load_model(model_file)


# 推荐N个物品给用户
def get_top_recommended_items(model, user_id, n=10):
    recommended_items = []
    user_items = sparse_data.T.tocsr()
    user_index = user_id
    item_users = user_items[model.item_factors.T.dot(model.user_factors[user_index])]
    # 去掉用户已经交互过的物品
    item_users[sparse_data[user_index].toarray().flatten() > 0] = 0
    # 推荐得分最高的N个物品
    top_users = np.argsort(item_users)[-n:]
    recommended_items = top_users.tolist()
    return recommended_items


print(get_top_recommended_items(model, 0))
