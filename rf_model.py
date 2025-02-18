import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score, roc_curve, auc,precision_recall_curve
from bayes_opt import BayesianOptimization
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn import metrics
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
from sklearn.metrics import confusion_matrix
from sklearn.datasets import make_classification
from sklearn.calibration import CalibratedClassifierCV
import shap
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_curve, auc
import pickle
import os

# 读取数据
data = pd.read_csv('E:\\RF_model_app.csv')

#将连续型数据和分类型数据进行分开
continuous_cols = data[["ALC", "ALB","leukocyte" ,"platelet"]]
categorical_cols = data[[ "chemotherapy regimens", "cycle", "G-CSF","BMI","underlying disease","breast cancer types"]]

#对连续型数据先进行标准化（可以助于保持数据的原始分布），将不同量纲统一
scaler = StandardScaler()
standardized_data = scaler.fit_transform(continuous_cols)

#columns=continuous_cols.columns 按照原来的列名进行保存
standardized_data = pd.DataFrame(standardized_data, columns=continuous_cols.columns)

#对标准化的数据进行归一化（0,1）
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(standardized_data)
normalized_data = pd.DataFrame(normalized_data, columns=continuous_cols.columns)

#将标准化、归一化的数据保存为连续型，为后期合并数据做准备
continuous_data = standardized_data

#对分类型数据进行编码处理
categorical_data=pd.get_dummies(categorical_cols,drop_first=True)

#读取原始数据的因变量
y_data=data['result']

#按照原始数据的列名对处理好的数据进行合并
processed_data = pd.concat([categorical_data, continuous_data, y_data], axis=1)

#获取特征及目标列
X=processed_data.drop('result', axis=1)
y=processed_data['result']

#划分数据
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y,random_state=42)

# # 定义随机森林模型评估函数
# def rf_cv(n_estimators, min_samples_split, max_features):
#     model = RandomForestClassifier(n_estimators=int(n_estimators),
#                                    min_samples_split=int(min_samples_split),
#                                    max_features=int(max_features),
#                                    random_state=42)
#     return cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc').mean()
#
# # 定义参数搜索空间
# pbounds = {'n_estimators': (1, 200),#400 350，200
#            'min_samples_split':(2,15),#(2,20)15
#            'max_features': (1, 11)}
#
# # 使用贝叶斯优化搜索最佳参数
# optimizer = BayesianOptimization(f=rf_cv, pbounds=pbounds, random_state=42)
# optimizer.maximize(init_points=10, n_iter=50)
#
# # 输出最佳参数
# best_params = optimizer.max['params']
# print("最佳参数：", best_params)

# 使用最佳参数训练模型
rf_model = RandomForestClassifier(n_estimators=99,
                                  min_samples_split=10,
                                  max_features=1,
                                  random_state=42)
rf_model.fit(X_train, y_train)
# 模型在测试集上的AUC
rf_score = rf_model.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, rf_score)
print('Test AUC:', roc_auc)
#模型在测试集上的性能指标
y_pred = rf_model.predict(X_test)
print(classification_report(y_test, y_pred))
print(X_train.columns)  # 如果使用了DataFrame
# 保存模型为 pkl 文件
with open('rf_model.pkl', 'wb') as file:
    pickle.dump(rf_model, file)
print("模型已保存为 rf_model.pkl")
print("文件路径：", os.path.abspath('rf_model.pkl'))
