 
from inspect import Attribute
from locale import normalize
from pyexpat import model
from sre_constants import RANGE_UNI_IGNORE
from time import time
from types import new_class
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import warnings
from random import random
from tkinter import N
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.rcsetup as rcsetup
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
import xgboost as xgb
from xgboost import plot_importance
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score,classification_report
import warnings
from sklearn.ensemble import VotingClassifier
from itertools import cycle
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize


# data preprocessing 
COVID_lab1 = pd.read_csv("C:/Users/woqls/Desktop/COVID_19/COVID-19_rawdata/final_data/yr_1_lab.csv")
COVID_lab2 = pd.read_csv("C:/Users/woqls/Desktop/COVID_19/COVID-19_rawdata/final_data/yr_2_lab.csv")


COVID_lab1.shape   # 420 x 37
COVID_lab1 = COVID_lab1.drop(['sample_id'],axis=1)
COVID_lab2 = COVID_lab2.drop(['sample_id'],axis=1)

COVID_lab2['status'].value_counts()
COVID_lab2 = COVID_lab2[(COVID_lab2['status']=='Severe')|(COVID_lab2['status']=='Moderate')]

COVID_lab2['status'] = COVID_lab2['status'].replace(['Severe'],[2])
COVID_lab2['status'] = COVID_lab2['status'].replace(['Moderate'],[1])



COVID = pd.concat([COVID_lab1,COVID_lab2])
COVID['status'].value_counts()
COVID.shape
COVID['LAB_PLATELET'].mean()
x1 = COVID.iloc[:,1:300]
y1 = COVID.iloc[:,0]
y1.value_counts()


x1.isna().sum().sum()  # 누락된 값 찾기
x1[x1.isna().any(axis=1)]  # 누락된 값의 행 출력
x1.isnull().sum()  # LAB_FOLATE
x1['LAB_FOLATE'].fillna((x1['LAB_FOLATE'].median()), inplace=True)  # median imputation 


x1 = x1[x1.columns.difference(['LAB_RBC','LAB_HEMATOCRIT','LAB_HEMOGLOBIN','LAB_APOA1','LAB_HDLC','LAB_ALBUMIN','LAB_PROTEIN','LAB_CREATININE','LAB_BUN'])]
x1.columns
x_train, x_test, y_train, y_test = train_test_split(x1,
                                                    y1,
                                                    stratify=y1,
                                                    test_size=0.3,
                                                    shuffle=True,
                                                    random_state=1004)



print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
y_train.value_counts()

scaler = StandardScaler()
x_train_scale = scaler.fit_transform(x_train)
x_train_scale
x_test_scale = scaler.transform(x_test)

# SMOTE

smote = SMOTE(random_state=0)
x_train_over, y_train_over = smote.fit_resample(x_train_scale,y_train)
print('SMOTE 적용 후 학습용 피처/레이블 데이터 세트:', x_train_over.shape, y_train_over.shape)
print('SMOTE 적용 후 레이블 값 분포: \n', pd.Series(y_train_over).value_counts())


# for CM visualization
label = ['healthy_control','moderate','severe']
# LR

softmax_reg = LogisticRegression(multi_class='multinomial',solver='lbfgs',C=10,random_state=42)
softmax_reg.fit(x_train_over,y_train_over)

# Evaluation
y_LR = softmax_reg.predict(x_test_scale)
print(classification_report(y_test,y_LR))
LR_cm = confusion_matrix(y_test, y_LR)
cm_display = ConfusionMatrixDisplay(LR_cm,display_labels=label).plot()
plt.show()
plt.rc('font', size=30)
accuracy_score(y_test,y_LR)


# RF - Hyperparameter option
params = { 'n_estimators' : [10,25,50,75,100],
           'max_depth' : [6, 8, 10, 12],
           'min_samples_leaf' : [8, 12, 18],
           'min_samples_split' : [8, 16, 20]
            }

# RandomForestClassifier-GridSearchCV
rf_clf1 = RandomForestClassifier(random_state = 2, n_jobs = -1)
grid_cv = GridSearchCV(rf_clf1, param_grid = params, cv = 10, n_jobs = -1)
grid_cv.fit(x_train_over, y_train_over)
print('최적 하이퍼 파라미터: ', grid_cv.best_params_)
print('최고 예측 정확도: {:.4f}'.format(grid_cv.best_score_))

# best parameter
rf_clf1 = RandomForestClassifier(max_depth =  6, 
                                 min_samples_leaf =8, 
                                 min_samples_split = 8, 
                                 n_estimators = 75,
                                 random_state=2,
                                 n_jobs=-1)
rf_train = rf_clf1.fit(x_train_over, y_train_over)     # lab:8,8,8,50,  cytokine:8,8,8,100
rf_pred = rf_clf1.predict(x_test_scale)
print('예측 정확도: {:.4f}'.format(accuracy_score(y_test,rf_pred)))
confusion_matrix(y_test,rf_pred)
print(classification_report(y_test, rf_pred, target_names=['class 0', 'class 1','class 2']))


label = ['healthy_control','moderate','severe']
Rf_cm = confusion_matrix(y_test, rf_pred)
cm_display = ConfusionMatrixDisplay(Rf_cm,display_labels=label).plot()
plt.show()
plt.rc('font', size=12)

 
ftr_importances_values = rf_clf1.feature_importances_
ftr_importances = pd.Series(ftr_importances_values, index = x_train.columns)
ftr_top20 = ftr_importances.sort_values(ascending=False)[:20]
ftr_top20
plt.figure(figsize=(3,3))
plt.title('Top 20 RF_Feature Importances')
sb.barplot(x=ftr_top20, y=ftr_top20.index)
plt.show()
plt.rc('ytick', labelsize=10)  # y축



# GradientBoosting

dtrain = xgb.DMatrix(x_train_over, y_train_over)
dtest = xgb.DMatrix(x_test_scale,y_test)

params_xgb = {'n_estimator': [100,200,300,400],
              'colsamole_bytree':[0.3,0.4,0.5,0,7,0.8,0.9,1],
              'max_depth': 3,
              'eta': 0.1,
              'objective':'multi:softmax',
              'eval_metric':'merror',
              'num_class': 3,
              'early_stopping' : 500,
              'booster':'gbtree'}

num_rounds = 400

wlist = [(dtrain, 'train'), (dtest, 'eval')]
xgb_model = xgb.train(params = params_xgb, dtrain=dtrain, num_boost_round = num_rounds, evals = wlist)
y_pred = xgb_model.predict(dtest)
accuracy_score(y_test,y_pred)
print(classification_report(y_test,y_pred))
print(confusion_matrix(y_test, y_pred))

from xgboost import XGBClassifier
new_xgb = XGBClassifier(n_estimator=400, learning_rate=0.1,max_depth=3,eta=0.1,random_state=2)

new_xgb.fit(x_train_over,y_train_over,
            early_stopping_rounds=500,
            eval_metric='merror',
            eval_set=[(x_test_scale,y_test)],
            verbose=True)

xg_pred = new_xgb.predict(x_test_scale)
accuracy_score(y_test,xg_pred)
print(classification_report(y_test,xg_pred))
print(confusion_matrix(y_test, xg_pred))

label = ['healthy_control','moderate','severe']
xgb_cm = confusion_matrix(y_test, xg_pred)
cm_display = ConfusionMatrixDisplay(xgb_cm,display_labels=label).plot()
plt.show()
plt.rc('font', size=20)

#Feature importance
from xgboost import plot_importance
ft_importances_values = new_xgb.feature_importances_
ft_importances = pd.Series(ft_importances_values, index = x_train.columns)
ft_top20 = ft_importances.sort_values(ascending=False)[:20]
ft_top20
plt.figure(figsize=(3,3))
plt.title('Top 20 XGB_Feature Importances')
sb.barplot(x=ft_top20, y=ft_top20.index)
plt.show()
plt.rc('ytick', labelsize=15)  # y축



# Ensemble 
# xgb_model, rf_clf1
vo_clf = VotingClassifier(estimators=[('RF',rf_clf1),('xgb',new_xgb),('LR',softmax_reg)],voting='soft')
vo_clf.fit(x_train_over,y_train_over)
En_pred = vo_clf.predict(x_test_scale)

accuracy_score(y_test,En_pred)
print(classification_report(y_test,En_pred))
print(confusion_matrix(y_test, En_pred))
print(accuracy_score(y_test, En_pred))

en_cm = confusion_matrix(y_test, En_pred)
cm_display = ConfusionMatrixDisplay(en_cm,display_labels=label).plot()
plt.show()
plt.rc('font', size=15)
plt.rc('ytick', labelsize=20)


# roc curve
labels =[0,1,2]
y_test = label_binarize(y_test,classes = labels)
y_pred = label_binarize(En_pred, classes = labels)


n_classes = 3
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])


# Plot of a ROC curve for a specific class
plt.figure(figsize=(15, 5))
for idx, i in enumerate(range(n_classes)):
    plt.subplot(131+idx)
    plt.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f)' % roc_auc[i])
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Class %0.0f' % idx)
    plt.legend(loc="lower right")
plt.show()

print("roc_auc_score: ", roc_auc_score(y_test, y_pred, multi_class='raise')) 


import shap
import skimage
explainer = shap.TreeExplainer(new_xgb)
shap_values = explainer.shap_values(x_test_scale)
shap.initjs()
shap.summary_plot(shap_values,x_test_scale)