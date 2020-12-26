import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

import pickle
import category_encoders as ce

from scipy.special import boxcox1p

from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, RandomForestClassifier, \
    ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score

# %%

train_path = os.path.join('data', 'kaggle')

for dirname, _, filenames in os.walk(train_path):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# %%
data = pd.read_csv('data/kaggle/train.csv')
y = (data['churn'] == 'yes').astype(int)
print(data.shape)
print(data.head())
print(data.info())

feature_cols = data.columns[:-1]
correlations = data[feature_cols].corrwith(y)
correlations.sort_values(inplace=True)
print(correlations)

os.chdir('data')

correlations.plot(kind='bar')
plt.show()

# %%
# Get the split indexes
from sklearn.model_selection import StratifiedShuffleSplit

# Get the split indexes
strat_shuf_split = StratifiedShuffleSplit(n_splits=1,
                                          test_size=0.3,
                                          random_state=42)

train_idx, test_idx = next(strat_shuf_split.split(data[feature_cols], data.churn))

# Create the dataframes
X_train = data.loc[train_idx, feature_cols]
y_train = data.loc[train_idx, 'churn']

X_test = data.loc[test_idx, feature_cols]
y_test = data.loc[test_idx, 'churn']

print(y_train.value_counts(normalize=True))
print(y_test.value_counts(normalize=True))


# %%
class NorTransfer(BaseEstimator, TransformerMixin):
    def __init__(self, attr_names):  # no *args or **kargs
        self.attr_names = 'test'

    def fit(self, X, y=None):
        return self  # nothing else to do

    def transform(self, X):
        skew_limit = 0.75
        X = pd.DataFrame(X, columns=list(train_num))
        mask = X.apply(lambda x: x.nunique()) > 2
        num_cols = X.columns[mask]

        skew_vals = X[num_cols].skew()
        skew_cols = (skew_vals[skew_vals > skew_limit].sort_values().to_frame().rename(columns={0: 'Skew'}))

        print('-----before normalization------')
        print(skew_cols)

        for col in skew_cols.index.tolist():
            if col == 'churn':
                continue
            X[col] = X[col].apply(np.log1p)
            # X[col] = X[col].apply(stats.boxcox)
            # X[col] = boxcox1p(X[col],0.25)
            # train[col] = np.log1p(train[col]) I am not sure about whether normalization before or after split
            # test[col] = test[col].apply(np.log1p)
            skew_vals = X[num_cols].skew()
            skew_cols = (skew_vals[skew_vals > skew_limit].sort_values().to_frame().rename(columns={0: 'Skew'}))

        print('-----after normalization------')
        print(skew_cols)

        return X.values


# %%
# Normalization applied to numerical attributes
train_num = X_train.select_dtypes(include=['float64', 'int64'])

num_pipeline = Pipeline([
    ('norTransfer', NorTransfer(attr_names='test')),
    ('std_scaler', StandardScaler())
])

# Mask to select obj columns
num_attribs = list(train_num)

mask_obj_columns = (X_train.dtypes == np.object)
obj_col = X_train.loc[:, mask_obj_columns]

mask_ohe_columns = (obj_col.apply(lambda x: x.nunique()) > 2)
obj_ohe = obj_col.loc[:, mask_ohe_columns]
cat_attribs = list(obj_ohe.columns)

mask_label_columns = (obj_col.apply(lambda x: x.nunique()) <= 2)
obj_label = obj_col.loc[:, mask_label_columns]
label_attribs = list(obj_label.columns)

print(num_attribs)
print(cat_attribs)
print(label_attribs)

full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", OneHotEncoder(sparse=False), cat_attribs),
    ("label", ce.BinaryEncoder(), label_attribs)
])

X_train_prepared = full_pipeline.fit_transform(X_train)
print(pd.DataFrame(X_train_prepared))

X_test_prepared = full_pipeline.fit_transform(X_test)
print(pd.DataFrame(X_test_prepared))

# %%
le = LabelEncoder()
y_train_prepared = le.fit_transform(y_train)
y_test_prepared = le.fit_transform(y_test)

print(y_train_prepared)
print(y_test_prepared)

# %%
sns.set_style('white')
pd.DataFrame(X_train_prepared).hist(bins=50, figsize=(20, 20), color='green')
plt.show()

# %% Logistic regression
# Standard logistic regression
param_lr = {
    'penalty': ['l1', 'l2'],
    'C': np.logspace(-4, 4, 20)
}
lr = GridSearchCV(LogisticRegression(solver='liblinear', max_iter=1000, random_state=42), param_lr, verbose=True, cv=4,
                  n_jobs=-1)
lr.fit(X_train_prepared, y_train_prepared)
print(lr.best_params_)
pkl_lr = open('lr.pkl', 'wb')
pickle.dump(lr, pkl_lr)
pkl_lr.close()
# %%
lr = pickle.load(open('lr.pkl', 'rb'))
print(lr.best_params_)

y_pred_lr = lr.predict(X_test_prepared)

precision, recall, fscore, _ = score(y_test_prepared, y_pred_lr, average='weighted')
accuracy = accuracy_score(y_test_prepared, y_pred_lr)

metrics = list()
metrics.append(pd.Series({'precision': precision, 'recall': recall,
                          'fscore': fscore, 'accuracy': accuracy},
                         name='lr'))

lr_cm = confusion_matrix(y_test_prepared, y_pred_lr)
sns.heatmap(lr_cm, annot=True, fmt='d', cmap=mpl.cm.binary)
plt.show()
# %%
param_knn = {
    'n_neighbors': list(range(2, 11)),
    'weights': ['distance', 'uniform']
}
knn = GridSearchCV(KNeighborsClassifier(), param_knn, verbose=True, cv=4, n_jobs=-1)

knn.fit(X_train_prepared, y_train_prepared)
print(knn.best_params_)

y_pred_knn = knn.predict(X_test_prepared)

precision_knn, recall_knn, fscore_knn, _knn = score(y_test_prepared, y_pred_knn, average='weighted')
accuracy_knn = accuracy_score(y_test_prepared, y_pred_knn)

metrics.append(pd.Series({'precision': precision_knn, 'recall': recall_knn,
                          'fscore': fscore_knn, 'accuracy': accuracy_knn},
                         name='knn'))

knn_cm = confusion_matrix(y_test_prepared, y_pred_knn)
sns.heatmap(knn_cm, annot=True, fmt='d', cmap=mpl.cm.binary)
plt.show()

# %%
param_tree = {
    'criterion': ['gini', 'entropy'],
    'splitter': ['best', 'random'],
    'max_depth': list(np.arange(3, 10))
}

tree = GridSearchCV(DecisionTreeClassifier(), param_tree, verbose=True, cv=4, n_jobs=-1)

tree.fit(X_train_prepared, y_train_prepared)
print(tree.best_params_)

y_pred_tree = tree.predict(X_test_prepared)

precision_tree, recall_tree, fscore_tree, _tree = score(y_test_prepared, y_pred_tree, average='weighted')
accuracy_tree = accuracy_score(y_test_prepared, y_pred_tree)

metrics.append(pd.Series({'precision': precision_tree, 'recall': recall_tree,
                          'fscore': fscore_tree, 'accuracy': accuracy_tree},
                         name='DTree'))

cm_tree = confusion_matrix(y_test_prepared, y_pred_tree)
sns.heatmap(cm_tree, annot=True, fmt='d', cmap=mpl.cm.binary)
plt.show()

# %%
param_rf = {
    'criterion': ['gini', 'entropy'],
    'n_estimators': [15, 20, 30, 40, 50, 100, 150, 200, 300, 400],
    'max_depth': list(np.arange(5, 15)),
    'warm_start': [True, False],
    'oob_score': [True]
}

rf = GridSearchCV(RandomForestClassifier(), param_rf, verbose=True, cv=4, n_jobs=-1)
rf.fit(X_train_prepared, y_train_prepared)
print(rf.best_params_)

y_pred_rf = rf.predict(X_test_prepared)

precision_rf, recall_rf, fscore_rf, _rf = score(y_test_prepared, y_pred_rf, average='weighted')
accuracy_rf = accuracy_score(y_test_prepared, y_pred_rf)

metrics.append(pd.Series({'precision': precision_rf, 'recall': recall_rf,
                          'fscore': fscore_rf, 'accuracy': accuracy_rf},
                         name='RandomForest'))

cm_rf = confusion_matrix(y_test_prepared, y_pred_rf)
sns.heatmap(cm_rf, annot=True, fmt='d', cmap=mpl.cm.binary)
plt.show()

# %%
param_etc = {
    'criterion': ['gini', 'entropy'],
    'n_estimators': [15, 20, 30, 40, 50, 100, 150, 200, 300, 400],
    'max_depth': list(np.arange(5, 15)),
    'warm_start': [True, False],
    'bootstrap': [True],
    'oob_score': [True, False]
}

extrTree = GridSearchCV(ExtraTreesClassifier(), param_etc, verbose=True, cv=4, n_jobs=-1)
extrTree.fit(X_train_prepared, y_train_prepared)
print(extrTree.best_params_)

y_pred_extrTree = extrTree.predict(X_test_prepared)

precision_extrTree, recall_extrTree, fscore_extrTree, _extrTree = score(y_test_prepared, y_pred_extrTree,
                                                                        average='weighted')
accuracy_extrTree = accuracy_score(y_test_prepared, y_pred_extrTree)

metrics.append(pd.Series({'precision': precision_extrTree, 'recall': recall_extrTree,
                          'fscore': fscore_extrTree, 'accuracy': accuracy_extrTree},
                         name='ExtraTrees'))

cm_extrTree = confusion_matrix(y_test_prepared, y_pred_extrTree)
sns.heatmap(cm_extrTree, annot=True, fmt='d', cmap=mpl.cm.binary)
plt.show()

# %%
param_gb = {
    'criterion': ['friedman_mse', 'mse'],
    'n_estimators': [30, 50, 100, 150, 200, 300],
    "learning_rate": [0.01, 0.05, 0.1, 0.15, 0.2],
    #    "min_samples_split": np.linspace(0.1, 0.5, 5),
    #    "min_samples_leaf": np.linspace(0.1, 0.5, 5),
    "max_depth": [3, 5, 8],
    "max_features": ["log2", "sqrt"],
    "subsample": [0.5, 0.75, 1.0],
}

gboost = GridSearchCV(GradientBoostingClassifier(), param_gb, verbose=True, cv=4, n_jobs=-1)
gboost.fit(X_train_prepared, y_train_prepared)
print(gboost.best_params_)
import pickle

pkl_gboost_file = open('gboot.pkl', 'wb')
pickle.dump(gboost, pkl_gboost_file)
pkl_gboost_file.close()
# %%
gboost = pickle.load(open('gboot.pkl', 'rb'))
print(gboost.best_params_)

y_pred_gboost = gboost.predict(X_test_prepared)

precision_gboost, recall_gboost, fscore_gboost, _gboost = score(y_test_prepared, y_pred_gboost, average='weighted')
accuracy_gboost = accuracy_score(y_test_prepared, y_pred_gboost)

metrics.append(pd.Series({'precision': precision_gboost, 'recall': recall_gboost,
                          'fscore': fscore_gboost, 'accuracy': accuracy_gboost},
                         name='GradientBoosting'))

cm_gboost = confusion_matrix(y_test_prepared, y_pred_gboost)
sns.heatmap(cm_gboost, annot=True, fmt='d', cmap=mpl.cm.binary)
plt.show()

# %%
param_aboost = {
    'n_estimators': [30, 50, 100, 150, 200, 300],
    "learning_rate": [0.01, 0.05, 0.1, 0.15, 0.2],
    "algorithm": ['SAMME', 'SAMME.R'],
}

aboost = GridSearchCV(AdaBoostClassifier(), param_aboost, verbose=True, cv=4, n_jobs=-1)
aboost.fit(X_train_prepared, y_train_prepared)
print(aboost.best_params_)

pkl_aboost_file = open('aboot.pkl', 'wb')
pickle.dump(aboost, pkl_aboost_file)
pkl_aboost_file.close()

# %%
aboost = pickle.load(open('gboot.pkl', 'rb'))
print(aboost.best_params_)

y_pred_aboost = aboost.predict(X_test_prepared)

precision_aboost, recall_aboost, fscore_aboost, _aboost = score(y_test_prepared, y_pred_aboost, average='weighted')
accuracy_aboost = accuracy_score(y_test_prepared, y_pred_aboost)

metrics.append(pd.Series({'precision': precision_aboost, 'recall': recall_aboost,
                          'fscore': fscore_aboost, 'accuracy': accuracy_aboost},
                         name='AdamBoosting'))

cm_aboost = confusion_matrix(y_test_prepared, y_pred_aboost)
sns.heatmap(cm_aboost, annot=True, fmt='d', cmap=mpl.cm.binary)
plt.show()

# %%
metrics_pd = pd.concat(metrics, axis=1)

print(metrics_pd)

# %% Plotting the Kaplan-Meier Curve
from lifelines import KaplanMeierFitter, CoxPHFitter

kmf = KaplanMeierFitter()

kmf.fit(data.account_length, data.churn, label='Kaplan Meier Estimate, full sample')

kmf.plot(linewidth=4, figsize=(12, 6))
plt.title('Customer Churn: Kaplan-Meier Curve')
plt.xlabel('Months')
plt.ylabel('Survival probability')
plt.show()

# %% Plotting the Kaplan-Meier Curve for two different groups

df1 = train[train.international_plan == 0]
df2 = train[train.international_plan == 1]

kmf.fit(df1.account_length, df1.churn)
kmf.plot(label='Domestic Plan', figsize=(12, 6))
kmf.fit(df2.account_length, df2.churn)
kmf.plot(label='International Plan')
plt.title('International Plan and Churn: Kaplan-Meier Curve')
plt.xlabel('Months')
plt.ylabel('Survival probability')
plt.show()

df1 = train[train.voice_mail_plan == 0]
df2 = train[train.voice_mail_plan == 1]

kmf.fit(df1.account_length, df1.churn)
kmf.plot(label='Voice Mail Plan', figsize=(12, 6))
kmf.fit(df2.account_length, df2.churn)
kmf.plot(label='Voice Mail Plan')
plt.title('Voice Mail Plan and Churn: Kaplan-Meier Curve')
plt.xlabel('Months')
plt.ylabel('Survival probability')
plt.show()
# %%
from lifelines.utils.sklearn_adapter import sklearn_adapter

from lifelines import CoxPHFitter
from sklearn.model_selection import cross_val_score

X = train.copy().drop('account_length', axis=1).copy()  # keep as a dataframe
Y = train.copy().pop('account_length').copy()

base_cox = sklearn_adapter(CoxPHFitter, event_col='churn')
wf = base_cox()

scores = cross_val_score(wf, X, Y, cv=10)
print(np.mean(scores))

# %% Fitting Cox Proportional Model

from lifelines import WeibullAFTFitter

X = train.copy().drop('account_length', axis=1)  # keep as a dataframe
Y = train.copy().pop('account_length').copy()
base_aft = sklearn_adapter(WeibullAFTFitter, event_col='churn')
aft = base_aft()

scores = cross_val_score(aft, X, Y, cv=10)
print(np.mean(scores))

# %%
from lifelines import LogLogisticAFTFitter
from lifelines import LogNormalAFTFitter

X = train.copy().drop('account_length', axis=1)  # keep as a dataframe
Y = train.copy().pop('account_length').copy()
base_lognor = sklearn_adapter(LogNormalAFTFitter, event_col='churn')
lognor = base_lognor()

scores = cross_val_score(lognor, X, Y, cv=10)
print(np.mean(scores))

# %%
from sklearn.model_selection import GridSearchCV

clf = GridSearchCV(wf, {
    "penalizer": 10.0 ** np.arange(-2, 3),
    "l1_ratio": [0, 1 / 3, 2 / 3],
    "model_ancillary": [True, False],
}, cv=4)
clf.fit(X, Y)

print(clf.best_estimator_)

from sklearn.model_selection import train_test_split

# Training and Validation set split based on the stratified churn groups
X = train.drop('churn', axis=1).copy()

y = train['churn'].copy()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y)

# %% test boosting works or not
import xgboost as xgb

clf_xgb = xgb.XGBClassifier(seed=42,
                            objective='binary:logistic',
                            gamma=0.25,
                            learn_rate=0.01,
                            max_depth=4,
                            reg_lambda=10,
                            scale_pos_weight=1,
                            subsample=0.9,
                            colsample_bytree=0.5)
clf_xgb.fit(X_train,
            y_train,
            verbose=True,
            early_stopping_rounds=10,
            eval_metric='aucpr',
            eval_set=[(X_test, y_test)])
