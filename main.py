import numpy as np
import pandas as pd
import os
import matplotlib as mpl
import matplotlib.pyplot as plt

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

import os, numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns

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
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from scipy.special import boxcox1p


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
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler, LabelBinarizer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import category_encoders as ce

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
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y_train_prepared = le.fit_transform(y_train)
y_test_prepared = le.fit_transform(y_test)

print(y_train_prepared)
print(y_test_prepared)

# %%
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('white')
pd.DataFrame(X_train_prepared).hist(bins=50, figsize=(20, 20), color='green')
plt.show()

# %% Logistic regression
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

# Standard logistic regression
param_lr = {
    'penalty': ['l1', 'l2'],
    'C': np.logspace(-4, 4, 20)
}
lr = GridSearchCV(LogisticRegression(solver='liblinear'), param_lr, verbose=True, cv=4, n_jobs=-1)
lr.fit(X_train_prepared, y_train_prepared)
print(lr.best_params_)

y_pred_lr = lr.predict(X_test_prepared)

precision, recall, fscore, _ = score(y_test_prepared, y_pred_lr, average='weighted')

print('precision', precision)
print('recall', recall)
print('fscore', fscore)

lr_cm = confusion_matrix(y_test_prepared, y_pred_lr)
sns.heatmap(lr_cm, annot=True, fmt='d', cmap=mpl.cm.binary)
plt.show()
# %%
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV


param_knn = {
    'n_neighbors': list(range(2, 11)),
    'weights': ['distance', 'uniform']
}
knn = GridSearchCV(KNeighborsClassifier(), param_knn, verbose=True, cv=4, n_jobs=-1)

knn.fit(X_train_prepared, y_train_prepared)
print(knn.best_params_)

y_pred_knn = knn.predict(X_test_prepared)

precision_knn, recall_knn, fscore_knn, _knn = score(y_test_prepared, y_pred_knn, average='weighted')

print('precision', precision_knn)
print('recall', recall_knn)
print('fscore', fscore_knn)


knn_cm = confusion_matrix(y_test_prepared, y_pred_knn)
sns.heatmap(knn_cm, annot=True, fmt='d', cmap=mpl.cm.binary)
plt.show()


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
