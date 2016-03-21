import pandas as pd
import numpy as np
# import pylab as pyl

from sklearn import cross_validation, datasets, preprocessing
from sklearn import linear_model, svm


boston_dataset = datasets.load_boston()
#print boston_dataset.data.shape
#print boston_dataset.keys()

boston_data = boston_dataset.data
boston_feature_names = boston_dataset.feature_names
boston_target = boston_dataset.target
n_samples = boston_data.shape[0]

boston_data_scaled = preprocessing.scale(boston_data)

boston_df = pd.DataFrame(data = boston_data_scaled, columns=list(boston_feature_names))
boston_df["target"] = boston_target
#print boston_df.head(15)

# boston_df.plot(kind="scatter", x="LSTAT", y="target")  #pd plot
# pyl.show()


myseed = 1234

lm = linear_model.LinearRegression()
en = linear_model.ElasticNet(random_state=myseed)
rid = linear_model.Ridge(random_state=myseed)
svr_lin = svm.SVR(kernel='linear')
svr_rbf = svm.SVR(kernel='rbf', gamma=0.7)
svr_pl3 = svm.SVR(kernel='poly', degree=3)
models = [lm, en, rid, svr_lin, svr_rbf, svr_pl3]

cv_kf5 = cross_validation.KFold(n=n_samples, n_folds=5, shuffle=True, random_state=myseed)
cv_loo = cross_validation.LeaveOneOut(n=n_samples)
cv_lpo10 = cross_validation.LeavePOut(n=n_samples, p=2)
cvs = [cv_kf5, cv_loo, cv_lpo10]


for model in models:
    for validator in cvs:
        scores = cross_validation.cross_val_score(model, boston_data_scaled, boston_target, cv=validator)
        print "Accuracy: %0.2f (+/- %0.2f), cvs: %0.2f" % ( scores.mean(), scores.std() * 2 , len(scores) )
    print "--"
