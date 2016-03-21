import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
#import pylab as pyl

from sklearn import cross_validation, datasets
from sklearn import linear_model, svm, neighbors


iris_dataset = datasets.load_iris()
# print iris_dataset.data.shape
# print iris_dataset.keys()

iris_data = iris_dataset.data
iris_feature_names = iris_dataset.feature_names
iris_target = iris_dataset.target
iris_target_names = iris_dataset.target_names
n_samples = iris_data.shape[0]

# need to regularise the data

iris_df = pd.DataFrame(data = iris_data, columns=list(iris_feature_names))
iris_df["target"] = iris_target


# scatter plots
#plt.scatter(x = iris_df["sepal length (cm)"], y = iris_df["sepal width (cm)"], c = iris_df["target"])  #plt plot
#iris_df.plot(kind="scatter", x="sepal length (cm)", y="sepal width (cm)", c = iris_df["target"])  #pd plot

# box plots
#plt.boxplot(iris_df, iris_df["target"])  #?? not sure how to get this to work
#iris_df.boxplot(by = "target", layout = (1,4))

#pyl.show()


myseed = 1234

logreg = linear_model.LogisticRegression()
svc_lin = svm.SVC(kernel='linear', random_state=myseed)
svc_rbf = svm.SVC(kernel='rbf', gamma=0.7, random_state=myseed)
svc_pl3 = svm.SVC(kernel='poly', degree=3, random_state=myseed)
knn3 = neighbors.KNeighborsClassifier(n_neighbors=3)
knn5 = neighbors.KNeighborsClassifier(n_neighbors=5)
models = [logreg, svc_lin, svc_rbf, svc_pl3, knn3, knn5]

cv_kf5 = cross_validation.KFold(n=n_samples, n_folds=5, shuffle=True, random_state=myseed)
cv_loo = cross_validation.LeaveOneOut(n=n_samples)
cv_lpo10 = cross_validation.LeavePOut(n=n_samples, p=2)
cvs = [cv_kf5, cv_loo, cv_lpo10]


for model in models:
    for validator in cvs:
        scores = cross_validation.cross_val_score(model, iris_data, iris_target, cv=validator)
        print "Accuracy: %0.2f (+/- %0.2f), cvs: %0.2f" % ( scores.mean(), scores.std() * 2 , len(scores) )
    print "--"


# X_train, X_test, y_train, y_test = cross_validation.train_test_split(iris_data, iris_target, test_size=0.4, random_state=myseed)
# logreg.fit(X_train, y_train)
# print logreg.score(X_test, y_test)
