import pandas as pd
import matplotlib.pyplot as plt
import pylab as pyl
from sklearn import datasets, cross_validation, svm


myseed = 1000
samples = 500
noise = 0.3
test_proportion = 0.2

moons_x, moons_y = datasets.make_moons(n_samples=samples, noise=noise, random_state=myseed)

moons_data = pd.DataFrame(data = moons_x, columns=["x1","x2"])
moons_data["target"] = moons_y

X_train, X_test, y_train, y_test = cross_validation.train_test_split(moons_data[["x1","x2"]], moons_data["target"], test_size=test_proportion, random_state=myseed)

svc_rbf = svm.SVC(kernel='rbf', gamma=0.7, random_state=myseed)
svc_rbf.fit(X_train, y_train)

y_test = y_test.to_frame(name="test_actual")
y_test["test_predict"] =  svc_rbf.predict(X_test)
y_test["check"] = (y_test["test_predict"] == y_test["test_actual"])

correct = y_test[y_test.check == True].shape[0]
perc_correct = (100.0 * correct/ X_test.shape[0])

print "train size = %s. test size = %s. test success = %s" %( X_train.shape[0], X_test.shape[0], perc_correct )

# draw all training dots with colour
fill_colormap = {0: (0,0.6,0), 1: (0,0,1)}
fill_color_train = [fill_colormap[y] for y in y_train]
fill_color_test = [fill_colormap[y] for y in y_test.test_actual]

line_colormap = {True: (0,0,0), False: (1,0,0)}
line_color_test =[line_colormap[y] for y in y_test.check]

plt.scatter( x=X_train.x1, y=X_train.x2, c=fill_color_train, alpha=0.7, lw=0, s=20 )
plt.scatter( x=X_test.x1, y=X_test.x2, c=fill_color_test, alpha=1, edgecolor=line_color_test, s=30 )

# plt.axis('off')
pyl.show()
print 'image..'
