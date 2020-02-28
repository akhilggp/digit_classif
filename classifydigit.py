import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import svm
digits=datasets.load_digits()
print(digits.data[0])
s=svm.SVC(gamma=0.001,C=100)
x,y=digits.data[:-1],digits.target[:-1]
s.fit(x,y)
print("prediction: {}".format(s.predict([digits.data[10]])))
plt.imshow(digits.images[10],cmap=plt.cm.gray_r,interpolation="nearest")
plt.show()


