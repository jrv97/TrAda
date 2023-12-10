from sklearn.linear_model import RidgeClassifier
from adapt.instance_based import TrAdaBoost
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

datat = pd.read_csv("data/mendeley.csv", header=0)
datas = pd.read_csv("data/DSA.csv", header=0)

yt = datat["label"]
Xt = datat.drop("label", axis=1)

ys = datas["label"]
Xs = datas.drop("label", axis=1)

Xs_train, Xs_test, ys_train, ys_test = train_test_split(
    Xs, ys, test_size=0.3, random_state=42, stratify=ys
)
Xt_train, Xt_test, yt_train, yt_test = train_test_split(
    Xt, yt, test_size=0.3, random_state=42, stratify=yt
)

clf = RidgeClassifier()
clf.fit(Xs_train, ys_train)
print(clf.score(Xs_test, ys_test))

clf2 = RidgeClassifier()
clf2.fit(Xt_train, yt_train)
print(clf2.score(Xt_test, yt_test))


model = TrAdaBoost(
    clf, n_estimators=10, Xt=Xt_train[:10], yt=yt_train[:10], random_state=0, verbose=0
)
model.fit(Xt_train[:10], yt_train[:10])
print(model.score(Xt_test, yt_test))

model2 = TrAdaBoost(
    clf2, n_estimators=10, Xt=Xs_train[:10], yt=ys_train[:10], random_state=0, verbose=0
)
model2.fit(Xs_train[:10], ys_train[:10])
print(model2.score(Xs_test, ys_test))

print(clf2.feature_names_in_)
print(model2)
