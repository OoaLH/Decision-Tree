from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import csr_matrix
import numpy as np
from sklearn.model_selection import train_test_split
import graphviz
def load_data():
    
    with open('clean_real.txt', 'r') as real:
        realdata=[lines.strip('\n') for lines in real]
        
    with open('clean_fake.txt', 'r') as fake:
        fakedata=[lines.strip('\n') for lines in fake]

    dataset=realdata+fakedata

    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(dataset)
    rlabels = np.ones((len(realdata), 1))
    flabels = np.zeros((len(fakedata), 1))
    labels = np.vstack((rlabels,flabels))
    train, other,ytrain,yother= train_test_split(X.toarray(),labels, test_size=0.3, random_state=22)
    validate, test,yvalidate,ytest= train_test_split(other,yother, test_size=0.5, random_state=22)
    return vectorizer,train, test, validate,ytrain,ytest,yvalidate
def select_model(sets, label, max_depths):
    ig=[]
    gini=[]
    for i in range(0,5):
        clf1 = DecisionTreeClassifier(max_depth=max_depths[i],random_state=22)
        ig.append(clf1.fit(sets,label))
        clf2 = DecisionTreeClassifier(max_depth=max_depths[i],criterion="entropy",random_state=22)
        gini.append(clf2.fit(sets,label))
    return ig,gini
[vectorizer,train,test,validate,ytrain,ytest,yvalidate]=load_data()
max_depths=[2,8,15,30,48]
[ig,gini]=select_model(train,ytrain,max_depths)
yvalidate=yvalidate.reshape(1,-1)
best=0
for i in range(0,5):
    k1=ig[i].predict(validate)
    k2=gini[i].predict(validate)
    A1=1-np.sum(abs(k1-yvalidate))/490
    A2=1-np.sum(abs(k2-yvalidate))/490
    print("ig",max_depths[i],A1)
    print("gini",max_depths[i],A2)
    if best<=max(A1,A2):
        if A1<=A2:
            clfbest=gini[i]
        else: 
            clfbest=ig[i]
        best=max(A1,A2)
dot_data = tree.export_graphviz(clfbest, max_depth=2,out_file=None,feature_names=vectorizer.get_feature_names(), class_names=["fake","real"],rounded=True,filled=True)
graph = graphviz.Source(dot_data)
graph.render("best")
