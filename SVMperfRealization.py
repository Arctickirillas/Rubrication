__author__ = 'Kirill Rudakov'

from sklearn.datasets import make_classification
from sklearn.cross_validation import train_test_split
import subprocess
import numpy as np
import scipy
from quantify import CCforDouble


class SVMperf():
    def __init__(self,x_train,y_train,x_test,y_test):
        self.train = 'train.txt'
        self.test = 'test.txt'
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        # For automatic
        # self.getRepresentation(x_train,y_train,self.train)
        # self.getRepresentation(x_test,y_test,self.test)

        # self.model = self.fitSVMperf(self.train)
        # self.predictions = self.predictSVMperf(self.test,self.model)


    def getRepresentation(self, x, y, name = None):
        if name!=None:
            file = open(str(name), 'w')
        else:
            file = open('name.txt', 'w')
            print len(y)
        # type ndarray
        if type(x) == type(np.ndarray(None)):
            for i in range(len(y)):
                if y[i] == 1:
                    file.write('1 ')
                    for m in range(len(x[i])):
                        if x[i][m]!=0:
                            file.write(str(m+1)+':'+str(x[i][m])+' ')
                    file.write('\n')
                else:
                    file.write('-1 ')
                    for m in range(len(x[i])):
                        if x[i][m]!=0:
                            file.write(str(m+1)+':'+str(x[i][m])+' ')
                    file.write('\n')
            file.close()
        #  type csr_matrix
        elif type(x) == type(scipy.sparse.csr_matrix(None)):
            for i in range(len(y)):
                if y[i] == 1:
                    file.write('1 ')
                    _x = x.getrow(i).toarray()[0]
                    for j in range(len(_x)):
                        if _x[j]!=0:
                            file.write(str(j+1)+':'+str(_x[j])+' ')
                    file.write('\n')
                else:
                    file.write('-1 ')
                    _x = x.getrow(i).toarray()[0]
                    for j in range(len(_x)):
                        if _x[j]!=0:
                            file.write(str(j+1)+':'+str(_x[j])+' ')
                    file.write('\n')
            file.close()

    def fitSVMperf(self, trainData, model = 'model.txt'):
        subprocess.Popen(["svm-perf-original/svm_perf_learn","-c","20",trainData,model], stdout=subprocess.PIPE)
        return model

    def predictSVMperf(self, testData, model, predictions = 'predictions.txt'):
        self.description = subprocess.Popen(["svm-perf-original/svm_perf_classify",testData,model,predictions], stdout=subprocess.PIPE)
        return predictions

    def getDescriptionSVM(self):
        return self.description.communicate()[0]

    def getPredictions(self):
        q = []
        f = open(self.predictions,'r')
        for line in f:
            if float(line) >= 0 :
                q.append(1)
            elif float(line) < 0:
                q.append(-1)
        f.close()
        return  np.array(q)

    def getKLD(self,p, q):
        p = np.asarray(p, dtype=np.float)
        q = np.asarray(q, dtype=np.float)
        return np.sum(np.where(p != 0,p * np.log(p / q), 0))

def generate_data():
    X,y=make_classification(n_samples=10000, n_features=10, n_informative=3, n_classes=2, n_clusters_per_class=2)
    return train_test_split(X,y, test_size=0.75)



# EXAMPLE
# X_train, X_test, y_train, y_test = generate_data()
#
#
# s = SVMperf(X_train, y_train, X_test, y_test)
# print s.getPredictions()
#
# q = CCforDouble(s.getPredictions())
# p = CCforDouble(s.y_test)
#
# print 'REAL:',p
# print 'EST:',q
#
#
# print 'KLD:',s.getKLD(p,q)
#
# print s.getDescriptionSVM()


