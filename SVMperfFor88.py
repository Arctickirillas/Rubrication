__author__ = 'Kirill Rudakov'

import SVMperfRealization as s
import pickle as p


# Obtaining Train Data From Dump
file = open('/Users/irina/Python/MyNLP/pickle_QuantOHSUMED/quant_OHSUMED_train.arff.pickle','rb')
pickle_file = p.load(file)
file.close()
# Splitting Into Trained, Predicted and Label
x_train = pickle_file[0] # x_train
y_train = pickle_file[1] # y_train
class_name = pickle_file[2]
y_train = y_train.T.toarray()


# Obtaining Test Data From Dump
file = open('/Users/irina/Python/MyNLP/pickle_QuantOHSUMED/quant_OHSUMED_test_88.arff.pickle','rb')
pickle_file = p.load(file)
file.close()
# Splitting Into
x_test = pickle_file[0] # x_test
y_test = pickle_file[1] # y_test
y_test = y_test.T.toarray()

svm = s.SVMperf(x_train, y_train[0], x_test, y_test[0])
# def ObtainDataRepresentationForSVMperf
# !!!!IF FILES NOT EXIST!!!
def obtainDataRepresentationForSVMperf():
    for i in range(88):
        print 'ObtainDataRepresentation Iterations:',i
        svm = s.SVMperf(x_train, y_train[i], x_test, y_test[i])
        svm.getRepresentation(x_train,y_train[i],'/Users/irina/Python/MyNLP/Data/train/trainData_'+str(i)+'.txt')
        svm.getRepresentation(x_test,y_test[i],'/Users/irina/Python/MyNLP/Data/test/testData_'+str(i)+'.txt')

# def GetModel
# !!!!IF FILES NOT EXIST!!!
def getModel():
    for i in range(88):
        print 'GetModel Iterations:',i
        train = '/Users/irina/Python/MyNLP/Data/train/trainData_'+str(i)+'.txt'
        svm.fitSVMperf(train,'/Users/irina/Python/MyNLP/SVMperf88/model/model_'+str(i)+'.txt')

# def PredictSVMperf
# !!!!IF FILES NOT EXIST!!!

def predictSVMperf():
    for i in range(88):
        print 'PredictSVMperf Iterations:',i
        model = '/Users/irina/Python/MyNLP/SVMperf88/model/model_'+str(i)+'.txt'
        for j in range(88):
            test =  '/Users/irina/Python/MyNLP/Data/test/testData_'+str(j)+'.txt'
            predictions = '/Users/irina/Python/MyNLP/SVMperf88/prediction/prediction_'+str(i)+'_'+str(j)+'.txt'
            svm.predictSVMperf(test,model,predictions)
            file = open('/Users/irina/Python/MyNLP/SVMperf88//report/report_'+str(i)+'_'+str(j),'wr')
            file.write(svm.getDescriptionSVM())
            file.close()

predictSVMperf()