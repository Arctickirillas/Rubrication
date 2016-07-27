# coding: utf-8
__author__ = 'Nikolay Karpov'
import scipy
import numpy as np

def split_by_topic(X_test, y_test, topics):
    #X_test=X_test.toarray()
    test_index, X_test_list, y_test_list = [], [], []
    utopics=np.unique(topics)
    for topic in utopics:
        test_index.append([])

    utopics_dict=dict(zip(utopics,range(len(utopics))))
    for topic,i in zip(topics,range(len(topics))):
        if topic in utopics_dict:
            test_index[utopics_dict[topic]].append(i)

    for index in test_index:
        X_test_list.append(X_test[index])
        y_test_list.append(y_test[index])
    return X_test_list, y_test_list, utopics

def read_semeval(fname='texts/2download/gold/dev/100_topics_100_tweets.sentence-three-point.subtask-A.dev.gold.tsv'):
    with open(fname, mode='r') as f:
        t=f.readlines()
        f.close()
    texts=[]
    marks=[]
    topics=[]
    ids=[]
    for line in t:
        substr=line.split('\t')
        texts.append(substr[-1])
        try:
            marks.append(int(substr[-2]))
        except:
            marks.append(substr[-2])
        ids.append(substr[0])
        if len(substr)>3:
            topics.append(substr[-3])
        else:
            topics.append(0)
    names=sorted(scipy.unique(marks))
    name_dict=dict(zip(names, range(len(names))))
    marks_num=[]
    for mark in marks:
        marks_num.append(name_dict[mark])
    return texts, marks_num, topics, ids

def write_semeval(pdict,fname):
    #print({val:key for key, val in cat_names.items()})
    pstr=''
    for key in pdict:
        pstr=pstr+'%s'%key
        for val in pdict[key]:
            st_val='%0.8f'%val
            pstr=pstr+'\t'+st_val[1:]
        pstr=pstr+'\n'
    with open('texts/2download/output/'+fname+'.pred.txt', 'w') as f:
        f.write(pstr)
        f.close()
    #print(pstr)

def getOutData(q, topic, name = 'out'):
    top = ''
    countOfTop = 1.
    neg = 0.
    pos = 0.
    file = open (str(name)+'.output','w')
    prev_list={}
    for i in range(len(q)):
        if top==topic[i]:
            if q[i]==1:
                pos += 1
            else:
                neg += 1
            countOfTop += 1
        else:
            if top!='':
                file.write(str(top)+'\t'+str(float(pos/countOfTop))+'\t'+str(float(neg/countOfTop)) + '\n')
                prev_list[str(top)]=[float(neg/countOfTop),float(pos/countOfTop)]
            top = topic[i]
            pos = 0.
            neg =0.
            if q[i]==1:
                pos += 1
            else:
                neg += 1
            countOfTop = 1
    file.write(str(top)+'\t'+str(float(pos/countOfTop))+'\t'+str(float(neg/countOfTop)) + '\n')
    prev_list[str(top)]=[float(neg/countOfTop),float(pos/countOfTop)]
    file.close()
    return prev_list

def readSentiScores(fname):
    with open(fname, mode='r') as f:
        t=f.readlines()
        f.close()
        texts=[]
    pos=[]
    neg=[]
    for line in t:
        substr=line.split('\t')
    #    print(substr[0])
        if len(substr)>1:
            try:
                pos.append(float(substr[0]))
                neg.append(float(substr[1]))
            except:
                pos.append(0)
                neg.append(0)
    #print(pos)
    return pos, neg

def bayes_calc(c_prev, c_prob):
    res=[]
    for i in range(len(c_prob)):
            summ=0
            temp1=c_prob[i]
            temp_res=[]
            for j in range(len(c_prev)):
                summ+=c_prev[j]*temp1[j]
            for j in range(len(c_prev)):
                temp_res.append(c_prev[j]*temp1[j]/summ)
            res.append(temp_res)
    return res

def write_TaskA(fname, task_in_process, ids, label):
    #id<TAB>label
    # where "label" can be 'positive', 'neutral' or 'negative'.
    pstr=""
    for i in range(len(ids)):
        pstr=pstr+str(ids[i])+'\t'+str(label[i])
        pstr=pstr+'\n'

    with open('texts/2download/output/'+fname+".scores"+task_in_process, 'w') as f:
        f.write(pstr)
        f.close()
    return pstr

def write_TaskBC(fname, task_in_process, ids, topic, label):
    #  id<TAB>topic<TAB>label
    #where "label" can be 'positive' or 'negative' (note: no 'neutral'!).
    pstr=""
    for i in range(len(ids)):
        pstr=pstr+str(ids[i])+'\t'+topic[i]+'\t'+str(label[i])
        pstr=pstr+'\n'

    with open('texts/2download/output/'+fname+".scores"+task_in_process, 'w') as f:
        f.write(pstr)
        f.close()
    return pstr


def write_TaskDE(fname, task_in_process, topics, prob):
#topic<TAB>positive<TAB>negative
#where positive and negative are floating point numbers between 0.0 and 1.0 and positive + negative must sum to 1.0
#where label-2 to label2 are floating point numbers between 0.0 and 1.0 and the five numbers must sum to 1.0. label-2 corresponds to the fraction of tweets labeled -2 in the data and so on.
    pstr=""
    for i in range(len(topics)):
        pstr=pstr+str(topics[i])
        for j in range(len(prob[0])):
            pstr=pstr+'\t'+str(prob[i][j])
        pstr=pstr+'\n'

    with open('texts/2download/output/'+fname+".scores"+task_in_process, 'w') as f:
        f.write(pstr)
        f.close()
    return pstr

def prob_to_scale(c_prob):
    scale_five=[-2,-1,0,1,2]
    scale_three=["negative","neutral","positive"]
    scale_PN=["negative", "positive"]
    scale=[0,1,scale_PN,scale_three,0,scale_five]
    res=[]
    res1=[]
    for i in range(len(c_prob)):
        res.append(scale[len(c_prob[i])][c_prob[i].index(max(c_prob[i]))])
        res1.append(c_prob[i].index(max(c_prob[i])))
    return res, res1

def semEvalP():
    task_in_process="A"
    #task_in_process="B"
    #task_in_process="C"
    #task_in_process="D"
    #task_in_process="E"
    if (task_in_process=="A"):
        fname='100_topics_100_tweets.sentence-three-point.subtask-A'
    elif (task_in_process=="C" or task_in_process=="E"):
        fname='100_topics_100_tweets.topic-five-point.subtask-CE'
    elif (task_in_process=="B"or task_in_process=="D"):
        fname='100_topics_XXX_tweets.topic-two-point.subtask-BD'

    train=read_semeval('texts/2download/gold/train/'+fname+'.train.gold.tsv')


    tp=text_processing(n=1)
    X_train=tp.fit_transform(train[0])#.toarray()
    y_train=np.asarray(train[1])

    test=read_semeval('texts/2download/gold/devtest/'+fname+'.devtest.gold.tsv')

    X_test=tp.transform(test[0])#.toarray()
    y_test=np.asarray(test[1])

    X_test_list, y_test_list, utopics=split_by_topic(X_test, y_test, test[2])

    q=Quantification(method='Iter1',is_clean=True)
    q.fit(X_train,y_train)
    prev=q.predict(X_test)
    #print(prev)

    if (task_in_process=="C"or task_in_process=="D"):
        prevED=[]
        for i in range(len(utopics)):
            prevED.append(q.predict(X_test_list[i]))
            #print(utopics[i])
            #print(prevED)

    #print (prev[1])
    # вывод результатов
    #write_semeval(dict(zip(utopics,prev)))
    svm=SVC(probability=True)
    svm.fit(X_train,y_train)
    prob=svm.predict_proba(X_test)

    svmL=linear_model.LogisticRegression()
    svmL.fit(X_train,y_train)
    probL=svm.predict_proba(X_test)

    for i in range(len(prob)):
        for j in range(len(prob[0])):
            prob[i][j]=(2*prob[i][j]+probL[i][j])/3

    #Вывод результатов в файл без учета априорных вероятностей
    new_labels, test_labels=prob_to_scale(prob.tolist())
    res=metrics.classification_report(y_test, test_labels)
    print(res)
    #Учет априорных вероятностей попадания в класс:
    b_prob=bayes_calc(prev, prob)
    new_labels, test_labels=prob_to_scale(b_prob)
    #s=write_TaskA(test[3], new_labels)

    if (task_in_process=="B"or task_in_process=="C"):
        new_labels=[]
        for i in range(len(utopics)):
            prob=svm.predict_proba(X_test_list[i])
            new_labels, test_labels= prob_to_scale(prob.tolist())
            res=metrics.classification_report(y_test_list[i], test_labels)
            print(res)
            print("Added Q")
            b_prob=bayes_calc(prevED[i], prob)
            new_labels,test_labels= prob_to_scale(b_prob)
            #new_labels.append()
            res=metrics.classification_report(y_test_list[i], test_labels)
            print(res)

    if (task_in_process=="A"):
        s=write_TaskA(fname, task_in_process, test[3], new_labels)
        res=metrics.classification_report(y_test, test_labels)
        print(res)
    elif (task_in_process=="B" or task_in_process=="C"):
        s=1
        #s=write_TaskBC(fname,task_in_process, test[3], test[2], new_labels)
    elif (task_in_process=="D" or task_in_process=="E"):
        s=write_TaskDE(fname,task_in_process, utopics.tolist(), prevED)
    #print(s)

def read_and_swop():
    name='90X.scoring_test_data.topic-two-point.subtask-BD'
    with open('texts/2download/output/'+name+'.pred.txt', 'r') as f:
        pstr=f.read()
        f.close()
    dict1={}
    for line in pstr.split('\n'):
        mas=line.split('\t')
        if len(mas)>1:
            dict1[mas[0]]=[float(mas[2]),float(mas[1])]
    #print(dict1)
    write_semeval(dict1,name+'.new')
