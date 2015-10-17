# Quantification 

Download parse_arff.py and quantification.py

Download RCV1 and OHSUMED corps split for quantification experiments  http://hlt.isti.cnr.it/quantification/ in ARFF format to ./texts/QuantRCV1 and ./texts/QuantOHSUMED directories respectively.

#To convert ARFF datasets to Python:

from parse_arff import Parse_ARFF

pa=Parse_ARFF()

pa.convert_arff_and_predict_proba(QuantOHSUMED, is_predict=False)

pa.convert_arff_and_predict_proba(QuantRCV1, is_predict=False)

#To count Quantifiacation score:
##Warning! Processing can take a long time. We recommend to perform it step by step

q=Quantification('QuantOHSUMED')

q.process_pipeline()

#To count Quantifiacation score step by step:

from quantification import Quantification

name='QuantOHSUMED' #QuantRCV1 or QuantOHSUMED

q=Quantification(name)

indexes=q.estimate_cl_indexes()

indexes=q._read_pickle('texts/cl_indexes_'+name+'.pickle')

td=q.classify_and_count(indexes[1])

ed1=q.classify_and_count(indexes[2])

ed2=q.adj_classify_and_count(indexes)


q.estimate_cl_prob()

q.unite_cl_prob()

prob=q._read_pickle('texts/cl_prob_'+name+'.pickle')


ed3=q.classify_and_count(prob[2], is_prob=True)

ed4=q.prob_classify_and_count(prob)

ed5, num_iter=q.expectation_maximization(prob)

q.count_diff(td,ed4)

q.count_diff1(td,ed5, num_iter)
