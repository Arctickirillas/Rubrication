# Quantification 
##Install

Download parse_arff.py and quantification.py

#To count quantifiacation score:

##q=Quantification(method='CC')

Parameter method: string, optional (default='CC'). It must be one of 'CC', 'ACC', 'PCC', 'EM', 'EM1', '', 'test'.

##q.fit(X, y)

Parameter X: Training set X

Parameter y: Training labels y

##q.predict(X, method='CC')

Parameter X: Test feature vector X

Parameter method: string, optional (default value from constructor). It must be one of 'CC', 'ACC', 'PCC', 'EM', 'EM1', ''.

Returns predicted prevalence.

##q.score(list_of_X, list_of_y, method='CC')

Parameter list_of_X: Test set of feature vectors [X1, X2]

Parameter list_of_y: Test set of original labels [y1, y2]

Returns average over test sets value of Kullback-Leibler divergences between original and predicted prevalences.
np.average([KLD(pr_orig_1,pr_pred_1),KLD(pr_orig_2,pr_pred_2)])

Parameter method: string, optional (default value from constructor). It must be one of 'CC', 'ACC', 'PCC', 'EM', 'EM1', ''.


#Install dataset RCV1 and OHSUMED

Download RCV1 and OHSUMED corps split for quantification experiments  http://hlt.isti.cnr.it/quantification/ in ARFF format to ./texts/QuantRCV1 and ./texts/QuantOHSUMED directories respectively.

#To convert ARFF datasets to Python format:

from parse_arff import Parse_ARFF

pa=Parse_ARFF()

pa.convert_arff(QuantOHSUMED, is_predict=False)

pa.convert_arff(QuantRCV1, is_predict=False)
