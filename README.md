# Quantification 
##Install

Download parse_arff.py and quantification.py

#Examples:

>>> import numpy as np

>>> from quantification import Quantification

>>> X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])

>>> y = np.array([1, 1, 2, 2])

>>> q=Quantification(method='CC')

>>> q.fit(X, y)

>>> print(q.predict([[-0.8, -1],[0.9, 1.1]]))

[0.5, 0.5]

>>> print(q.score([[[-0.8, -1],[0.9, 1.1]]], [[1, 2]]))

0.0

#Methods:
## __init__(method='CC')

Parameter method: string, optional (default='CC'). It must be one of 'CC', 'ACC', 'PCC', 'PACC', 'EM', 'EM1', '', 'test'.

## fit(X, y)

Parameter X: Training set X

Parameter y: Training labels y

## predict(X, method='CC')

Parameter X: Test feature vector X

Parameter method: string, optional (default value from constructor). It must be one of 'CC', 'ACC', 'PCC', 'PACC', 'EM', 'EM1'.

Returns: predicted prevalence.

## score(list_of_X, list_of_y, method='CC')

Parameter list_of_X: Test set of feature vectors [X1, X2]

Parameter list_of_y: Test set of original labels [y1, y2]

Parameter method: string, optional (default value from constructor). It must be one of 'CC', 'ACC', 'PCC', 'PACC', 'EM', 'EM1'.

Returns: average over test sets value of Kullback-Leibler divergences between original and predicted prevalences.
np.average([KLD(pr_orig_1,pr_pred_1),KLD(pr_orig_2,pr_pred_2)])


#Install datasets RCV1 and OHSUMED

Download RCV1 and OHSUMED corps split for quantification experiments  http://hlt.isti.cnr.it/quantification/ in ARFF format to ./texts/QuantRCV1 and ./texts/QuantOHSUMED directories respectively.

#To convert ARFF datasets to Python format:

from parse_arff import Parse_ARFF

pa=Parse_ARFF()

pa.convert_arff(QuantOHSUMED, is_predict=False)

pa.convert_arff(QuantRCV1, is_predict=False)

q=Quantification(method='test')