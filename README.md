# Quantification 
##Install

Download parse_arff.py and quantification.py

#Examples:

import numpy as np

from quantification import Quantification

X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])

y = np.array([1, 1, 2, 2])

q=Quantification(method='CC')

q.fit(X, y)

print(q.predict([[-0.8, -1],[0.9, 1.1]]))

> [0.5, 0.5]

print(q.score([[-0.8, -1],[0.9, 1.1]], [1, 2]))

> 0.0

#Methods:
## __init__(method='CC')

Parameter method: string, optional (default='CC'). It must be one of 'CC', 'ACC', 'PCC', 'PACC', 'EM', 'EM1', 'Iter', 'Iter1', '', 'test'.

## fit(X, y)

Parameter X: Training set X

Parameter y: Training labels y

## predict(X, method='CC')

Parameter X: Test feature vector X

Parameter method: string, optional (default value from constructor). It must be one of 'CC', 'ACC', 'PCC', 'PACC', 'EM', 'EM1'.

Returns: predicted prevalence.

## score(X, y, method='CC')

Parameter X: Test feature vectors X

Parameter y: Test labels y

Parameter method: string, optional (default value from constructor). It must be one of 'CC', 'ACC', 'PCC', 'PACC', 'EM', 'EM1'.

Returns: average value of Kullback-Leibler divergences between original and predicted prevalences. KLD(pr_orig,pr_pred)


#Install datasets RCV1 and OHSUMED

Download RCV1 and OHSUMED corps split for quantification experiments  http://hlt.isti.cnr.it/quantification/ in ARFF format to ./texts/QuantRCV1 and ./texts/QuantOHSUMED directories respectively.

#To convert ARFF datasets to Python format:

from parse_arff import Parse_ARFF

pa=Parse_ARFF()

pa.convert_arff(QuantOHSUMED, is_predict=False)

pa.convert_arff(QuantRCV1, is_predict=False)

q=Quantification(method='test')