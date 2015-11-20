# Quantification 
##Install

Download parse_arff.py and quantification.py

#To count quantifiacation score:

q=Quantification(method='CC')

Parameter method: string, optional (default='CC'). It must be one of ‘CC’, ‘PCC’, ‘EM’.

q.fit(X, y)
	
prev_pred=q.predict(X)
	
avg_KLD=q.score([X], [y])


#Install dataset RCV1 and OHSUMED

Download RCV1 and OHSUMED corps split for quantification experiments  http://hlt.isti.cnr.it/quantification/ in ARFF format to ./texts/QuantRCV1 and ./texts/QuantOHSUMED directories respectively.

#To convert ARFF datasets to Python format:

from parse_arff import Parse_ARFF

pa=Parse_ARFF()

pa.convert_arff(QuantOHSUMED, is_predict=False)

pa.convert_arff(QuantRCV1, is_predict=False)
