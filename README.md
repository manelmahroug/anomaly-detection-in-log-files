# anomaly-detection-in-log-files

Computer-generated records, commonly known as logs, capture timestamped data as it relates to actions and decisions taken by applications, operating systems and devices. Businesses leverage this data to ensure that their applications and tools are fully operational and secure.
In this project, and as part of our Capstone project for the Master of Applied Data Science, will explore anomaly detection in application log data. This will be done similarly to the paper **Anomaly Detection for Application Log Data[1]**, but our approach will use _generalized feature extraction_ without using any log file specific parsers.

The question we will explore is how well does this generalized feature extraction work? That is, how do models trained on the generalized feature extraction output compare to models trained on the output of customized feature extraction used in other research studies.

Our approach is divided into two parts:
1. Supervised Learning: framed as a classification task, we trained a logistic regression model, a gradient boosted tree, and an XGBoost model to classify log lines as normal or anomalous.
2. Unsupervised Learning: in an attempt to detect anomalies in the log files, we made use of the K-means clustering algorith, one class SVM, and Isolation forest.

## **Data** <br>
BGL and Thunderbird data were used in this project. The data is provided by the Loghub collection:

Shilin He, Jieming Zhu, Pinjia He, Michael R. Lyu. Loghub: A Large Collection of System Log Datasets towards Automated Log Analytics. Arxiv, 2020.

Information on BGL can be found [here](https://github.com/logpai/loghub/tree/master/BGL)<br>
Information on Thunderbird can be found [here](https://github.com/logpai/loghub/tree/master/Thunderbird)<br>




[1] Grover, Aarish, "Anomaly Detection for Application Log Data" (2018). Master's Projects. 635. DOI: https://doi.org/10.31979/etd.znsb-bw4d

