# anomaly-detection-in-log-files

## **Introduction** <br>
Computer-generated records, commonly known as logs, capture timestamped data as it relates to actions and decisions taken by applications, operating systems and devices. Businesses leverage this data to ensure that their applications and tools are fully operational and secure.
In this project, and as part of our Capstone project for the Master of Applied Data Science, will explore anomaly detection in application log data. This will be done similarly to the paper **Anomaly Detection for Application Log Data[1]**, but our approach will use _generalized feature extraction_ without using any log file specific parsers.

## **Approach** <br>
The question we will explore is how well does this generalized feature extraction work? That is, how do models trained on the generalized feature extraction output compare to models trained on the output of customized feature extraction used in other research studies.

Our approach is divided into two parts:
1. Supervised Learning: framed as a classification task, we trained a logistic regression model, a gradient boosted tree, and an XGBoost model to classify log lines as normal or anomalous.
2. Unsupervised Learning: in an attempt to detect anomalies in the log files, we made use of the K-means clustering algorith, one class SVM, and Isolation forest.

## **Data** <br>
BGL and Thunderbird data were used in this project. The data is provided by the Loghub collection:

Shilin He, Jieming Zhu, Pinjia He, Michael R. Lyu. Loghub: A Large Collection of System Log Datasets towards Automated Log Analytics. Arxiv, 2020.

Information on BGL can be found [here](https://github.com/logpai/loghub/tree/master/BGL)<br>
**Note**: the data in the _data folder_ is just a sample. The raw logs can be requested from Zenodo: https://doi.org/10.5281/zenodo.1144100 <br>

Information on Thunderbird can be found [here](https://github.com/logpai/loghub/tree/master/Thunderbird)<br>
**Note** the data in the _data folder_ is just a sample. The raw logs can be requested from Zenodo: https://doi.org/10.5281/zenodo.1144100




[1] Grover, Aarish, "Anomaly Detection for Application Log Data" (2018). Master's Projects. 635. DOI: https://doi.org/10.31979/etd.znsb-bw4d
## **How to run the code** <br>
- **ad_feature_extraction.py** preprocesses the data and generates the features used in the supervised and unsupervised machine learning sections of the project. The input to this file is the data listed above. Since generating this data takes a long time, you can download the generated files [here](https://drive.google.com/file/d/1IxKSUTRAW8z2uIYL4t5aOziK2TeLjS7-/view?usp=drive_link).
- **Supervised_learning** folder contains the jupyter notebooks that detail the various supervised learning models we ran during the course of the project.
- **unsupervised_learning** folder contains jupyter notebooks that detail the various unsupervised learning models we ran during the course of the project.
- **F1 Scores.ipynb** evaluates the performance of the previously selected models under various conditions
- **images** contains some of the visualizations and the code to generate them
*data* contains **samples** of the processed data. To get the full dataset, run the *ad_feature_extraction.py* on the raw dataset.
