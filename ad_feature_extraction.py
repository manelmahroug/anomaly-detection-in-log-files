#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 12:46:31 2023

@author: tps
"""
import os
import re
import csv
import pandas as pd
import numpy as np
import collections
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score
import uuid
import matplotlib.pyplot as plt
import math


BASE_DIR = '/home/thanuja/Dropbox/capstone/raw_files'
SUB_DIR = ['BGL','Hadoop','OpenStack']
OUTPUT_DIR = '/home/thanuja/Dropbox/capstone/output2/'
WINDOW_SIZE = 100
ANOMALY_DIR = '/home/thanuja/Dropbox/capstone/anomalies'

# Anomalies for Hadoop are labeled based on the file name. Anomalies for
# OpenStack are based on vm instance id. Creates a set for both log types to
# label anomalous log lines later.
def mark_anomalies():
    hadoop_anomalies = set()
    open_stack_anomalies = set()
    for root, dirs, files in os.walk(ANOMALY_DIR):
       for file in files:
            f_name = os.path.join(root, file)
            print(f_name)
            with open(f_name, "r+") as anomaly_file:
                if 'Hadoop' in f_name:
                    for line in anomaly_file:
                        hadoop_anomalies.add(line.strip())
                elif 'OpenStack' in f_name:
                    for line in anomaly_file:
                        if len(line.strip()) == 0:
                            continue
                        open_stack_anomalies.add(line.strip())
    return hadoop_anomalies, open_stack_anomalies

            
hadoop_anomalies, open_stack_anomalies = mark_anomalies()

# parses timestamp and writes intial cluster CSVs
def process_raw_files():
    backup_ext = str(uuid.uuid1())
    if os.path.isdir(OUTPUT_DIR):
        os.rename(OUTPUT_DIR, OUTPUT_DIR[:-1] + '.' + backup_ext)
    os.makedirs(OUTPUT_DIR)
    recursive_files = list(os.walk(BASE_DIR))
    recursive_files.sort()
    csvs_with_ts = set()
    for root, dirs, files in recursive_files:
        for file in files:
            if file.endswith('.log'):
                file_name = os.path.join(root,file)
                print(file_name)
                app_sys_name = re.search(r'raw_files/([^/]+)/',file_name).group(1)
                csvfile_with_ts = parse_ts(file_name, app_sys_name)
                csvs_with_ts.add(csvfile_with_ts)
    for csvfile_with_ts in csvs_with_ts:
        get_initial_clusters(csvfile_with_ts)

# processes all cluster output files to create a sliding window file for each.
def apply_sliding_window():
   out_files = os.listdir(OUTPUT_DIR)
   for file in out_files:
       if not file.endswith('_params.csv'):
           continue
       print(file)
       sliding_window(OUTPUT_DIR + file)

# counts number of clusters in each window of the sliding window. Also records
# the length of time of each window, and the row numbers of the original csv.
def sliding_window(csv_with_clusters):
    CLUSTER_COL=5
    w = 100
    de = collections.deque([], w+1)
    k = 0
    with open(csv_with_clusters, 'r') as ip_file:
        reader = csv.reader(ip_file)
        for row in reader:
            lst_row = list(row)
            cluster_str = lst_row[CLUSTER_COL]
            if not cluster_str.isnumeric():
                continue
            cluster = int(cluster_str)
            if cluster > k:
                k = cluster
    cluster_counts = [0]*(k+7)
    outfile = csv_with_clusters.replace('_params.csv', '_sliding_window.csv')
    with open(csv_with_clusters, 'r') as ip_file, open(outfile, 'w+') as sw_file:
        reader = csv.DictReader(ip_file)
        param_labels = [col for col in reader.fieldnames if col.startswith('p-')]
        csvwriter = csv.writer(sw_file)
        cluster_labels = ['cluster_' + str(i) for i in range(k+1)]
        cluster_labels += ['label', 'row_start', 'row_end', 'time_length', 'timestamp', 'filename']
        cluster_labels += param_labels
        csvwriter.writerow(cluster_labels)
        for row in reader:
            lst_row = list(row.values())
            cluster_str = lst_row[CLUSTER_COL]
            if not cluster_str.isnumeric():
                continue
            cluster = int(cluster_str)
            de.append(row)
            cluster_counts[cluster] += 1
            if len(de) <= w:
                continue
            row_old = de.popleft()
            lst_row_old = list(row_old.values())
            d_ts = float(row['timestamp'])-float(row_old['timestamp'])
            cluster_counts[int(row_old['clusters'])] -= 1
            #cluster_counts[-1]=lst_row[-2]
            cluster_counts[-1]=row['filename']
            cluster_counts[-2]= row['timestamp']
            cluster_counts[-3]= d_ts
            #cluster_counts[-4]= lst_row[0]
            cluster_counts[-4]= lst_row[0]
            #cluster_counts[-5]= row_old[0]
            cluster_counts[-5]= lst_row_old[0]
            cluster_counts[-6]= row['label']
            
            p_agg = []
            for p in param_labels:
                total = 0
                count = 0
                for p_row in de:
                    if not p in p_row or not p_row[p].replace('.', '').isnumeric():
                        continue
                    try:
                        total += float(p_row[p])
                        count += 1
                    except ValueError:
                        continue
                if count > 0:
                    p_agg.append(total / count)
                else:
                    p_agg.append('')
            csvwriter.writerow(cluster_counts + p_agg)

# parses a single line of a hadoop log file
def parse_hadoop_files(file_path, line):
    is_anomaly = False
    words = line.split(' ')
    if len(words) < 2:
        raise ValueError()
    folder_name = file_path[-2]
    is_anomaly = folder_name in hadoop_anomalies
    w1 = words.pop(0)
    w2 = words.pop(0)
    str_ts = w1+'-'+w2
    epochts = datetime.strptime(str_ts,'%Y-%m-%d-%H:%M:%S,%f').timestamp()*1000        
    remainder_ll = ' '.join(words)
    return epochts, remainder_ll, is_anomaly, folder_name

# parses a single line of an openstack log file
def parse_openstack_files(file_path, line):
    words = line.split(' ')
    if len(words) < 2:
        raise ValueError()
    filename = words.pop(0)
    w1 = words.pop(0)
    w2 = words.pop(0)
    str_ts = w1+'-'+w2
    epochts = datetime.strptime(str_ts,'%Y-%m-%d-%H:%M:%S.%f').timestamp()*1000 
    remainder_ll = ' '.join(words)
    is_anomaly = False
    for ele in open_stack_anomalies:
        if ele in remainder_ll:
            is_anomaly = True
    return epochts, remainder_ll, is_anomaly, filename

# parses a single line of a BGL log line
def parse_bgl_files(file_path, line):
    is_anomaly = True
    words = line.split(' ')
    if len(words) < 1:
        raise ValueError()
    if words[0]=='-':
        is_anomaly = False
    if len(words) < 4:
        raise ValueError()
    if is_anomaly:
        pass
    words.pop(0)
    words.pop(0)
    words.pop(0)
    words.pop(0)
    str_ts = words.pop(0)
    epochts = datetime.strptime(str_ts,'%Y-%m-%d-%H.%M.%S.%f').timestamp()*1000        
    machine = words.pop(0)
    remainder_ll = ' '.join(words)
    return epochts, remainder_ll, is_anomaly, machine

parsers = {'Hadoop':parse_hadoop_files,'OpenStack':parse_openstack_files,'BGL':parse_bgl_files}

# parses a raw log file. Log parsing specific to each log file is done in the
# appropriate handler, such as parse_bgl_files.
def parse_ts(file_name,app_sys_name):
    log_file = open(file_name, "r+")
    out_file = OUTPUT_DIR+app_sys_name+'.csv'
    is_file = os.path.isfile(out_file)
    csv_file = open(out_file, 'a+')
    file_path = file_name.split('/')
    with csv_file, log_file:
        csvwriter = csv.writer(csv_file, quoting=csv.QUOTE_NONNUMERIC, escapechar='\\')
        if not is_file:
            csvwriter.writerow(['timestamp','text','label','filename'])
        count = 0
        remainder_ll = is_anomaly = filename = None
        line_out = ''
        for line in log_file:
            line = line.rstrip('\n')
            count+=1
            if count%10000==0:
                print(count)
            if app_sys_name not in parsers:
                raise ValueError("no parser found for " + app_sys_name)
            run_parser = parsers[app_sys_name]
            remainder_ll = ''
            try:
                epochts,remainder_ll,is_anomaly,filename = run_parser(file_path, line)
            except ValueError:
                line_out += line
                continue
            if len(remainder_ll.strip()) == 0:
                continue
            csvwriter.writerow([epochts,line_out,is_anomaly,filename])
            line_out = remainder_ll
    log_file.close()
    return out_file

def simple_split(df):
    split_on = int(len(df.values)*0.80)
    train = df.values[:split_on]
    test = df.values[split_on:]
    train_df = pd.DataFrame(data=train, columns=df.columns)
    test_df = pd.DataFrame(data=test, columns=df.columns)
    return train_df, test_df

# cluster the tfidf vectors for a each raw log line, to apply cluster labels that
# correspond to the different types of log lines.
def get_initial_clusters(csv_log_file):
    print('finding clusters for ', csv_log_file)
    csv_log_df = pd.read_csv(csv_log_file).dropna()
    csv_log_df.sort_values('timestamp', inplace=True)
    # pull anomalies out to their own df and put 80% in train and 20% in test.
    normal_df = csv_log_df[csv_log_df["label"] == False]
    normal_train_df, normal_test_df = simple_split(normal_df)
    anomaly_df = csv_log_df[csv_log_df["label"] == True]
    anomaly_train_df, anomaly_test_df = simple_split(anomaly_df)
    train_df = pd.concat([normal_train_df, anomaly_train_df], axis=0)
    train_df.sort_values('timestamp', inplace=True)
    train_df.reset_index(inplace=True, drop=True)
    test_df = pd.concat([normal_test_df, anomaly_test_df], axis=0)
    test_df.sort_values('timestamp', inplace=True)
    test_df.reset_index(inplace=True, drop=True)
    
    out_train_file = os.path.splitext(csv_log_file)[0]+'_train_clusters.csv'
    out_test_file = os.path.splitext(csv_log_file)[0]+'_test_clusters.csv'
    tvec = TfidfVectorizer(min_df=100, tokenizer=simple_tokenizer)
    print(train_df.text[:5])
    tvec_weights_train = tvec.fit_transform(train_df.text)
    tvec_weights_test = tvec.transform(test_df.text)
    print('tvec_vocab: ',len(tvec.get_feature_names_out()))
    max_smallest_cluster_size = 10
    smallest_cluster_size = 1000
    max_rows = 100000
    num_rows = tvec_weights_train.shape[0]
    best_score = 0
    best_kmeans = None
    k = 10
    if num_rows > max_rows:
        tvec_weights_sample = tvec_weights_train[np.random.choice(num_rows, max_rows, replace=False), :]
    else:
        tvec_weights_sample = tvec_weights_train
    # automatically determining optimal k. Repeat k-means until small_cluster_size < max_smallest_cluster_size
    while smallest_cluster_size > max_smallest_cluster_size:
        kmeans = KMeans(n_clusters=k, n_init=5, algorithm='elkan').fit(tvec_weights_sample)
        labels = kmeans.predict(tvec_weights_sample)
        dense_weights = tvec_weights_sample.toarray()
        ch_score = calinski_harabasz_score(dense_weights, labels)
        db_score = davies_bouldin_score(dense_weights, labels)
        score = math.log(k) * ch_score / (10000 * db_score * db_score)
        if best_kmeans is None or best_score < score:
            best_score = score
            best_kmeans = kmeans
        smallest_cluster_size = min(np.bincount(kmeans.labels_))
        print('k', k, 'smallest cluster', smallest_cluster_size, 'ch', ch_score, 'db', db_score, 'score', score)
        k+=10
    train_labels = best_kmeans.predict(tvec_weights_train)
    clusters_train_df = pd.DataFrame({'clusters': train_labels})
    result_train_df = pd.concat([train_df, clusters_train_df], axis=1, join='inner')
    result_train_df.to_csv(out_train_file, quoting=csv.QUOTE_NONNUMERIC)

    test_labels = best_kmeans.predict(tvec_weights_test)
    clusters_test_df =  pd.DataFrame({'clusters': test_labels})
    result_test_df = pd.concat([test_df, clusters_test_df], axis=1, join='inner')
    result_test_df.to_csv(out_test_file, quoting=csv.QUOTE_NONNUMERIC)
    return out_train_file, out_test_file

def dataset_balance():
    out_files = os.listdir(OUTPUT_DIR)
    for file in out_files:
        if not file.endswith('_clusters.csv'):
            continue
        print(file)
        cluster_df = pd.read_csv(OUTPUT_DIR + file).dropna()
        cluster_df['label'].value_counts().plot.bar(title=file)
        plt.show()

def simple_tokenizer(line):
    for c in '[](){}/,':
        line  = line.replace(c, '')
    return line.split(" ")

def get_params_within_cluster():
    cluster_files = os.listdir(OUTPUT_DIR)
    for file in cluster_files:
        if not file.endswith('_clusters.csv'):
            continue
        print(file)
        cluster_df = pd.read_csv(OUTPUT_DIR + file,index_col=[0])
        num_clusters = max(cluster_df.clusters) + 1
        
        for i in range(num_clusters):
            sub_df = cluster_df[cluster_df["clusters"] == i]
            tvec = TfidfVectorizer(min_df=1, max_df = 0.5, tokenizer=simple_tokenizer)
            try:
                tvec.fit_transform(sub_df.text)
            except:
                print('No parameters for ', sub_df["text"].head(1))
                continue
            
            print('tvec_vocab: ',len(tvec.get_feature_names_out()))
            count = 0
            for index, row in sub_df.iterrows():
                words = row["text"].split(" ")
                p = 0
                count += 1
                if count % 10000:
                    print(count)
                #if count < 3: print(row["text"])
                #print('index',index, row)
                for j, word in enumerate(words):
                    if not word.replace('.', '').isnumeric():
                        continue
                    if word in tvec.vocabulary_:
                        cluster_df.at[index, "p-"+str(i)+"-"+str(p)] = word
                        #if count < 3: print('\tFound param "%s" in row %d at position %d(p=%d) in cluster %d' % (word, row[0], j, p, i))
                        p += 1

        outfile = OUTPUT_DIR + file.replace('_clusters.csv', '_params.csv')
        cluster_df.to_csv(outfile, quoting=csv.QUOTE_NONNUMERIC)


#get_initial_clusters(OUTPUT_DIR+'OpenStack'+'.csv')
process_raw_files()
get_params_within_cluster()
apply_sliding_window()
#dataset_balance()
print('DONE')
   
