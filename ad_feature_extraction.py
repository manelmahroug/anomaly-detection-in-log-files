#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 12:46:31 2023

@author: tps
"""
import os
import re
import csv
import shutil
import pandas as pd
import numpy as np
import collections
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score

BASE_DIR = '/home/thanuja/Dropbox/capstone/raw_files'
SUB_DIR = ['BGL','Hadoop','OpenStack']
OUTPUT_DIR = '/home/thanuja/Dropbox/capstone/output/'
WINDOW_SIZE = 100
ANOMALY_DIR = '/home/thanuja/Dropbox/capstone/anomalies'

# Anomalies for Hadoop are labeled based on the file name. Anomalies for
# OpenStack are based on vm instance id. Create a set for both log types to
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
    shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)
    recursive_files = list(os.walk(BASE_DIR))
    recursive_files.sort()
    csvs_with_ts = set()
    #is_anomaly = False
    for root, dirs, files in recursive_files:
        #print(dirs)
        for file in files:
            if file.endswith('.log'):
                file_name = os.path.join(root,file)
                print(file_name)
                app_sys_name = re.search(r'raw_files/([^/]+)/',file_name).group(1)
                csvfile_with_ts = parse_ts(file_name, app_sys_name)
                csvs_with_ts.add(csvfile_with_ts)
    for csvfile_with_ts in csvs_with_ts:
        get_initial_clusters(csvfile_with_ts)

# process all cluster output files to create a sliding window file for each.
def apply_sliding_window():
   out_files = os.listdir(OUTPUT_DIR)
   for file in out_files:
       if not file.endswith('clusters.csv'):
           continue
       print(file)
       sliding_window(OUTPUT_DIR + file)

# counts number of clusters in each window of the sliding window. Also record
# the length of time of each window, and the row numbers of the original csv.
def sliding_window(csv_with_clusters):
    CLUSTER_COL=4
    LABEL_COL=3
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
    cluster_counts = [0]*(k+6)
    outfile = csv_with_clusters.replace('clusters.csv', '_sliding_window.csv')
    with open(csv_with_clusters, 'r') as ip_file, open(outfile, 'w+') as sw_file:
        reader = csv.reader(ip_file)
        csvwriter = csv.writer(sw_file)
        for row in reader:
            lst_row = list(row)
            cluster_str = lst_row[CLUSTER_COL]
            l_ts = lst_row[1]
            if not cluster_str.isnumeric():
                continue
            cluster = int(cluster_str)
            de.append(lst_row)
            cluster_counts[cluster] += 1
            if len(de) <= w:
                continue
            #print('cluster_counts:',cluster_counts)
            #print('len', len(de))
            row_old = de.popleft()
            d_ts = float(l_ts)-float(row_old[1])
            #    print('row_old: ',row_old)
            cluster_counts[int(row_old[CLUSTER_COL])] -= 1
            cluster_counts[-1]= l_ts
            cluster_counts[-2]= d_ts
            cluster_counts[-3]= lst_row[0]
            cluster_counts[-4]= row_old[0]
            cluster_counts[-5]= lst_row[LABEL_COL]
            csvwriter.writerow(cluster_counts)

# parse a single line of a hadoop log file
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
    return epochts, remainder_ll, is_anomaly

# parse a single line of an openstack log file
def parse_openstack_files(file_path, line):
    words = line.split(' ')
    if len(words) < 2:
        raise ValueError()
    words.pop(0)
    w1 = words.pop(0)
    w2 = words.pop(0)
    str_ts = w1+'-'+w2
    epochts = datetime.strptime(str_ts,'%Y-%m-%d-%H:%M:%S.%f').timestamp()*1000 
    remainder_ll = ' '.join(words)
    is_anomaly = False
    for ele in open_stack_anomalies:
        if ele in remainder_ll:
            is_anomaly = True
    return epochts, remainder_ll, is_anomaly

# parse a single line of a BGL log line
def parse_bgl_files(file_path, line):
    is_anomaly = False
    words = line.split(' ')
    if len(words) < 1:
        raise ValueError()
    if words[0]=='-':
        is_anomaly = True
        words.pop(0)
    if len(words) < 4:
        raise ValueError()
    words.pop(0)
    words.pop(0)
    words.pop(0)
    str_ts = words.pop(0)
    epochts = datetime.strptime(str_ts,'%Y-%m-%d-%H.%M.%S.%f').timestamp()*1000        
    remainder_ll = ' '.join(words)
    return epochts, remainder_ll, is_anomaly

parsers = {'Hadoop':parse_hadoop_files,'OpenStack':parse_openstack_files,'BGL':parse_bgl_files}

# parse a raw log file. Log parsing specific to each log file is done in the
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
            csvwriter.writerow(['timestamp','text','label'])
        count = 0
        epochpts = 0
        for line in log_file:
            line = line.rstrip('\n')
            count+=1
            if count%10000==0:
                print(count)
            if app_sys_name not in parsers:
                raise ValueError("no parser found for " + app_sys_name)
            run_parser = parsers[app_sys_name]
            try:
                epochts,remainder_ll,is_anomaly = run_parser(file_path, line)
                epochpts = epochts
            except ValueError:
                csvwriter.writerow([epochpts,line,is_anomaly])
                epochts = epochpts
                continue
            if len(remainder_ll.strip()) == 0:
                continue
            csvwriter.writerow([epochts,remainder_ll,is_anomaly])
    log_file.close()
    return out_file

# cluster the tfidf vectors for a each raw log line, to apply cluster labels that
# correspond to the different types of log lines.
def get_initial_clusters(csv_log_file):
    print('finding clusters for ', csv_log_file)
    csv_log_df = pd.read_csv(csv_log_file).dropna()
    outfile = os.path.splitext(csv_log_file)[0]+'clusters.csv'
    tvec = TfidfVectorizer(min_df=100)
    print(csv_log_df.head().text)
    tvec_weights = tvec.fit_transform(csv_log_df.text)
    print('tvec_vocab: ',len(tvec.get_feature_names_out()))
    max_smallest_cluster_size = 10
    smallest_cluster_size = 1000
    k = 10
    # automatically determining optimal k. Repeat k-means until small_cluster_size < max_smallest_cluster_size
    while smallest_cluster_size > max_smallest_cluster_size:
        kmeans = KMeans(n_clusters=k, n_init=5).fit(tvec_weights)
        dense_weights = tvec_weights[:100000].toarray()
        labels = kmeans.predict(dense_weights)
        # print out some cluster metrics, which may be useful later.
        ch_score = calinski_harabasz_score(dense_weights, labels)
        db_score = davies_bouldin_score(dense_weights, labels)
        smallest_cluster_size = min(np.bincount(kmeans.labels_))
        print(k, smallest_cluster_size, max_smallest_cluster_size, 'ch', ch_score, 'db', db_score)
        k+=10
        # print(kmeans.labels_)
    clusters_df = pd.DataFrame({'clusters': kmeans.labels_})       
    # print(clusters_df.head(10))
    result_df = pd.concat([csv_log_df, clusters_df], axis=1, join='inner')
    # print(result_df.head(10))
    result_df.to_csv(outfile, quoting=csv.QUOTE_NONNUMERIC)
    return outfile

#get_initial_clusters(OUTPUT_DIR+'OpenStack'+'.csv')
process_raw_files()
apply_sliding_window()
print('DONE')
   
