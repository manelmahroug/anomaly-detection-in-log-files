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
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score
import uuid
import matplotlib.pyplot as plt
import subprocess
import random

BASE_DIR = '/home/thanuja/Dropbox/capstone/raw_files'
SUB_DIR = ['BGL','Hadoop','OpenStack','Thunderbird']
OUTPUT_DIR = '/home/thanuja/Dropbox/capstone/output/'
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

def backup_output():
    backup_ext = str(uuid.uuid1())
    if os.path.isdir(OUTPUT_DIR):
        os.rename(OUTPUT_DIR, OUTPUT_DIR[:-1] + '.' + backup_ext)
    os.makedirs(OUTPUT_DIR)

# parses timestamp and writes intial cluster CSVs
def process_raw_files(split_test_train=False):
    recursive_files = list(os.walk(BASE_DIR))
    recursive_files.sort()
    csvs_with_ts = set()
    for root, dirs, files in recursive_files:
        for file in files:
            if file.endswith('.log'):
                file_name = os.path.join(root,file)
                print(file_name)
                app_sys_name = re.search(r'raw_files/([^/]+)/',file_name).group(1)
                csvfile_with_ts = parse_ts_sample(file_name, app_sys_name)
                #parse_timeline_data(file_name, app_sys_name)
                #csvfile_with_ts = parse_ts(file_name, app_sys_name)
                csvs_with_ts.add(csvfile_with_ts)
    for csvfile_with_ts in csvs_with_ts:
        get_clusters(csvfile_with_ts, split_test_train=split_test_train)

# processes all cluster output files to create a sliding window file for each.
def apply_sliding_window():
   out_files = os.listdir(OUTPUT_DIR)
   for file in out_files:
       if not file.endswith('_clusters2.csv'):
           continue
       print(file)
       sliding_window(OUTPUT_DIR + file)

# counts number of clusters in each window of the sliding window. Also records
# the length of time of each window, and the row numbers of the original csv.
def sliding_window(csv_with_clusters):
    CLUSTER_COL=-1
    w = 20
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
    cluster_counts = [0]*(k+8)
    outfile = csv_with_clusters.replace('_clusters2.csv', '_sliding_window.csv')
    with open(csv_with_clusters, 'r') as ip_file, open(outfile, 'w') as sw_file:
        reader = csv.DictReader(ip_file)
        param_labels = [col for col in reader.fieldnames if col.startswith('p-')]
        csvwriter = csv.writer(sw_file)
        cluster_labels = ['cluster_' + str(i) for i in range(k+1)]
        cluster_labels += ['precision_label', 'recall_label', 'row_start', 'row_end', 'time_length', 'timestamp', 'filename']
        cluster_labels += param_labels
        csvwriter.writerow(cluster_labels)
        output_count = 0
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
            output_count += 1
            if output_count % 1000 == 0:
                print(output_count)
            row_old = de.popleft()
            lst_row_old = list(row_old.values())
            d_ts = float(row['timestamp'])-float(row_old['timestamp'])
            cluster_counts[int(row_old['cluster2'])] -= 1
            #cluster_counts[-1]=lst_row[-2]
            cluster_counts[-1]=row['filename']
            cluster_counts[-2]= row['timestamp']
            cluster_counts[-3]= d_ts
            #cluster_counts[-4]= lst_row[0]
            cluster_counts[-4]= lst_row[0]
            #cluster_counts[-5]= row_old[0]
            cluster_counts[-5]= lst_row_old[0]
            # Precision label is just the label
            cluster_counts[-7]= row['label']
            cluster_counts[-6]=False
            # If *any* row in sliding window has true label, mark recall_label=True
            for label_row in de:
                if label_row['label'] == 'True':
                    cluster_counts[-6]=True
            
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

def parse_thunderbird_files(file_path, line):
    is_anomaly = True
    words = line.split(' ')
    if len(words) < 1:
        raise ValueError()
    if words[0]=='-':
        is_anomaly = False
    if len(words) < 4:
        raise ValueError()
    if is_anomaly:
        #print('Found anomaly at', line)
        pass

    words.pop(0)
    str_ts = words.pop(0)
    words.pop(0)
    words.pop(0)
    words.pop(0)
    words.pop(0)
    words.pop(0)
    epochts = int(str_ts)*1000
    machine = words.pop(0)
    remainder_ll = ' '.join(words)
    return epochts, remainder_ll, is_anomaly, machine


parsers = {'Hadoop':parse_hadoop_files,'OpenStack':parse_openstack_files,'BGL':parse_bgl_files, 'Thunderbird':parse_thunderbird_files}

# parses a raw log file. Log parsing specific to each log file is done in the
# appropriate handler, such as parse_bgl_files.
# write to the output file in a way that maintains anomaly percent of original file. Write an entire window at a time to preserve context.
# this is done so that, anomalies always have context and to limit the data to make the model trainable.
def parse_ts(file_name,app_sys_name):
    num_anomalies = int(subprocess.run(['grep', '-cv', '^-', file_name], stdout=subprocess.PIPE).stdout)
    num_normal = int(subprocess.run(['grep', '-c', '^-', file_name], stdout=subprocess.PIPE).stdout)
    anomaly_target_percent = num_anomalies / (num_anomalies + num_normal)
    # Some slack is needed otherwise algorithm rarely adds anomalous or non-anomalous rows
    max_anomaly_percent = anomaly_target_percent * 1.1
    min_anomaly_percent = anomaly_target_percent * 0.9
    print('Found %d anomalies in %d lines for a target percentage of %f' % (num_anomalies, num_normal, anomaly_target_percent))
    log_file = open(file_name, "r+", encoding="utf8", errors='ignore')
    out_file = OUTPUT_DIR+app_sys_name+'.csv'
    is_file = os.path.isfile(out_file)
    csv_file = open(out_file, 'a+')
    file_path = file_name.split('/')
    min_lines = 100000
    min_anomalies = 5000
    
    with csv_file, log_file:
        csvwriter = csv.writer(csv_file, quoting=csv.QUOTE_NONNUMERIC, escapechar='\\')
        if not is_file:
            csvwriter.writerow(['timestamp','text','label','filename'])
        count = 0
        anomaly_output_count = 0
        output_count = 0
        remainder_ll = is_anomaly = filename = None
        # make sure that 1/w > anomaly_target_percent, so that a single anomaly can
        # increase current percentage even if no other anomalies are in current window
        w = int(1/anomaly_target_percent/2)
        de = collections.deque([], w+1)
        for line in log_file:
            line = line.rstrip('\n')
            count+=1
            if count%10000==0:
                print("%d / %d / %d" % (anomaly_output_count, output_count, count))
            if app_sys_name not in parsers:
                raise ValueError("no parser found for " + app_sys_name)
            run_parser = parsers[app_sys_name]
            remainder_ll = ''
            try:
                epochts,remainder_ll,is_anomaly,filename = run_parser(file_path, line)
            except ValueError:
                # when the line does not follow the normal format, assume it is continuation of 
                # previous log line. Errors than span multiple lines, are put together in a single line.
                if len(de) > 0:
                    epochts,remainder_ll,is_anomaly,filename = de.pop()
                    de.append(epochts,remainder_ll+line,is_anomaly,filename)
                continue
            if len(remainder_ll.strip()) == 0:
                continue
            #csvwriter.writerow([epochts,line_out,is_anomaly,filename])
            de.append((epochts, remainder_ll, is_anomaly,filename))
            anomaly_percent = anomaly_output_count / (output_count + 1) # avoid division by 0
            if is_anomaly and anomaly_percent < max_anomaly_percent:
                for epochts, remainder_ll, window_anomaly,filename in de:
                    csvwriter.writerow([epochts,remainder_ll,window_anomaly,filename])
                    if window_anomaly: anomaly_output_count += 1
                    output_count += 1
                de.clear()
            if not is_anomaly and anomaly_percent > min_anomaly_percent:
                for epochts, remainder_ll, window_anomaly,filename in de:
                    if window_anomaly: continue # don't add more anomalies here
                    csvwriter.writerow([epochts,remainder_ll,window_anomaly,filename])
                    output_count += 1
                de.clear()
            if output_count >= min_lines and anomaly_output_count >= min_anomalies:
                print('Output %d lines, stopping %s' % (output_count, file_name))
                break
        print("ACTUAL ANOMALY PERCENTAGE: ", anomaly_output_count / output_count, anomaly_percent)
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
def get_clusters(csv_log_file, split_test_train=False):
    print('finding clusters for ', csv_log_file)
    csv_log_df = pd.read_csv(csv_log_file).fillna('')
    csv_log_df.sort_values('timestamp', inplace=True)
    if split_test_train:
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
    else:
        train_df = csv_log_df
        out_train_file = os.path.splitext(csv_log_file)[0]+'_all_clusters.csv'
        out_test_file = ''

    max_smallest_cluster_size = 20
    
    tvec = TfidfVectorizer(min_df=max_smallest_cluster_size, tokenizer=simple_tokenizer)
    #unique_rows_df = train_df.copy()
    #unique_rows_df.text = unique_rows_df.text.replace('[0-9.]+', '', regex=True)
    #unique_rows_df.drop_duplicates(subset=['text'], inplace=True)
    #print(csv_log_file, train_df.size, 'rows has', unique_rows_df.size, 'unique rows')
    print(train_df.text[:5])
    tvec_weights_train = tvec.fit_transform(train_df.text)
    if split_test_train:
        tvec_weights_test = tvec.transform(test_df.text)
    print('tvec_vocab: ',len(tvec.get_feature_names_out()))
    smallest_cluster_size = 1000
    max_rows = 200000
    num_rows = tvec_weights_train.shape[0]
    best_score = 0
    best_kmeans = None
    k = 10
    if num_rows > max_rows:
        tvec_weights_sample = tvec_weights_train[np.random.choice(num_rows, max_rows, replace=False), :]
    else:
        tvec_weights_sample = tvec_weights_train
    # automatically determining optimal k. Repeat k-means until small_cluster_size < max_smallest_cluster_size
    while k < 100 or smallest_cluster_size > max_smallest_cluster_size:
        #kmeans = KMeans(n_clusters=k, n_init=10, algorithm='elkan', random_state=0).fit(tvec_weights_sample)
        kmeans = MiniBatchKMeans(n_clusters=k, n_init=10, batch_size=10000, random_state=0).fit(tvec_weights_sample)
        labels = kmeans.predict(tvec_weights_sample)
        dense_weights = tvec_weights_sample.toarray()
        ch_score = calinski_harabasz_score(dense_weights, labels)
        db_score = davies_bouldin_score(dense_weights, labels)
        score = ch_score / (10000 * db_score * db_score)
        if best_kmeans is None or best_score < score:
            best_score = score
            best_kmeans = kmeans
        smallest_cluster_size = min(np.bincount(kmeans.labels_))
        print('k', k, 'smallest cluster', smallest_cluster_size, 'ch', ch_score, 'db', db_score, 'score', score)
        k=int(k*1.1)+1
    train_labels = best_kmeans.predict(tvec_weights_train)
    clusters_train_df = pd.DataFrame({'clusters': train_labels})
    result_train_df = pd.concat([train_df, clusters_train_df], axis=1, join='inner')
    result_train_df.to_csv(out_train_file, quoting=csv.QUOTE_NONNUMERIC)

    if split_test_train:
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
    for c in '[](){},':
        line = line.replace(c, '')
    for c in '=/:':
        line = line.replace(c, ' ')
    return line.split(" ")

def get_params_within_cluster():
    cluster_files = os.listdir(OUTPUT_DIR)
    for file in cluster_files:
        if not file.endswith('_clusters.csv'):
            continue
        print(file)
        cluster_df = pd.read_csv(OUTPUT_DIR + file,index_col=[0])
        num_clusters = max(cluster_df.clusters) + 1
        cluster_df["tfidf_text"] = cluster_df["text"]
        
        for i in range(num_clusters):
            sub_df = cluster_df[cluster_df["clusters"] == i]
            tfidf_filter = TfidfVectorizer(min_df=1, max_df = 0.05, tokenizer=simple_tokenizer)
            tfidf_extractor = TfidfVectorizer(min_df=1, max_df = 0.5, tokenizer=simple_tokenizer)
            try:
                tfidf_filter.fit_transform(sub_df.text)
                tfidf_extractor.fit_transform(sub_df.text)
            except:
                print('No parameters for ', sub_df["text"].head(1))
                continue
            
            print('cluster', i, 'filter words', len(tfidf_filter.get_feature_names_out()),
                  'extractor words', len(tfidf_extractor.get_feature_names_out()))
            count = 0
            for index, row in sub_df.iterrows():
                words = simple_tokenizer(row["text"])
                p = 0
                count += 1
                if count % 10000 == 0:
                    print(count)
                #if count < 3: print(row["text"])
                #print('index',index, row)
                for j, word in enumerate(words):
                    # Remove if it has any digits. Assume all character words are meaningful.
                    if word in tfidf_filter.vocabulary_: # and bool(re.search(r'\d', word)):
                        cluster_df.at[index, 'tfidf_text'] = cluster_df.at[index, 'tfidf_text'].replace(word, '')

                    # Convert hex to int string
                    if re.match(r'^0x[0-9abcdef]+$', word):
                        cluster_df.at[index, 'tfidf_text'] = cluster_df.at[index, 'tfidf_text'].replace(word, '')
                        word = str(int(word, 16))
                    # Only extract decimal
                    if not word.replace('.', '').isnumeric():
                        continue

                    # Filter numbers and ip addresses
                    cluster_df.at[index, 'tfidf_text'] = cluster_df.at[index, 'tfidf_text'].replace(word, '')
                    
                    if re.match(r'(\d*\.\d*\.)(\d*\.)*', word):
                        continue
                    if word in tfidf_extractor.vocabulary_:
                        cluster_df.at[index, "p-"+str(i)+"-"+str(p)] = word
                        #if count < 3: print('\tFound param "%s" in row %d at position %d(p=%d) in cluster %d' % (word, row[0], j, p, i))
                        p += 1

        outfile = OUTPUT_DIR + file.replace('_clusters.csv', '_params.csv')
        cluster_df.to_csv(outfile, quoting=csv.QUOTE_NONNUMERIC)


def parse_ts_sample(file_name, app_sys_name):
    SAMPLE_SIZE = 100000
    SAMPLE_ANOMALY_SIZE = 5000
    CHUNK_SIZE = 1000
    file_path = file_name.split('/')
    lines = []
    num_anomalies = int(subprocess.run(['grep', '-cv', '^-', file_name], stdout=subprocess.PIPE).stdout)
    num_normal = int(subprocess.run(['grep','-cv','^_',file_name],stdout=subprocess.PIPE).stdout)
    total_lines = num_anomalies + num_normal
    output_line_perc = SAMPLE_SIZE/total_lines
    output_anomaly_perc = SAMPLE_ANOMALY_SIZE/num_anomalies
    total_output_percentage = max(output_line_perc,output_anomaly_perc)
    log_file = open(file_name, "r+", encoding="utf8", errors='ignore')
    out_file = OUTPUT_DIR + app_sys_name + '.csv'
    csv_file = open(out_file, 'w')
    anomaly_output_count = 0
    line_output_count = 0
    line_read_count = 0
    if app_sys_name not in parsers:
        raise ValueError("no parser found for " + app_sys_name)
    run_parser = parsers[app_sys_name]
    with csv_file,log_file:
        csvwriter = csv.writer(csv_file, quoting=csv.QUOTE_NONNUMERIC, escapechar='\\')
        csvwriter.writerow(['timestamp','text','label','filename'])
        for line in log_file:
            line = line.rstrip('\n')
            line_read_count += 1
            lines.append(line)
            if len(lines) != CHUNK_SIZE:
                continue
            r = random.random()
            if r > total_output_percentage:
                lines = []
                continue
            #print(r, total_output_percentage)
            #print(r)
            prev = None
            for block_line in lines:
                try:
                    epochts,remainder_ll,is_anomaly,filename = run_parser(file_path, block_line)
                    if is_anomaly:
                        anomaly_output_count +=1;
                except ValueError:
                    # when the line does not follow the normal format, assume it is continuation of 
                    # previous log line. Errors than span multiple lines, are put together in a single line.
                    prev[1] += block_line # 1 is the column of the text
                    continue
                if prev is not None:
                    # only write a line when there is no chance more data will be added to it.
                    # so when the current line parses successfully, the previous line must be complete.
                    csvwriter.writerow(prev)
                prev = [epochts,remainder_ll,is_anomaly,filename]
            csvwriter.writerow(prev)
            line_output_count += 1000
            if line_output_count % 10000 == 0:
                print(anomaly_output_count, line_output_count, line_read_count)
            lines = []
    print('line_output_count ',line_output_count)
    print('anomaly_output_count ',anomaly_output_count)
    return out_file

def parse_timeline_data(file_name, app_sys_name):
    file_path = file_name.split('/')
    log_counts = collections.defaultdict(int)
    anomaly_counts = collections.defaultdict(int)
    log_file = open(file_name, "r+", encoding="utf8", errors='ignore')
    out_file = OUTPUT_DIR+app_sys_name+'_timeline.csv'
    csv_file = open(out_file, 'w')
    line_count = 0
    if app_sys_name not in parsers:
        raise ValueError("no parser found for " + app_sys_name)
    run_parser = parsers[app_sys_name]
    with csv_file,log_file:
        csvwriter = csv.writer(csv_file, quoting=csv.QUOTE_NONNUMERIC, escapechar='\\')
        csvwriter.writerow(['date','num_logs','num_anomalies'])
        for line in log_file:
            line = line.rstrip('\n')
            try:
                epochts,_,is_anomaly,_= run_parser(file_path, line)
                date = datetime.utcfromtimestamp(epochts/1000).strftime('%Y-%m-%d-%H')
                log_counts[date] += 1
                if is_anomaly:
                    anomaly_counts[date] += 1
            except ValueError:
                continue
            line_count += 1
            if line_count % 100000 == 0:
                print(line_count)
        for date in log_counts:
            num_logs = log_counts[date]
            num_anomalies = anomaly_counts[date]
            csvwriter.writerow([date, num_logs, num_anomalies])
    return out_file


def redo_clusters():
    cluster_files = os.listdir(OUTPUT_DIR)
    for file in cluster_files:
        if not file.endswith('_all_params.csv'):
            continue
        print(file)
        cluster_df = pd.read_csv(OUTPUT_DIR + file,index_col=[0])
        num_clusters = max(cluster_df.clusters) + 1
        X = cluster_df.drop(columns = ['label','timestamp','filename'])
    
        tfidf = TfidfVectorizer()
        X_tfidf = tfidf.fit_transform(X['tfidf_text'])
    
        best_score = 0
        best_kmeans = None
        k = num_clusters - 5
        # automatically determining optimal k. Repeat k-means until small_cluster_size < max_smallest_cluster_size
        while k < num_clusters + 5:
            #kmeans = KMeans(n_clusters=k, n_init=10, algorithm='elkan', random_state=0).fit(tvec_weights_sample)
            kmeans = MiniBatchKMeans(n_clusters=k, n_init=10, batch_size=10000, random_state=0).fit(X_tfidf)
            labels = kmeans.predict(X_tfidf)
            if X_tfidf.shape[0] > 50000:
                indices = np.random.choice(X_tfidf.shape[0], 50000, replace=False)
                dense_weights = X_tfidf[indices].toarray()
                dense_labels = labels[indices]
            else:
                dense_weights = X_tfidf.toarray()
                dense_labels = labels
            ch_score = calinski_harabasz_score(dense_weights, dense_labels)
            db_score = davies_bouldin_score(dense_weights, dense_labels)
            score = ch_score / (10000 * db_score * db_score)
            if best_kmeans is None or best_score < score:
                best_score = score
                best_kmeans = kmeans
            smallest_cluster_size = min(np.bincount(kmeans.labels_))
            print('k', k, 'smallest cluster', smallest_cluster_size, 'ch', ch_score, 'db', db_score, 'score', score)
            k+=1
    
        kmeans = best_kmeans
        cluster_seq = kmeans.predict(X_tfidf)
        cluster_df['cluster2'] = cluster_seq
        
        outfile = OUTPUT_DIR + file.replace('_all_params.csv', '_clusters2.csv')
        cluster_df.to_csv(outfile, quoting=csv.QUOTE_NONNUMERIC)


if __name__ == '__main__':
    random.seed(0)
    #hadoop_anomalies, open_stack_anomalies = mark_anomalies()

    #get_clusters(OUTPUT_DIR+'BGL_all_params.csv')
    #parse_ts_sample('/home/thanuja/Dropbox/capstone/raw_files/BGL/BGL.log','BGL')
    #parse_timeline_data('/home/thanuja/Dropbox/capstone/raw_files/BGL/BGL.log','BGL')
    #parse_timeline_data('/home/thanuja/Dropbox/capstone/raw_files/Thunderbird/Thunderbird.log','Thunderbird')
    #backup_output()
    #process_raw_files(split_test_train=False) # for unsupervised training
    #process_raw_files(split_test_train=True) # for supervised training
    #get_params_within_cluster()
    #redo_clusters()
    apply_sliding_window()
    
    #dataset_balance()
    print('DONE')