## Written and copyright by Ekansh Gupta
## Georgia Institute of Technology
## egupta8@gatech.edu

#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
from scipy.signal import *
import matplotlib
from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt
import csv
from scipy.interpolate import interp1d
import blink
from sklearn import svm
from random import randrange
import random
import seaborn as sn
import pandas as pd
from math import ceil
from mpl_toolkits.mplot3d import axes3d, Axes3D
from scipy import signal
from scipy.signal import *
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
import itertools

blink_hist_list=[5,30,178,62,104,157,130,220,280]
blink_exclusion_list=[]
# blink_exclusion_list=[0,30,31,32,38,72,106,111,112,116,122,124,126,127,140,149,234,235,246,330,364,442,460,472,479,529,530,
# 1,2,27,88,90,93,102,288,550,551,593,646,711,942]
# user_exclusion_list=[1,4,5,6,18]
user_exclusion_list=[1,4,5,18]
# user_exclusion_list=[]
user_accuracy = [0]*20
train_split=0.8
PCA_var=0.9
NO_BLINK_OVERLAP=999999
duplicate_blinks=0
global_label_list=[]
# testing_mode="double"                                                           #mode can be single or double
testing_mode="single"                                                           #mode can be single or double
blink_combination=1
total_rejected_blinks=[]
num_experiment=5
# svm_mode="OneClassSVM"
svm_mode="Multiclass"
user_blink_stats=[[0 for x in range(20)] for x in range(20)]

def isOverlap(user_id,start_pt,end_pt):
    global duplicate_blinks
    indices = total_indices_per_user[user_id]
    for blink in indices:
        start1 = blink[0]
        end1 = blink[1]
        if (start_pt>=start1 and start_pt<=end1) or (end_pt>=start1 and end_pt<=end1)or \
        (start_pt>=start1 and end_pt<=end1) or (start_pt<=start1 and end_pt>=end1):
            duplicate_blinks=duplicate_blinks+1
            return blink[2]                                                             #User ID of blink
    return NO_BLINK_OVERLAP

def to_percent(y, position):
    # Ignore the passed in position. This has the effect of scaling the default
    # tick locations.
    s = str(100 * y)

    # The percent symbol needs escaping in latex
    if matplotlib.rcParams['text.usetex'] is True:
        return s + r'$\%$'
    else:
        return s + '%'

def ICA_test():
    a_l=[]
    b_l=[]
    for i in range(0,100000):
        a = randrange(100)
        b = randrange(100)
        a_l.append(a+b)
        b_l.append(3*a+5*b)
    plt.scatter((a_l),(b_l))
    plt.show()

def ICA_test_2():
    a_l=[]
    b_l=[]
    for i in range(0,100000):
        a = randrange(100)
        a_l.append(a)
    b_l = a_l[10:]
    b_l = [x * 5 for x in b_l]
    a_l = a_l[:-10]
    print(len(a_l),len(b_l))
    plt.scatter((a_l),(b_l))
    plt.show()

def preproc_blink(sample_pt_1):
    sample_pt_1 = sample_pt_1 - sample_pt_1[0]
    len_sp1 = len(sample_pt_1)
    final_val1 = sample_pt_1[len_sp1-1]
    corr_vec_1 = np.linspace(0,final_val1,len_sp1)
    sample_pt_1 = sample_pt_1 - corr_vec_1
    return sample_pt_1

def get_blink_fft(blink_1):
    len_blink_1 = len(blink_1)
    fft_bl = np.fft.fft(blink_1)
    fft_mirror=[]
    fft_mirror[0:ceil(len_blink_1/2)] = (abs(fft_bl))[ceil(len_blink_1/2):]
    fft_mirror[ceil(len_blink_1/2):] = (abs(fft_bl))[0:ceil(len_blink_1/2)]
    return fft_mirror

def get_convexity(blink_1):
    diff_1 = np.diff(blink_1)
    B,A = butter(4,(20)/(250/2), btype='low')
    diff_1 = lfilter(B, A, diff_1, axis=0)
    diff_2 = np.diff(diff_1)
    B,A = butter(4,(20)/(250/2), btype='low')
    diff_2 = lfilter(B, A, diff_2, axis=0)
    return diff_1,diff_2

def plot_hist(sample_pt,bins1,percent1,bins2,percent2,mid_pt):
    plt.figure(figsize=(10,6))
    plt.subplot(131)
    plt.plot(range(0,(len(sample_pt))),sample_pt)
    plt.plot(mid_pt, sample_pt[mid_pt],'r.')
    plt.ylabel(str(x)+": "+str(total_blink_num))
    plt.subplot(132)
    plt.bar(bins1[:-1],percent1,width=bins1[1]-bins1[0])
    plt.subplot(133)
    plt.bar(bins2[:-1],percent2,width=bins2[1]-bins2[0])
    plt.show()

def get_features(sample_pt,mid_pt1):
    feature_list=[]                                                 #list of feature values per blink. Will be appended to global features later
    #just append any new feature to feature_list
    conv1, conv1_dd = get_convexity(sample_pt)                      #getting single and double derivative after low pass filter
    conv1 = np.array(conv1).tolist()
    conv1_dd = np.array(conv1_dd).tolist()
    energy_arr = np.square(sample_pt)                               #getting energy inside blink
    fft_sample = get_blink_fft(sample_pt)                           #getting fft

    feature_list.append(sum(sample_pt)/len(sample_pt))              #mean
    feature_list.append(max(conv1))                                 #max slope going up
    feature_list.append(min(conv1))                                 #max slope coming down
    feature_list.append(min(sample_pt))                             #peak
    feature_list.append(len(sample_pt))                             #duration
    feature_list.append(sum(abs(sample_pt)))                        #area
    feature_list.append(np.var(sample_pt))                          #variance
    feature_list.append(sum(energy_arr))                            #energy
    # feature_list.append((sample_pt[mid_pt1]-sample_pt[0])/mid_pt1)
    # feature_list.append((sample_pt[len(sample_pt)-1]-sample_pt[mid_pt1])/(len(sample_pt)-mid_pt1))

    # hist1,bins1 = ((np.histogram(sample_pt,bins=10)))                       #10 bins for blink value histogram
    # percent1 = [i/sum(hist1)*100 for i in hist1]                            #converting to percentage
    # for index in percent1:
    #     feature_list.append(index)
    # hist2,bins2 = ((np.histogram(conv1,bins=np.linspace(-20,20,16))))       #15 bins (16 separations) for slope histogram
    # percent2 = [i/sum(hist2)*100 for i in hist2]                            #converting to percentage
    # for index in percent2:
    #     feature_list.append(index)
    # hist3,bins3 = ((np.histogram(fft_sample[0:int((len(sample_pt))/2)],bins=3)))                      #3 bins for fft left side as it is symmetric
    # percent3 = [i/sum(hist3)*100 for i in hist3]                            #converting to percentage
    # for index in percent3:
    #     feature_list.append(index)

    return feature_list

def get_blinks():
    global mode1, exclusion_list, total_indices_per_user, total_rejected_blinks
    total_features_train=[]
    total_labels_train=[]
    total_blinks=[]
    rangeVal = 20 if mode1 is True else 12
    num_samples = []
    total_blink_num=-1
    for channel in range(1,3):                                                          #DO NOT CHANGE THIS LOOP ORDER
        for x in range(0,rangeVal):                                                     #DO NOT CHANGE THIS LOOP ORDER
            individual_user_indices=[]
            blink_num_1=0                                                               #number of blinks accepted into training/testing set
            indices, data_sig, interval_corrupt=blink.init_data(mode1, x, 250, channel, True, 2.0, 0.2, 100, 0.2, 0.7)          #running blink algo
            indices = indices*250                                                       #checking corruption not required as blink does not return corrupt indices
            print("user "+str(x+1)+" channel "+str(channel))
            blink_num = len(indices)                                                    #total number of blinks
            if x in user_exclusion_list:                                                #list of excluded users by ID
                total_indices_per_user.append([])                                       #need empty element to address it by index
                total_blink_num=total_blink_num+blink_num                               #need to add blink nums for accurate outlier removal
                continue
            else:
                if x not in global_label_list:
                    global_label_list.append(x)
            for y in range(0,blink_num):                                                #processing each blink
                total_blink_num=total_blink_num+1                                       #incrementing blink number we have seen so far (before discarding)
                if total_blink_num in blink_exclusion_list:                             #filtering manually identified identifier
                    total_rejected_blinks.append(total_blink_num)
                    continue

                start_pt = int(indices[y,0])
                mid_pt = int(indices[y,1])
                end_pt = int(indices[y,2])
                sample_pt = preproc_blink(data_sig[start_pt:end_pt,1])                  #straightening the blink
                blink_len = len(sample_pt)
                #The method of using extended blink energy bins is not a good way of removing outliers so I've deleted it
                feature_list = get_features(sample_pt,(mid_pt-start_pt))                                  #getting blink features
                # if(total_blink_num in blink_hist_list):
                #     plt.plot(range(0,len(sample_pt)),sample_pt)
                #     plt.show()
                if(channel!=1):
                    blink_id = isOverlap(x,start_pt,end_pt)
                    if(blink_id==NO_BLINK_OVERLAP):
                        blink_id = 10000*channel + total_blink_num                        #so that channel 1 and 2 are unique
                else:
                    blink_id = 10000*channel + total_blink_num
                individual_user_indices.append([start_pt,end_pt,blink_id])              #insert blink ID to beginning
                feature_list.insert(0,blink_id)

                total_features_train.append(feature_list)                               #appending to global features
                total_labels_train.append(x)                                            #adding label
                total_blinks.append(sample_pt)                                          #appending to global collection of blinks
                blink_num_1=blink_num_1+1                                               #increasing number of blinks not discarded

            num_samples.append(blink_num_1)
            total_indices_per_user.append(individual_user_indices)
    return total_features_train, total_labels_train, num_samples, total_blinks

def find_accuracy(test_result,total_labels_test,num_users):
    test_accuracy = test_result == total_labels_test
    test_accuracy_per_user = [0] * num_users
    test_total_per_user = [0] * num_users

    for i in range(0,len(test_accuracy)):
        index = total_labels_test[i]
        test_total_per_user[index]=test_total_per_user[index]+1
        if(test_accuracy[i]==1):
            test_accuracy_per_user[index]=test_accuracy_per_user[index]+1
    for i in range(0,len(test_accuracy_per_user)):
        if(test_total_per_user[i]!=0):
            test_total_per_user[i]=int((test_accuracy_per_user[i]/test_total_per_user[i])*1000)/10
    print(str(sum(test_accuracy))+"/"+str(len(test_accuracy)))
    print(sum(test_accuracy)/len(test_accuracy))
    print(test_total_per_user)
    print()

def find_accuracy_double_blinks(test_result,total_labels_test,num_users):
    test_accuracy = test_result == total_labels_test
    test_accuracy_per_user = [0] * num_users
    test_total_per_user = [0] * num_users

    for i in range(0,len(test_accuracy)):
        index = total_labels_test[i]
        test_total_per_user[index]=test_total_per_user[index]+1
        if(test_accuracy[i]==1):
            test_accuracy_per_user[index]=test_accuracy_per_user[index]+1
    print(sum(test_accuracy_per_user)/sum(test_total_per_user))
    return test_accuracy

def addLists(lists,num_lists):
    sum_list = [0]*len(lists[0])
    for ind_list in lists:
        for i in range(0,len(ind_list)):
            sum_list[i] = sum_list[i]+ind_list[i]
    return sum_list

def calculate_error_row_for_user(score_list,index):
    global user_blink_stats
    for element in score_list:
        user_blink_stats[index][element]=user_blink_stats[index][element]+1


def svn_combined_blinks(clf,total_features_test_local,total_labels_test_local,blink_combination,num_users):
    test_accuracy_per_user = [0] * num_users
    test_total_per_user = [0] * num_users

    for i in range(0,len(total_labels_test_local)):
        index = total_labels_test_local[i]
        test_total_per_user[index]=test_total_per_user[index]+1

    sum_pt=0
    for index in range(0,len(test_total_per_user)):                             #user is index
        elem = test_total_per_user[index]
        if elem==0:
            continue
        feature_user_set = total_features_test_local[sum_pt:sum_pt+elem]
        feature_combinations = itertools.combinations(feature_user_set, blink_combination)
        feature_combinations=list(feature_combinations)
        aggregate_list=[]
        aggregate_score=[]
        for set in feature_combinations:
            test_result_set = clf.predict_proba(set)
            final_prob_array=addLists(test_result_set,blink_combination)
            aggregate_score.append(global_label_list[return_max_index(final_prob_array)])
            aggregate_list.append(index)
        # print("\n")
        # print(aggregate_score)
        calculate_error_row_for_user(aggregate_score,index)
        test_accuracy = np.array(aggregate_score) == np.array(aggregate_list)
        print(str(sum(test_accuracy))+"/"+str(len(aggregate_list))+" = "+str(sum(test_accuracy)/len(aggregate_list)))
        user_accuracy[index]=user_accuracy[index]+(sum(test_accuracy)/len(aggregate_list))
        sum_pt=sum_pt+elem

def get_SVN_clf(total_features_train_local, total_labels_train_local,total_features_test_local, total_labels_test_local):
    scaler = StandardScaler()
    print(np.array(total_features_train_local).shape,np.array(total_labels_train_local).shape,np.array(total_features_test_local).shape
    ,np.array(total_labels_test_local).shape)
    scaler.fit(total_features_train_local)
    total_features_train_local = scaler.transform(total_features_train_local)
    total_features_test_local = scaler.transform(total_features_test_local)
    pca = PCA(PCA_var)
    pca.fit(total_features_train_local)
    total_features_train_local = pca.transform(total_features_train_local)
    total_features_test_local = pca.transform(total_features_test_local)
    print(total_features_train_local.shape)

    clf = svm.SVC(gamma='scale',probability=True)
    # print(len(total_features_train),len(total_labels_train))
    clf.fit(total_features_train_local, total_labels_train_local)
    train_result_local = clf.predict(total_features_train_local)
    print("\nTraining accuracy:")
    find_accuracy(train_result_local,total_labels_train_local,20)
    if blink_combination==1:
        test_result_local = clf.predict(total_features_test_local)
        conf_mat=confusion_matrix(total_labels_test_local, test_result_local)
        print(conf_mat)
        for i1 in range(0,len(conf_mat)):
            for j1 in range(0,len(conf_mat)):
                user_blink_stats[i1][j1]=user_blink_stats[i1][j1]+conf_mat[i1][j1]
        print("Test accuracy:")
        find_accuracy(test_result_local,total_labels_test_local,20)
    else:
        svn_combined_blinks(clf,total_features_test_local,total_labels_test_local,blink_combination,20)

def return_max_index(list1):
    list1 = l = [int(x * 100000) for x in list1]
    max1 = max(list1)
    for item in range(0,len(list1)):
        if list1[item] is max1:
            return item
    return 99999

def get_SVN_clf_double_blinks(total_features_train_local, total_labels_train_local,total_features_test_local, total_labels_test_local):

    num_features = np.array(total_features_test_local).shape[2]
    num_test_blinks = len(total_labels_test_local)
    total_features_test_local_column = \
    np.array(total_features_test_local).reshape(2*num_test_blinks,num_features)
    print(total_features_test_local_column.shape)
    scaler = StandardScaler()
    print(np.array(total_features_train_local).shape,np.array(total_labels_train_local).shape,np.array(total_features_test_local).shape
    ,np.array(total_labels_test_local).shape)
    scaler.fit(total_features_train_local)
    total_features_train_local = scaler.transform(total_features_train_local)
    total_features_test_local_column = scaler.transform(total_features_test_local_column)
    pca = PCA(PCA_var)
    pca.fit(total_features_train_local)
    total_features_train_local = pca.transform(total_features_train_local)
    total_features_test_local_column = pca.transform(total_features_test_local_column)
    print(total_features_train_local.shape)
    chan1_blinks=total_features_test_local_column[::2]
    chan2_blinks=total_features_test_local_column[1::2]

    clf = svm.SVC(gamma='scale',probability=True)
    # print(len(total_features_train),len(total_labels_train))
    clf.fit(total_features_train_local, total_labels_train_local)

    train_result_local = clf.predict(total_features_train_local)
    print("\nTraining accuracy:")
    find_accuracy(train_result_local,total_labels_train_local,20)

    test_result_local_chan1_prob = clf.predict_proba(chan1_blinks)
    test_result_local_chan2_prob = clf.predict_proba(chan2_blinks)

    aggregate_score_1=[]
    aggregate_score_2=[]
    for i in range(len(test_result_local_chan1_prob)):
        temp_list=test_result_local_chan1_prob[i]
        aggregate_score_1.append(global_label_list[return_max_index(temp_list)])
        temp_list=test_result_local_chan2_prob[i]
        aggregate_score_2.append(global_label_list[return_max_index(temp_list)])

    test_res_chan1=find_accuracy_double_blinks(np.array(aggregate_score_1),total_labels_test_local,20)
    test_res_chan2=find_accuracy_double_blinks(np.array(aggregate_score_2),total_labels_test_local,20)
    print(aggregate_score_1)
    print(aggregate_score_2)

    aggregate_score=[]
    aggregate_list=[]
    for i in range(len(test_result_local_chan1_prob)):
        temp_list=[sum(x) for x in zip(test_result_local_chan1_prob[i],test_result_local_chan2_prob[i])]
        aggregate_list.append(temp_list)
        aggregate_score.append(global_label_list[return_max_index(temp_list)])

    print(aggregate_score)
    test_res_aggr=find_accuracy_double_blinks(np.array(aggregate_score),total_labels_test_local,20)
    for i in range(len(test_res_aggr)):
        if aggregate_score_1[i]==aggregate_score_2[i] and aggregate_score[i]!=aggregate_score_1[i]:
            print(str(i)+" "+str(aggregate_score_1[i])+" "+str(aggregate_score[i]))
            print(test_result_local_chan1_prob[i])
            print(test_result_local_chan2_prob[i])
            print(aggregate_list[i])
            print("\n")

def get_SVN_oc_clf(total_features_train_1):
    clf = svm.OneClassSVM(nu=0.01,kernel="rbf",gamma='scale')
    clf.fit(total_features_train_1)
    return clf

def plt_decision_boundary(clf1,extra_data):
    points=[]
    for x in range (-3000,-2000,1):
        for y in range(-25500,-24500,1):
            points.append([x,y])
    result = clf1.predict(points)
    positive_pt=[]
    negative_pt=[]
    count=0
    for p in result:
        if(p == 1):
            positive_pt.append(points[count])
        else:
            negative_pt.append(points[count])
        count=count+1

    print(len(positive_pt),len(negative_pt))
    xs = [x[0] for x in negative_pt]
    ys = [x[1] for x in negative_pt]
    plt.plot(xs, ys, 'o',ms=10, color='red')

    xs = [x[0] for x in positive_pt]
    ys = [x[1] for x in positive_pt]
    plt.plot(xs, ys, 'o',ms=10, color='blue')

    if extra_data != []:
        xs = [x[0] for x in extra_data]
        ys = [x[1] for x in extra_data]
        plt.plot(xs, ys, 'o',ms=1, color='black')

    plt.show()

def plt_blink_dist():
    global total_features_train
    xs = [x[0] for x in total_features_train]
    ys = [x[1] for x in total_features_train]
    zs = [x[2] for x in total_features_train]
    sum_pt=0
    # plt.plot(xs[0:20], ys[0:20], 'o')
    fig = plt.figure()
    ax = Axes3D(fig)
    for x in range(0,len(num_samples)):
        # if(x==2) or (x==6):
        if(x<7):
            ax.scatter(xs[sum_pt:sum_pt+num_samples[x]], ys[sum_pt:sum_pt+num_samples[x]], zs[sum_pt:sum_pt+num_samples[x]], 'o', color=colors[x])
            sum_pt = sum_pt+num_samples[x]
    plt.show()

def shuffle_pts():
    global total_features_train, total_labels_train, num_samples_total
    sum_pt=0
    for x in range(0,len(num_samples_total)):
        if num_samples_total[x]==0:
            continue
        copy = total_features_train[sum_pt:sum_pt+num_samples_total[x]]
        random.shuffle(copy)
        total_features_train[sum_pt:sum_pt+num_samples_total[x]]= copy
        sum_pt = sum_pt+num_samples_total[x]

def get_conf_row(results_1,start,num_train):
    global num_samples_total
    list_ret=[]
    sum_pt=0
    results_1 = [0 if x==-1 else 1 for x in results_1]
    sum_remove = sum(results_1[start:start+num_train])
    for x in range(0,len(num_samples_total)):
        if(num_samples_total[x]==0):
            continue
        sum_val = sum(results_1[sum_pt:sum_pt+num_samples_total[x]])
        acc_percent = round(((sum_val/num_samples_total[x])*100),2)
        if(sum_pt==start):
            sum_val=sum_val-sum_remove
            acc_percent = round(((sum_val/(num_samples_total[x]-num_train))*100),2)                              #remove training accuracy
        list_ret.append(acc_percent)
        # list_ret.append(sum_val)
        sum_pt = sum_pt+num_samples_total[x]
    return list_ret

def getOneClass():
    global total_features_train, num_samples_total, total_labels_train
    print(np.array(total_features_train).shape)
    scaler = StandardScaler()
    print(np.array(total_features_train).shape,np.array(total_labels_train).shape)
    total_features_train=scaler.fit_transform(total_features_train)
    # print((total_features_train).mean(axis=0))
    # print((total_features_train).std(axis=0))
    pca = PCA(PCA_var)
    pca.fit(total_features_train)
    total_features_train = pca.transform(total_features_train)
    # print((total_features_train).mean(axis=0))
    # print((total_features_train).std(axis=0))
    print('Shuffling and training one class SVM')
    shuffle_pts()
    sum_pts=0;
    conf_mat=[]
    MAX_SET_COUNT=10000
    for x in range(0,len(num_samples_total)):
        tot = num_samples_total[x]
        if tot==0:
            continue
        train_num = int(train_split*tot)
        train_sample = total_features_train[sum_pts:sum_pts+train_num]
        test_sample_orig_user = total_features_train[sum_pts+train_num:sum_pts+tot]
        print('Training for class '+str(x+1)+' with '+str(len(train_sample))+' samples')
        if(blink_combination==1):
            clf1=get_SVN_oc_clf(train_sample)
            results = clf1.predict(total_features_train)
            conf_row = get_conf_row(results,sum_pts,train_num)
            print(conf_row)
            conf_mat.append(conf_row)
        else:
            clf2=get_SVN_oc_clf(train_sample)
            sum_pt_comb=0
            conf_row=[]
            for index in range(0,len(num_samples_total)):
                tot_index_num=num_samples_total[index]
                if(tot_index_num==0):
                    continue
                test_sample_comb = total_features_train[sum_pt_comb:sum_pt_comb+tot_index_num]
                # print(test_sample_comb[0])
                if(index==x):
                    feature_user_set = test_sample_orig_user
                else:
                    feature_user_set = test_sample_comb
                if(len(feature_user_set)>30):                                        #a limit to reduce complexity
                    feature_user_set = feature_user_set[:30]
                feature_combinations = itertools.combinations(feature_user_set, blink_combination)
                feature_combinations=list(feature_combinations)
                aggregate_list=[]
                aggregate_score=[]
                count_set=0

                # print(len(test_result_set))
                for set in feature_combinations:
                    if(count_set>MAX_SET_COUNT):                                #limit max number of permutations to MAX_SET_COUNT
                        break
                    test_result_set = clf2.predict(set)
                    final_prob=sum(test_result_set)
                    if(final_prob<0):
                        aggregate_score.append(-1)
                    else:
                        aggregate_score.append(index)
                    aggregate_list.append(index)
                    count_set=count_set+1
                # print(str(len(feature_combinations))+" "+str(count_set))
                # print(aggregate_score)
                # print('Testing for class '+str(index+1))
                test_accuracy = np.array(aggregate_score) == np.array(aggregate_list)
                print(str(sum(test_accuracy))+"/"+str(len(aggregate_list))+" = "+str(sum(test_accuracy)/len(aggregate_list)))
                user_accuracy[index]=user_accuracy[index]+(sum(test_accuracy)/len(aggregate_list))
                conf_row.append(int(1000*(sum(test_accuracy)/len(aggregate_list)))/10)
                user_blink_stats[x][index] = user_blink_stats[x][index]+(int(1000*(sum(test_accuracy)/len(aggregate_list)))/10)
                sum_pt_comb=sum_pt_comb+tot_index_num
            conf_mat.append(conf_row)
        sum_pts = sum_pts+tot


    if(blink_combination==1):
        df_cm = pd.DataFrame(conf_mat, index = [i for i in range(0,len(global_label_list))],columns = [i for i in range(0,len(global_label_list))])
        plt.figure(figsize = (10,7))
        sn.heatmap(df_cm)

    for elem in conf_mat:
        print(elem)

def sort_data():
    global total_features_train,total_labels_train
    num_samples_total=[0]*20
    for i in range(0,20):
        temp_array = np.array([i]*(len(total_labels_train)))
        count_val = sum(temp_array==np.array(total_labels_train))
        num_samples_total[i] = count_val

    train_zip = zip((total_labels_train),(total_features_train))
    train_zip = sorted(train_zip)
    total_labels_train = [x for x,y in train_zip]
    total_features_train = [y for x,y in train_zip]
    num_samples_train = [int(x * (train_split)) for x in num_samples_total]
    return num_samples_total,num_samples_train

def split_data_all_blinks(num_samples_total,num_samples_train):
    global total_features_train,total_labels_train
    features_train_final=[]
    features_test_final=[]
    labels_train_final=[]
    labels_test_final=[]
    num_samples_test=[]

    sum_pts=0
    for i in range(0,len(num_samples_total)):
        if num_samples_total[i]!=0:
            total_samples = num_samples_total[i]
            total_samples_test = total_samples-num_samples_train[i]
            array_of_features = total_features_train[sum_pts:sum_pts+total_samples]
            random.shuffle(array_of_features)
            array_of_features_test = array_of_features[:total_samples_test]
            array_of_features_train = array_of_features[total_samples_test:]
            for y in range(0,total_samples):
                ind_feature=total_features_train[sum_pts+y]                             #get yth feature
                total_features_train[sum_pts+y] = ind_feature[1:]                       #remove ID
            for x in array_of_features_test:
                features_test_final.append(x[1:])                                       #0 is blink ID
                labels_test_final.append(i)
            for x in array_of_features_train:
                features_train_final.append(x[1:])
                labels_train_final.append(i)
            num_samples_test.append(total_samples_test)
            sum_pts = sum_pts+total_samples
        else:
            num_samples_test.append(0)

    return features_train_final,features_test_final,labels_train_final,labels_test_final,num_samples_test

def split_data_double_blinks(num_samples_total,num_samples_train,single_blink_features_local,single_blink_labels_local):
    global total_features_train_dou,total_labels_train
    features_train_final=[]
    features_test_final=[]
    labels_train_final=[]
    labels_test_final=[]
    num_samples_test=[]
    single_blink_features_local_final=[]
    for blink_feature in single_blink_features_local:
        single_blink_features_local_final.append(blink_feature[1:])

    sum_pts=0
    for i in range(0,len(num_samples_total)):
        if num_samples_total[i]!=0:
            total_samples = num_samples_total[i]
            total_samples_test = total_samples-num_samples_train[i]
            array_of_features = double_blink_features[sum_pts:sum_pts+total_samples]
            print(np.array(array_of_features).shape)
            num = randrange(10)
            # for i in range(0,num):
            random.shuffle(array_of_features)
            array_of_features_test = array_of_features[:total_samples_test]
            array_of_features_train = array_of_features[total_samples_test:]
            for x in array_of_features_test:
                x_0=x[0]
                x_1=x[1]
                x_0=x_0[1:]
                x_1=x_1[1:]
                temp1=[]
                temp1.append(x_0)
                temp1.append(x_1)
                features_test_final.append(temp1)                                       #0 is blink ID
                labels_test_final.append(i)
            for x in array_of_features_train:
                x_0=x[0]
                x_1=x[1]
                features_train_final.append(x_0[1:])
                features_train_final.append(x_1[1:])
                labels_train_final.append(i)
                labels_train_final.append(i)
            num_samples_test.append(total_samples_test)
            sum_pts = sum_pts+total_samples
        else:
            num_samples_test.append(0)

    features_train_final = features_train_final+single_blink_features_local_final
    labels_train_final=labels_train_final+single_blink_labels_local
    return features_train_final,features_test_final,labels_train_final,labels_test_final,num_samples_test

def blink_id_exists(blink_id):
    global total_features_train_alt
    if(len(total_features_train_alt)==0):
        return NO_BLINK_OVERLAP
    for i in range(0,len(total_features_train_alt)):
        blink = total_features_train_alt[i]
        if(blink_id==blink[0]):
            return i
    return NO_BLINK_OVERLAP

def process_double_blinks():
    single_blink_features=[]
    double_blink_features=[]
    single_blink_labels=[]
    double_blink_labels=[]
    error_count=0
    while len(total_features_train_alt) > 0:
        blink_0 = total_features_train_alt.pop(0)
        blink_id = blink_0[0]
        label_0 = total_labels_train_alt.pop(0)
        duplicate_id = blink_id_exists(blink_id)
        if(duplicate_id!=NO_BLINK_OVERLAP):                                    #blink exists
            blink_1 = total_features_train_alt.pop(duplicate_id)
            label_1 = total_labels_train_alt.pop(duplicate_id)
            if(label_0!=label_1):
                error_count=error_count+1
            double_blink=[]
            double_blink.append(blink_0)
            double_blink.append(blink_1)
            double_blink_features.append(double_blink)
            double_blink_labels.append(label_0)
        else:
            single_blink_features.append(blink_0)
            single_blink_labels.append(label_0)
    print("error is: "+str(error_count))
    num_samples_total_double=[0]*20
    for i in range(0,20):
        temp_array = np.array([i]*(len(double_blink_labels)))
        count_val = sum(temp_array==np.array(double_blink_labels))
        num_samples_total_double[i] = count_val
    num_samples_train_double = [int(x * (train_split)) for x in num_samples_total_double]
    return num_samples_total_double,num_samples_train_double,single_blink_features,double_blink_features,single_blink_labels,double_blink_labels

mode1=True
colors = ['blue','green','red','cyan','magenta','yellow','black','maroon','pink','gray','orange','brown','khaki','olivedrab','lime','turquoise',
'teal','deepskyblue','violet','hotpink']
for i in range(0,num_experiment):
    total_indices_per_user=[]
    print('Getting Blink waveforms for trial: '+str(i+1))
    total_features_train, total_labels_train, num_samples, total_blinks = get_blinks()
    print(len(total_rejected_blinks))
    print(total_rejected_blinks)
    num_samples_total,num_samples_train = sort_data()
    print(num_samples_total)

    if testing_mode is "single":
        features_train_final,features_test_final,labels_train_final,labels_test_final,num_samples_test = \
        split_data_all_blinks(num_samples_total,num_samples_train)
        if svm_mode is "OneClassSVM":
            getOneClass()
        else:
            get_SVN_clf(features_train_final, labels_train_final,features_test_final, labels_test_final)
    else:
        total_features_train_alt = total_features_train.copy()
        total_labels_train_alt = total_labels_train.copy()
        # print(np.array(total_indices_per_user).shape, np.array(total_features_train).shape,duplicate_blinks)
        num_samples_total_double,num_samples_train_double,single_blink_features,double_blink_features,single_blink_labels,double_blink_labels = \
        process_double_blinks()
        # print(num_samples_total_double)
        #all single blink features go for training. double ones are split
        print(num_samples_total_double,num_samples_train_double)
        features_train_final_double,features_test_final_double,labels_train_final_double,labels_test_final_double,num_samples_test_double = \
        split_data_double_blinks(num_samples_total_double,num_samples_train_double,single_blink_features,single_blink_labels)
        get_SVN_clf_double_blinks(features_train_final_double,labels_train_final_double,features_test_final_double,labels_test_final_double)

user_accuracy = [x/num_experiment for x in user_accuracy]
print(user_accuracy)
print("\n\n")
print(user_blink_stats)
index_1 = 0


# plt.show()
