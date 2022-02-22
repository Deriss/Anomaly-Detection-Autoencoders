import numpy as np
import pandas as pd


path = '../data/raw/'
train_file = 'd00.dat' # train data file name
test_file = 'd00_te.dat' # file name for test data wthout faults
# load train data
train_free = pd.read_table(path+train_file,sep='\s+',header=None).T
# load test data without faults
test_free = pd.read_table(path+test_file,sep='\s+',header=None)
old_col_names = list(train_free.columns)
manipulated = [f'xmv_{i}' for i in range(1,12)]
measured = [f'xmeas_{i}' for i in range(1,42)]
new_col_names = measured + manipulated

names_dict = dict(zip(old_col_names,new_col_names))
train_free.rename(columns = names_dict,inplace=True)
test_free.rename(columns = names_dict,inplace=True)
# load test data with faults and add them to an unique dataframe
test_faulty = None
for fault_number in range(1,22):
    test_faulty_data = pd.read_table(path+f'd{fault_number:02d}_te.dat',sep='\s+',header=None)
    test_faulty_data.rename(columns = names_dict,inplace=True)
    test_faulty_data['fault_number'] = fault_number
    if test_faulty is None:
        test_faulty = test_faulty_data
    else:
        test_faulty = pd.concat((test_faulty,test_faulty_data))
# save data to csv files
train_free.to_csv('../data/processed/TEP_train_free.csv',index_label = 'sample')
test_free.to_csv('../data/processed/TEP_test_free.csv',index_label = 'sample')
test_faulty.to_csv('../data/processed/TEP_test_faulty.csv',index_label = 'sample')
