import numpy as np
import pandas as pd


path = '../data/raw/'
train_file = 'd00.dat' # train data file name
test_file = 'd00_te.dat' # file name for test data wthout faults
# load train data
train_free = pd.read_table(path+train_file,sep='\s+',header=None).T
# load test data without faults
test_free = pd.read_table(path+test_file,sep='\s+',header=None)

# load test data with faults and add them to an unique dataframe
test_faulty = None
for fault_number in range(1,22):
    test_faulty_data = pd.read_table(path+f'd{fault_number:02d}.dat',sep='\s+',header=None)
    test_faulty_data['fault_number'] = fault_number
    if test_faulty is None:
        test_faulty = test_faulty_data
    else:
        test_faulty = pd.concat((test_faulty,test_faulty_data))
# save data to csv files
train_free.to_csv('../data/processed/TEP_train_free.csv',index_label = 'sample')
test_free.to_csv('../data/processed/TEP_test_free.csv',index_label = 'sample')
test_faulty.to_csv('../data/processed/TEP_test_faulty.csv',index_label = 'sample')
