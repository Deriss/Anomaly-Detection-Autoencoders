import streamlit as st

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import string

# Auxiliary function
def ROI_curve2(train_mae,test_mae,test_free_mae,plot=True):
    
    # Create mask for normal points in the test faulty set
    index = np.arange(len(test_mae["1"]))
    normal = index[(index%960)<160]
    mask_normal = np.zeros(len(test_mae["1"]), dtype=bool)
    mask_normal[normal] = 1
    # Select quantiles for the creation of thresholds
    quantiles = np.concatenate((np.arange(0,0.1,0.001),np.arange(0.1,0.9,0.01),np.arange(0.9,1.001,0.001)))
    #thresholds = np.quantile(train_mae_loss,quantiles)
    
    # Create thresholds for each autoencoder
    thresholds = np.array([np.quantile(train_mae[str(i)],quantiles) for i in range(1,5) ])
    
    #Create arrays for TPR and FPR
    true_positive_rate  = np.empty_like(thresholds[0])
    false_positive_rate = np.empty_like(thresholds[0])
    # Calculate totals
    total_positive = test_mae["1"].shape[0]  - mask_normal.sum() # do not count normal points 
    total_negative = test_free_mae["1"].shape[0] + mask_normal.shape[0] # add normal points in test faulty
    # Initialize anomaly arrays
    anom_test = []
    anom_test_free = []
    # Calculate anomalies

    for k in range(thresholds.shape[1]):
        anom_test.append(np.zeros_like(test_mae["1"],dtype=np.bool))
        anom_test_free.append(np.zeros_like(test_free_mae["1"],dtype=np.bool))
        for i in range(4):
            anom_test[k] = anom_test[k] | (test_mae[str(i+1)] > thresholds[i,k])
            anom_test_free[k] = anom_test_free[k] | (test_free_mae[str(i+1)]>thresholds[i,k])
        true_positive_count = (anom_test[k]*~mask_normal ).sum()
        false_positive_count = anom_test_free[k].sum() + (anom_test[k]*mask_normal).sum()
        true_positive_rate[k] = true_positive_count/total_positive
        false_positive_rate[k] = false_positive_count/total_negative
    false_positive_rate= np.insert(false_positive_rate,0,1)
    true_positive_rate = np.insert(true_positive_rate,0,1)
    if plot:
        plt.figure()
        plt.title("ROC Curve")
        plt.plot(false_positive_rate, true_positive_rate,'-')
        plt.plot([0,1],[0,1],'-')
        plt.ylabel("TPR")
        plt.xlabel("FPR")
        plt.xlim((0,1.01))
        plt.ylim((0,1.01))
        plt.show()
    return((false_positive_rate,true_positive_rate))
##############

test_faulty = pd.read_csv('../data/processed/TEP_test_faulty.csv')
test_free = pd.read_csv('../data/processed/TEP_test_free.csv')
anomaly = np.genfromtxt('../outputs/Anomaly_4AE.csv', delimiter=',')
anomaly_free = np.genfromtxt('../outputs/Anomaly_4AE.csv', delimiter=',')
train_mae = pd.read_csv('../outputs/train_mae_4AE.csv',index_col=0)
test_mae = pd.read_csv('../outputs/test_mae_4AE.csv',index_col=0)
test_free_mae = pd.read_csv('../outputs/test_free_mae_4AE.csv',index_col=0)

# Create variable description 
continuous = ['A Feed (stream 1)',
                      'D Feed (stream 2)',
                      'E Feed (stream 3) ',
                      'A and C Feed (stream 4)',
                      'Recycle Flow (stream 8)',
                      'Reactor Feed Rate (stream 6)',
                      'Reactor Pressure',
                      'Reactor Level',
                      'Reactor Temperature',
                      'Purge Rate (stream 9)',
                      'Product Sep Temp',
                      'Product Sep Level',
                      'Prod Sep Pressure',
                      'Prod Sep Underflow (stream 10)',
                      'Stripper Level',
                      'Stripper Pressure',
                      'Stripper Underflow (stream 11)',
                      'Stripper Temperature',
                      'Stripper Steam Flow',
                      'Compressor Work ',
                      'Reactor Cooling Water Outlet Temp',
                      'Separator Cooling Water Outlet Temp']

reactor_feed_analysis = [f'6{i}' for i in string.ascii_uppercase[:6]]

purge_gas_analysis = [f'9{i}' for i in string.ascii_uppercase[:8]]

product_analysis = [f'11{i}' for i in string.ascii_uppercase[3:8]]

manipulated = ['D Feed Flow (stream 2)',
                     'E Feed Flow (stream 3)',
                    'A Feed Flow (stream 1)',
                    'A and C Feed Flow (stream 4)',
                    'Compressor Recycle Valve',
                    'Purge Valve (stream 9)',
                    'Separator Pot Liquid Flow (stream 10)',
                    'Stripper Liquid Product Flow (stream 11)',
                    'Stripper Steam Valve',
                    'Reactor Cooling Water Flow',
                    'Condenser Cooling Water Flow']          

labels = continuous+reactor_feed_analysis+ purge_gas_analysis+product_analysis+ manipulated

# Create dictionary to get variable name from its description
zip_iterator = zip(labels, test_faulty.columns[3:])
vars_dict = dict(zip_iterator)

# Title
st.title('Tennessee Eastman Anomaly Detection using Multiple Autoencoders')

# Fault selection
col1,_,_ = st.columns(3)
with col1:
    label='Select fault to visualize'
    fault_number = st.number_input(label=label, min_value=1,max_value=21)

# Variable selection
large_col, _ = st.columns([3,2])
label2='Select variable to visualize'
options = test_faulty.columns[3:]
with large_col:
    selected_col = st.selectbox(label2, labels)
# Add predictions
autoencoder = st.checkbox("Autoencoder Anomaly Detection")

# Add slider quantile


if autoencoder:
    _, middle_col, _ = st.columns([1,10,1])
    with middle_col:
        quantile = st.slider(label = "Select quantile for threshold",min_value=0,max_value=100,value=99)
    thresholds = [np.quantile(train_mae[str(i)],quantile/100) for i in range(1,5) ]
    anom_test_99 = np.zeros_like(test_mae[str(1)],dtype=np.bool)
    anom_test_free_99 = np.zeros_like(test_free_mae[str(1)],dtype=np.bool)
    for i in range(4):
        anom_test_99 = anom_test_99 | (test_mae[str(i+1)] > thresholds[i])
        anom_test_free_99 = anom_test_free_99 | (test_free_mae[str(i+1)]>thresholds[i])
        

# Mask values using autoencoder results
if autoencoder:
    masked_values = np.ma.array(data=test_faulty[vars_dict[selected_col]][960*(fault_number-1):960*fault_number],mask=~(anom_test_99[960*(fault_number-1):960*fault_number]))

# Figure
fig = plt.figure(figsize=(15,8))
plt.title(f"Anomalies detected using Autoencoder Ensemble - Fault {fault_number}")
plt.plot(test_faulty.index[:960],test_faulty[vars_dict[selected_col]][960*(fault_number-1):960*fault_number])
plt.ylabel(selected_col)
if autoencoder: 
    plt.plot(test_faulty.index[:960],masked_values,'r')
plt.axvline(160,color='g')
if autoencoder:
    plt.legend(["Normal","Anomaly"])
else:
    plt.legend(["Normal"])
plt.show()

#Figure ROC



false_positive_rate, true_positive_rate = ROI_curve2(train_mae,test_mae,test_free_mae,plot=False)

# ROC curve
fig2 = plt.figure(figsize=(2,3))
plt.title("ROC Curve")
plt.plot(false_positive_rate, true_positive_rate,'-')
plt.plot([0,1],[0,1],'-')
plt.ylabel("TPR")
plt.xlabel("FPR")
plt.xlim((0,1.01))
plt.ylim((0,1.01))
plt.show()


# Draw Figure
st.pyplot(fig)

_, col_m,_ = st.columns([1,8,1])
with col_m:
    st.pyplot(fig2)
