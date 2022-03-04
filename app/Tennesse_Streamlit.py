import streamlit as st

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import string

st.set_page_config(page_title="Anomaly Detection Using Autoencoders", page_icon=None, layout='wide')
# Auxiliary function
def ROI_curve2(train_mae,test_mae,test_free_mae,plot=True):
    
    # Create mask for normal points in the test faulty set
    index = np.arange(len(test_mae["1"]))
    normal = index[(index%960)<160]
    mask_normal = np.zeros(len(test_mae["1"]), dtype=bool)
    mask_normal[normal] = 1
    # Select quantiles for the creation of thresholds
    quantiles = np.concatenate((np.arange(0,0.1,0.001),np.arange(0.1,0.9,0.01),np.arange(0.9,1.001,0.001)))
    
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
# Read files
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
#All labels
labels = continuous+reactor_feed_analysis+ purge_gas_analysis+product_analysis+ manipulated

# Create dictionary to get variable name from its description
zip_iterator = zip(labels, test_faulty.columns[1:])
vars_dict = dict(zip_iterator)

##################### START APP
# Title
st.title('Tennessee Eastman Anomaly Detection using Multiple Autoencoders')
st.markdown('by [Deris Spina](https://github.com/Deriss)')

st.write('Due to their ability to recognize nonlinear patterns, autoencoders are useful to reconstruct multivariate data. However, autoencoders would encounter difficulties to reconstruct anomaly points correctly and, thus, present a larger reconstruction error. We can take advantage of this fact to detect anomalies. In order to do this, we can establishing a threshold reconstruction error, beyond which a data point is consider an anomaly.')
st.write('In this work, an ensemble of four autoencoders was used to detect anomalies for the Tennesse Eastman Process. Each autoencoder uses a different set of variables, grouped by the equipment that they are related to: Reactor, Separator, Stripper or Streams. This arquitecture improves the detection of anomalies that only affects the process locally. The mean absolute error was selected as a measure for the reconstruction error. A test set containing 21 different failures was used.')
# Fault selection
col1,_ = st.columns([2,10])
with col1:
    label='Select fault to visualize'
    fault_number = st.number_input(label=label, min_value=1,max_value=21)

# Variable selection
large_col, _ = st.columns([3,2])
label2='Select variable to visualize'
options = test_faulty.columns[1:54]
with large_col:
    selected_col = st.selectbox(label2, labels)
# Add predictions
autoencoder = st.checkbox("Autoencoder Anomaly Detection")

# Add slider quantile


if autoencoder:
    m_col, _ = st.columns([8,4])
    with m_col:
        quantile = st.slider(label = "Select quantile for threshold [0-100%]",min_value=0,max_value=100,value=99,format = '%d%%' )
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
col_l, col_r = st.columns([7,3])
fig = plt.figure(figsize=(15,8))
plt.title(f"Anomalies detected using Autoencoder Ensemble - Fault {fault_number}\n",fontsize=18)
plt.plot(test_faulty.index[:960],test_faulty[vars_dict[selected_col]][960*(fault_number-1):960*fault_number])
plt.xlim([0,960])
plt.ylabel(selected_col)
if autoencoder: 
    plt.plot(test_faulty.index[:960],masked_values,'r')
plt.axvline(160,color='g')
if autoencoder:
    plt.legend(["Normal","Anomaly"])
else:
    plt.legend(["Normal"])
with col_l:
    st.pyplot(fig)


if autoencoder:
    #Figure ROC
    false_positive_rate, true_positive_rate = ROI_curve2(train_mae,test_mae[960*(fault_number-1):960*fault_number],test_free_mae,plot=False)
    fig2 = plt.figure(figsize=(3.5,4.9))
    plt.title("ROC Curve\n")
    plt.plot(false_positive_rate, true_positive_rate,'-')
    plt.plot([0,1],[0,1],'-')
    plt.ylabel("TPR")
    plt.xlabel("FPR")
    plt.xlim((0,1.01))
    plt.ylim((0,1.01))
    
    with col_r:
        st.pyplot(fig2)



# Figure 3 - MAE

if autoencoder:
    col_ae,_ = st.columns([3,9])
    label3 = "Autoencoder to visualize"
    labels_ae = ["Autoencoder 1 - Reactor","Autoencoder 2 - Separator", "Autoencoder 3 - Stripper", "Autoencoder 4 - Streams"]
    with col_ae:
        selected_ae = st.selectbox(label3, labels_ae)
    
    fig3 = plt.figure(figsize=(15,8))
    plt.plot(test_mae[str(labels_ae.index(selected_ae)+1)][960*(fault_number-1):960*fault_number],'k')
    plt.xlim([0,960])
    plt.title("Evaluation Metric for each Autoencoder of the Ensemble\n", fontsize=18)
    plt.legend([selected_ae])
    plt.ylabel("Mean Absolute Error", fontsize=16)
    plt.axhline(thresholds[labels_ae.index(selected_ae)],color='red')
    col_l2, col_r2 = st.columns([7,3])
    with col_l2:
        st.pyplot(fig3)



# ROC curve



    
    

