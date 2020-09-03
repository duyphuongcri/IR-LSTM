import os.path as path
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import misc
import math 
def drop_timepoints(_frame):
    """ Remove problematic rows """
    # exclude timepoint with wrong measurements.        
    misc.censor_d1_table(_frame)

def filter_problematic_subjects(frame, _features):
    # Check data is available or not
    drop_timepoints(frame)
    
    # Sort frame by month
    frame = frame.sort_values(by=['RID', 'Month_bl'])
    frame['round_month'] = frame['Month_bl'].round()
    
    # save csv
    frame.to_csv("./output/data_cleaned.csv", index=False)

columns = ['PTID', 'RID', 'DXCHANGE', 'EXAMDATE', "DX" , "Month_bl"]
features = misc.load_feature("./data/features") 
frame = pd.read_csv("./data/TADPOLE_D1_D2.csv", usecols=columns + features)
filter_problematic_subjects(frame, features.copy()) 

frame = pd.read_csv("./output/data_cleaned.csv")
subjects = np.unique(frame.RID) # np.unique([1, 1, 2, 2, 3, 3]) -> array([1, 2, 3])
subjects_2tp = [rid for rid in subjects if np.sum(frame.RID == rid) >= 2]
frame = frame.set_index('RID')
#Drop subjects
for rid in subjects:
    # exclude subjects with only baseline timepoint
    if rid not in subjects_2tp :
        frame.drop(rid, inplace=True)
        continue

    # Fill missing dianosis
    old_dx = 'NL'
    rec_dx = ''
    list_dx = list(frame.loc[rid]['DX'])
    list_idx_miss = []
    for idx in range(len(list_dx)):
        if list_dx[idx] != list_dx[idx]: # check nan
            list_idx_miss.append(idx)
        else:
            rec_dx = list_dx[idx]
            if len(list_idx_miss) > 0 and rec_dx == old_dx:
                for j in list_idx_miss:
                    frame.loc[rid]['DX'].iloc[[j]] = old_dx
            list_idx_miss = [] #reset
            old_dx = rec_dx
        
    # exlucde subjects with AD at baseline 
    if 'Dementia' in frame.loc[rid]['DX'].iloc[[0]].to_string() :#and 'MCI' not in frame.loc[rid]['DX'].iloc[[0]].to_string(): 
        frame.drop(rid, inplace=True)
        continue     

    max_val = 0
    diagnosis_list = [misc.DX_conv(diag) for diag in frame.loc[rid,  "DX"]]
    for i in range(len(diagnosis_list)):
        if np.isnan(float(diagnosis_list[i])):
            continue
        elif diagnosis_list[i] >= max_val:
            max_val = diagnosis_list[i]
        else:
            frame.drop(rid, inplace=True)
            break
# save data after cleaning
frame.to_csv("./output/data_cleaned.csv", index=True)

