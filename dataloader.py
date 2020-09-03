import numpy as np 
import pandas as pd 
import misc

def extract(frame, features):
    """
    Extract and interpolate time series for each subject in data frame
    Args:
        frame (Pandas frame): input data frame
        strategy (string): name of the interpolation strategy
        features (list of strings): list of features
        defaults (dict): contains default value for each feature
    Returns:
        ret (dict): contain 3 arrays for each subject
            - input: input to RNN/LSS
            - mask: boolean mask indicating availability of ground truth
            - truth: ground truth values
        fields (list of strings):
    """
    nb_classes = 3
    #fields = ['Month_bl', 'DX'] + features
    fields = ['DX'] + features
    ret = dict()
    for rid, sframe in misc.get_data_dict(frame, fields).items():
        #print(rid)
        xin = sframe.index.values # thang [0 12 24 36 60 ...]
        assert len(xin) == len(set(xin)), rid
        #print(xin, rid)
        xin -= xin[0] 
        xout = np.arange(xin[-1] - xin[0] + 1)
        
        in_cat_val_t = np.full((len(xout), len(features) + nb_classes), np.nan, np.float)
        mk_cat_cal_t = np.zeros((len(xout), len(features) + nb_classes),  dtype=bool)
        #delta_time_t = np.zeros((len(xout), len(features) + nb_classes))
        
        sframe = sframe.to_numpy()
        it = np.nditer(xout, flags=['f_index'])
        n = 0
        #last_observed_time = np.zeros(len(features) + nb_classes)
        while not it.finished:
            #delta_time_t[it.index]
#             if n > 0:
#                 print(rid, "miss", it.index-xin[n-1], "it", it.index, n)
#             else:
#                 print(rid, "miss", 0, "it", it.index, n)
            if it.index in xin:
                categorical_class = np.squeeze(misc.to_categorical(sframe[n, :1], nb_classes))     
                in_cat_val_t[it.index][:] = np.concatenate((categorical_class, sframe[n, 1:]))
                mk_cat_cal_t[it.index][:] = ~np.isnan(in_cat_val_t[it.index][:])
                n += 1

#             for f in range(nb_classes + len(features)):
#                 delta_time_t[it.index][f] = it.index - last_observed_time[f]
#                 if mk_cat_cal_t[it.index][f] == True:
#                     last_observed_time[f] = it.index
            it.iternext()
        
        if  np.isnan(in_cat_val_t[0][0]): 
            #print("d", in_cat_val_t[0][:3]) # DX first timepoint
            in_cat_val_t[0][:3] = 0.
            in_cat_val_t[0][0] = 1 # NC
        assert in_cat_val_t.shape == mk_cat_cal_t.shape, rid
        ret[rid] = {'input':in_cat_val_t}
        ret[rid]['mask'] = mk_cat_cal_t
        #ret[rid]['delta_time'] = delta_time_t
    #print(ret.keys())
    return ret

def spilit_batch(data, batch_size):
    list_subjects = []
    for i in data.keys():
        list_subjects.append(i)
    subjects = np.sort(np.array(list_subjects))
    #print(len(subjects))
  
    data_batches = []
    mask_batches = []
    delta_batches = []
    

                
    for i in range(int(np.ceil(len(subjects) / batch_size))):
        
        rid_list = subjects[i*batch_size:(i + 1)*batch_size] # len = 128
        maxlen = max([len(data[rid]['input']) for rid in rid_list])
        
        in_cat_val_pad = [
                        np.pad(data[rid]['input'], [(0, maxlen - len(data[rid]['input'])), (0, 0)], 'constant',constant_values=(np.nan))[:, None, :] 
                        for rid in rid_list
                        ]
        in_cat_val_batch = np.concatenate(in_cat_val_pad, axis=1)
        
        mk_cat_val_pad = [
                        np.pad(data[rid]['mask'], [(0, maxlen - len(data[rid]['mask'])), (0, 0)], 'constant')[:, None, :] 
                        for rid in rid_list
                        ]
        mk_cat_val_batch = np.concatenate(mk_cat_val_pad, axis=1)
        
        
        #print(in_cat_val_batch.shape, mk_cat_val_batch.shape)
        ###################
        delta_time_batch = np.zeros(mk_cat_val_batch.shape)
        #print(delta_time_batch.shape)
        for idx_subj, rid in enumerate(rid_list):
            #print(data[rid]['mask'].shape, len(data[rid]['mask']))
            last_observed_time = np.zeros(in_cat_val_batch.shape[-1])
            #print(last_observed_time.shape)
            for idx_tps in range(maxlen):
                for f in range(in_cat_val_batch.shape[-1]):
                    delta_time_batch[idx_tps][idx_subj][f] = idx_tps - last_observed_time[f]
                    if mk_cat_val_batch[idx_tps][idx_subj][f] == True:
                        last_observed_time[f] = idx_tps
                #print(delta_time_batch[idx_tps][idx_subj])

        data_batches.append(in_cat_val_batch)
        mask_batches.append(mk_cat_val_batch)
        delta_batches.append(delta_time_batch)
#         if in_cat_val_batch.shape[0] == 1:
#             print(rid_list)
#     print(len(data_batches), len(mask_batches))
#     print(len(subjects))
    return data_batches, mask_batches, delta_batches, subjects