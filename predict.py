import torch
import pandas as pd 
import numpy as np

from model import *
import dataloader
import evaluation_metrics 
import misc

num_classes = 3
batch_size = 128

if __name__=="__main__":
    
    list_mauc = []
    list_bca = []
    list_mae_u = []
    list_mae_fore = []
    for fold in range(5):
        model = torch.load("./checkpoint/fold_{}_model_best_lstm_refined_v1.pt".format(fold))
        #print(model)
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        model.to(device)
        if device:
            print(" GPU is activated")
        else:
            print(" CPU is activated")

        model.eval()

        fields = misc.load_feature("./data/features")
        fields.remove("ICV")
        columns = ['RID', 'Month_bl', 'DX'] + fields
        # Load val data
        frame = misc.load_table("./output/data_cleaned.csv", columns)
        mask_dir = "./output/fold{}_mask.csv".format(fold)
        _, _, test_mask = misc.get_mask(mask_dir) # len(train_mask) = len(pred_mask)
        frame_test = frame.loc[test_mask, columns]

        # Normalize and standardize testing set.
        frame_test[fields] = (frame_test[fields] - model.mean) /model.std

        data_test = dataloader.extract(frame_test, fields)
        test_batches, test_mask_batches, test_delta_batches, test_subjects = dataloader.spilit_batch(data_test, batch_size)

        true_dia, pred_dia = [], []
        for i in range(len(test_batches)):
            # Forward pass : Compute predicted y by passing train data to the model
            pred_cat_val, u_seq, forecast_seq = model.predict(test_batches[i],  test_mask_batches[i])
            true, pred = evaluation_metrics.postprocess(pred_cat_val[:,:, :num_classes], test_batches[i][1:, :, :num_classes], test_mask_batches[i][1:, :, :num_classes])
            true_dia.extend(list(true))
            pred_dia.extend(list(pred))

            mae_u = evaluation_metrics.mae(u_seq, test_batches[i][:-1, :, num_classes:], test_mask_batches[i][:-1, :, num_classes:])  
            list_mae_u.append(mae_u) 
            mae_fore = evaluation_metrics.mae(forecast_seq, test_batches[i][1:, :, num_classes:], test_mask_batches[i][1:, :, num_classes:]) 
            list_mae_fore.append(mae_fore) 
        
        true_pred_pairing = [(true_dia[i], pred_dia[i]) for i in range(len(true_dia)) ]
        mauc = evaluation_metrics.MAUC(true_pred_pairing, no_classes=num_classes)
        bca = evaluation_metrics.calcBCA(np.argmax(pred_dia, axis=1), np.asarray(true_dia), no_classes=num_classes)

        list_mauc.append(mauc)
        list_bca.append(bca)
        print('Fold {}  mAUC val: {:.5f}| BCA val: {:.5f}'.format(fold, mauc, bca))

    list_mauc = np.asarray(list_mauc)
    list_bca = np.asarray(list_bca)
    list_mae_u = np.asarray(list_mae_u)
    list_mae_fore = np.asarray(list_mae_fore)
    print()
    print("mAUC mean/ std: {:.5f} - {:.5f}".format(list_mauc.mean(), list_mauc.std() ))
    print("BCA mean/  std: {:.5f} - {:.5f}".format(list_bca.mean(), list_bca.std() ))
    print("MAEu mean/ std: {:.5f} - {:.5f}".format(list_mae_u.mean(), list_mae_u.std() ))
    print("MAEf mean/ std: {:.5f} - {:.5f}".format(list_mae_fore.mean(), list_mae_fore.std() ))
