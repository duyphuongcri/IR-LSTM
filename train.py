import torch
import pandas as pd 
import numpy as np 
from tqdm import tqdm
import os 

import dataloader
from model import *
import loss
import evaluation_metrics
import misc


batch_size = 128
lrate = 0.01
weight_decay=5e-7
num_classes = 3
epochs = 100
mode = "lstm_refined_v1"  #   lstm_refined_v2

if __name__=="__main__":

    fields = misc.load_feature("./data/features")
    fields.remove("ICV")
    columns = ['RID', 'Month_bl', 'DX'] + fields
    frame = misc.load_table("./output/data_cleaned.csv", columns)

    for fold in tqdm(range(5)):
        mask_dir = "./output/fold{}_mask.csv".format(fold)
        train_mask, val_mask, test_mask = misc.get_mask(mask_dir) # len(train_mask) = len(pred_mask)
        frame_train = frame.loc[train_mask, columns]
        frame_val = frame.loc[val_mask, columns]

        # Normalize and standardize dataset.
        mean = frame_train[fields].mean()
        std = frame_train[fields].std()
        frame_train[fields] = (frame_train[fields] - mean) / std
        frame_val[fields] = (frame_val[fields] - mean) / std

        data_train = dataloader.extract(frame_train, fields)
        data_val = dataloader.extract(frame_val, fields)
        # Split batch
        data_batches, mask_batches, delta_batches, subjects = dataloader.spilit_batch(data_train, batch_size)
        val_batches, val_mask_batches, val_delta_batches, val_subjects = dataloader.spilit_batch(data_val, batch_size)
        #print(len(subjects), len(val_subjects))
        verbose = True
        log = print if verbose else lambda *x, **i: None
        np.random.seed(10)
        torch.manual_seed(10)

        if 'lstm_refined_v1' in mode:
            model = Refine_LSTM_v1(input_size=len(fields) + num_classes, hidden_size=256, output_size=len(fields)+ num_classes, mean=mean, std=std)
        if 'lstm_refined_v2' in mode:
            model = Refine_LSTM_v2(input_size=len(fields) + num_classes, hidden_size=256, output_size=len(fields)+ num_classes, mean=mean, std=std)

        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        model.to(device)
        if device:
            print(" GPU is activated")
        else:
            print(" CPU is activated")

        log(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=lrate, weight_decay=weight_decay)
        best_val_acc = 0
        for epoch in range(epochs):
            # ent_train_loss, mae_train_loss, total_train_loss = [], [], []
            total_ent_train = total_mae_train = 0

            model.train()
            #with torch.autograd.set_detect_anomaly(True):
            for i in range(len(data_batches)):
                optimizer.zero_grad()
                # Forward pass : Compute predicted y by passing train data to the model
                pred_cat_val, x_seq, z_seq, u_seq = model(data_batches[i], mask_batches[i], delta_batches[i]) # lstm

            
                #print(pred_cat_val)
                ent = loss.ent_loss(pred_cat_val[:,:, :num_classes], data_batches[i][1:, :, :num_classes], mask_batches[i][1:, :, :num_classes])
                mae = loss.mae_loss(pred_cat_val[:,:, num_classes:], data_batches[i][1:, :, num_classes:], mask_batches[i][1:, :, num_classes:])

  
                loss_x = loss.diag_bio_loss(x_seq, data_batches[i], mask_batches[i], num_classes)
                loss_z = loss.diag_bio_loss(z_seq, data_batches[i], mask_batches[i], num_classes)
                loss_u = loss.diag_bio_loss(u_seq, data_batches[i], mask_batches[i], num_classes)
                total_loss = ent + mae + (loss_x + loss_z + loss_u)

                total_loss.backward()
                optimizer.step()

                total_ent_train += ent.item() * data_batches[i].shape[1]
                total_mae_train += mae.item() * data_batches[i].shape[1]
                
            # Val\---------------------------------------------------------------------------------------------------------------
            ent_val_loss, mae_val_loss, total_val_loss = [], [], []
            total_ent_val = total_mae_val = 0
            true_dia, pred_dia = [], []

            model.eval()
            for i in range(len(val_batches)):
                # Forward pass : Compute predicted y by passing train data to the model
                pred_cat_val, u_seq, forecast_seq = model.predict(val_batches[i], val_mask_batches[i])

                ent_val = loss.ent_loss(pred_cat_val[:,:, :num_classes], val_batches[i][1:, :, :num_classes], val_mask_batches[i][1:, :, :num_classes])
                mae_val = loss.mae_loss(pred_cat_val[:,:, num_classes:], val_batches[i][1:, :, num_classes:], val_mask_batches[i][1:, :, num_classes:])
                total_ent_val += ent_val.item() * val_batches[i].shape[1]
                total_mae_val += mae_val.item() * val_batches[i].shape[1]
                # evaluate
                true, pred = evaluation_metrics.postprocess(pred_cat_val[:,:, :num_classes], val_batches[i][1:, :, :num_classes], val_mask_batches[i][1:, :, :num_classes])
                true_dia.extend(list(true))
                pred_dia.extend(list(pred))

            true_pred_pairing = [(true_dia[i], pred_dia[i]) for i in range(len(true_dia)) ]
            mauc_val = evaluation_metrics.MAUC(true_pred_pairing, no_classes=num_classes)
            bca_val = evaluation_metrics.calcBCA(np.argmax(pred_dia, axis=1), np.asarray(true_dia), no_classes=num_classes)

            log_info = (epoch + 1, epochs,  total_ent_train / len(subjects), total_mae_train / len(subjects), \
                                            total_ent_val / len(val_subjects), total_mae_val / len(val_subjects), \
                                            mauc_val, bca_val)
            log('%d/%d  ENT_train %.3f| MAE_train %.3f| ENT_val %.3f| MAE_val %.3f| mAUC val %.5f| BCA val %.5f' % log_info)

            # save model
            is_best = mauc_val >= best_val_acc
            if is_best:
                print("saving model")
                torch.save(model, "./checkpoint/fold_{}_model_best_{}.pt".format(fold, mode))
                best_val_acc = mauc_val






    
