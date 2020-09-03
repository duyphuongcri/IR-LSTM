import torch
import numpy as np 

def ent_loss(pred, true, mask):
    """
    Calculate cross-entropy loss
    Args:
        pred: predicted probability distribution,
              [nb_timpoints, nb_subjects, nb_classes]
        true: true class, [nb_timpoints, nb_subjects, nb_classes]
        mask: timepoints to evaluate, [nb_timpoints, nb_subjects, nb_classes]
    Returns:
        cross-entropy loss
    """
    nb_subjects = true.shape[1]
    mask_ = np.full(mask.shape[:2], True)
    true_ = np.full(true.shape[:2], np.nan)
    
    #print(mask_.shape)
    #mask_[true[:][:][0] == True] = False
    #true_[true[:][:][0] == False] = np.argmax(true)
    for i in range(len(mask)):
        for j in range(mask.shape[1]):
            if np.isnan(true[i][j][0]):
                mask_[i][j] = False
            else:
                true_[i][j] = np.argmax(true[i][j])
#             print("a", mask[i][j], mask_[i][j], true[i][j], true_[i][j])
    #print(true.shape) #(max_timepoint, batch_size, 1)
    pred = pred.reshape(pred.size(0) * pred.size(1), -1)
    mask_ = mask_.reshape(-1, 1)
    #print("mask: ",mask_)
    
    o_true = pred.new_tensor(true_.reshape(-1, 1)[mask_], dtype=torch.long)
    o_pred = pred[pred.new_tensor(mask_.squeeze(1).astype(np.uint8), dtype=torch.bool)]

    nb_timepoints = o_true.shape[0]

    return torch.nn.functional.cross_entropy(o_pred, o_true, reduction='sum') / nb_subjects


def mae_loss(pred, true, mask):
    """
    Calculate mean absolute error (MAE)
    Args:
        pred: predicted values, [nb_timpoints, nb_subjects, nb_features]
        true: true values, [nb_timpoints, nb_subjects, nb_features]
        mask: values to evaluate, [nb_timpoints, nb_subjects, nb_features]
    Returns:
        MAE loss
    """
    assert isinstance(pred, torch.Tensor)
    assert isinstance(true, np.ndarray) and isinstance(mask, np.ndarray)
    nb_subjects = true.shape[1]

    invalid = ~mask
    true[invalid] = 0
    indices = pred.new_tensor(invalid.astype(np.uint8), dtype=torch.bool)
    assert pred.shape == indices.shape
    pred[indices] = 0

    return torch.nn.functional.l1_loss(pred, pred.new(true), reduction='sum') / nb_subjects

def diag_bio_loss(pred, true, mask, num_classes):
    return ent_loss(pred[:,:, :num_classes], true[:,:, :num_classes], mask[:,:, :num_classes]) + \
             mae_loss(pred[:,:, num_classes:], true[:,:, num_classes:], mask[:,:, num_classes:])