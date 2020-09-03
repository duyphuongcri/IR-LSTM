import numpy as np
import pandas as pd 
import misc
import os

def gen_mask_frame(data, train, val, test):
    """
    Create a frame with 3 masks:
        train: timepoints used for training model
        val: timepoints used for validation
        test: timepoints used for testing model
    """
    col = ['RID', 'EXAMDATE']
    ret = pd.DataFrame(data[col], index=range(train.shape[0]))
    ret['train'] = train
    ret['val'] = val
    ret['test'] = test

    return ret  

def gen_fold(data, nb_folds, outdir):
    """ Generate *nb_folds* cross-validation folds from *data """
    subjects = np.unique(data.RID) # np.unique([1, 1, 2, 2, 3, 3]) -> array([1, 2, 3])
    # Check subject has more than two visits
    has_2tp = np.array([np.sum(data.RID == rid) >= 2 for rid in subjects])
    potential_targets = np.random.permutation(subjects[has_2tp])
    # 20% for testing
    threshold_point = int(len(potential_targets)/nb_folds)
    test_subj = potential_targets[:threshold_point]
    train_val_subj = potential_targets[threshold_point:]
    # 80% train- 20 % valid ( in remain 80%)
    folds = np.array_split(train_val_subj, nb_folds)

    leftover = [subjects[~has_2tp]]
    for val_fold in range(nb_folds):
        #val_fold = (test_fold + 1) % nb_folds
        train_folds = [i for i in range(nb_folds) if (i != val_fold)]

        train_subj = np.concatenate(
            [folds[i] for i in train_folds] + leftover, axis=0)
        val_subj = folds[val_fold]

        #train_timepoints = (np.in1d(data.RID, train_subj) & data.has_data).astype(int)
        #val_timepoints = (np.in1d(data.RID, val_subj) & data.has_data).astype(int)
        train_timepoints = np.in1d(data.RID, train_subj).astype(int)
        val_timepoints = np.in1d(data.RID, val_subj).astype(int)
        test_timepoints = np.in1d(data.RID, test_subj).astype(int)


        mask_frame = gen_mask_frame(data, train_timepoints, val_timepoints, test_timepoints)
        mask_frame.to_csv(os.path.join(outdir, 'fold%d_mask.csv' % val_fold), index=False)

        print("Number of subjects for training fold {}: {}".format(val_fold, len(train_subj)))
        print("Number of subjects for validation fold {}: {}".format(val_fold, len(val_subj)))
        print()

    print("Number of subjects for testing: {}".format(len(test_subj)))

if __name__=="__main__":

    columns = ['RID', 'DXCHANGE', 'EXAMDATE']
    features = misc.load_feature("./data/features") 
    frame = pd.read_csv("./output/data_cleaned.csv",usecols=columns + features, converters=misc.CONVERTERS)
    gen_fold(frame, 5, "./output")