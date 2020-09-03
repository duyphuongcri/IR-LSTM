import numpy as np
import pandas as pd 
from datetime import datetime

def get_mask(csv_path):
    """ Get masks from CSV file """
    columns = ['RID', 'EXAMDATE', 'train', 'val', 'test']
    frame = pd.read_csv(csv_path, usecols=columns)
    train_mask = frame.train == 1
    pred_mask = frame.val == 1
    test_mask = frame.test == 1
    return train_mask, pred_mask, test_mask

def has_data_mask(frame):
    """
    Check whether rows has any valid value (i.e. not NaN)
    Args:
        frame: Pandas data frame
    Return:
        (ndarray): boolean mask with the same number of rows as *frame*
        True implies row has at least 1 valid value
    """
    return ~frame.isnull().apply(np.all, axis=1)
    
def get_data_dict(frame, features):
    """
    From a frame of all subjects, return a dictionary of frames
    The keys are subjects' ID
    The data frames are:
        - sorted by *Month_bl* (which are integers)
        - have empty rows dropped (empty row has no value in *features* list)
    Args:
        frame (Pandas data frame): data frame of all subjects
        features (list of string): list of features
    Return:
        (Pandas data frame): prediction frame
    """
    ret = {}
    frame_ = frame.copy()
    frame_['Month_bl'] = frame_['Month_bl'].round().astype(int)
    for subj in np.unique(frame_.RID):
        subj_data = frame_[frame_.RID == subj].sort_values('Month_bl')
        subj_data = subj_data[has_data_mask(subj_data[features])]
        subj_data = subj_data.set_index('Month_bl', drop=True)
        ret[subj] = subj_data.drop(['RID'], axis=1)
    return ret

def censor_d1_table(_table):
    """ Remove problematic rows """
    _table.drop(3229, inplace=True)  # RID 2190, Month = 3, Month_bl = 0.45
    _table.drop(4372, inplace=True)  # RID 4579, Month = 3, Month_bl = 0.32
    _table.drop(
        8376, inplace=True)  # Duplicate row for subject 1088 at 72 months
    _table.drop(
        8586, inplace=True)  # Duplicate row for subject 1195 at 48 months
    _table.loc[
        12215,
        'Month_bl'] = 48.  # Wrong EXAMDATE and Month_bl for subject 4960
    _table.drop(10254, inplace=True)  # Abnormaly small ICV for RID 4674
    _table.drop(12245, inplace=True)  # Row without measurements, subject 5204


def load_table(csv, columns):
    """ Load CSV, only include *columns* """
    table = pd.read_csv(csv, converters=CONVERTERS, usecols=columns)
    #censor_d1_table(table)

    return table

def load_feature(feature_file_path):
    """
    Load list of features from a text file
    Features are separated by newline
    """
    return [l.strip() for l in open(feature_file_path)]


def to_categorical(y, nb_classes):
    """ Convert list of labels to one-hot vectors """
    if len(y.shape) == 2:
        y = y.squeeze(1)

    ret_mat = np.full((len(y), nb_classes), np.nan)
    good = ~np.isnan(y)

    ret_mat[good] = 0
    ret_mat[good, y[good].astype(int)] = 1. # = [1. 0. 0.]
    return ret_mat


def PET_conv(value):
    '''Convert PET measures from string to float '''
    try:
        return float(value.strip().strip('>'))
    except ValueError:
        return float(np.nan)


def Diagnosis_conv(value):
    '''Convert diagnosis from string to float '''
    if value == 'CN':
        return 0.
    if value == 'MCI':
        return 1.
    if value == 'AD':
        return 2.
    return float('NaN')


def DX_conv(value):
    '''Convert change in diagnosis from string to float '''
    if isinstance(value, str):
        if value.endswith('Dementia'):
            return 2.
        if value.endswith('MCI'):
            return 1.
        if value.endswith('NL'):
            return 0.

    return float('NaN')


def str2date(string):
    """ Convert string to datetime object """
    return datetime.strptime(string, '%Y-%m-%d')

# Converters for columns with non-numeric values
CONVERTERS = {
    'CognitiveAssessmentDate': str2date,
    'ScanDate': str2date,
    'Forecast Date': str2date,
    'EXAMDATE': str2date,
    'Diagnosis': Diagnosis_conv,
    'DX': DX_conv,
    'PTAU_UPENNBIOMK9_04_19_17': PET_conv,
    'TAU_UPENNBIOMK9_04_19_17': PET_conv,
    'ABETA_UPENNBIOMK9_04_19_17': PET_conv
}