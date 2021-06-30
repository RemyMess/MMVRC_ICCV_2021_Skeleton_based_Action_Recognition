import numpy as np
from src.preprocessing.pre_normaliser import preNormaliser
import pickle

raw_data = np.load(r'src\data_grabbing\raw_data\train_raw_data.npy')

pre_normaliser = preNormaliser(confidence = 2)
data = pre_normaliser.train_prenorm_data
labels = pre_normaliser.train_prenorm_label

np.save(r"..\..\MMVRAC_train_split\train_process_data.npy",data)
np.save(r"..\..\MMVRAC_train_split\labels.npy", labels)
