import numpy as np
from src.preprocessing.pre_normaliser import preNormaliser
import pickle


#creating train files:
pre_normaliser = preNormaliser(pad=True, centre=1, rotate=0, switchBody =True, eliminateSpikes = True, scale = 2, parallel = True, smoothen = True,
                 setPerson0 = 2, confidence = 0, flag = 'train')
data = pre_normaliser.train_prenorm_data
labels = pre_normaliser.train_prenorm_label

print( r"Saving preprocessed training data in src/data_grabbing/preprocessed_data" )
np.save(r"src/data_grabbing/preprocessed_data/train_process_data.npy", data)
np.save(r"src/data_grabbing/preprocessed_data/labels.npy", labels)

#creating test files:
pre_normaliser = preNormaliser(pad=True, centre=1, rotate=0, switchBody=True, eliminateSpikes=True, scale=2,parallel=True, smoothen=True,
                    setPerson0=2, confidence=0, flag='test')
data = pre_normaliser.test_prenorm_data
names = pre_normaliser.test_filenames

print( r"Saving preprocessed training data in src/data_grabbing/preprocessed_data" )
np.save(r"src/data_grabbing/preprocessed_data/test_process_data.npy", data)
with open(r"src/data_grabbing/preprocessed_data/test_filenames.pkl", 'wb') as f:
    pickle.dump(pre_normaliser.test_filenames, f)


