import numpy as np
from src.preprocessing.pre_normaliser import preNormaliser
import pickle
from datetime import datetime

"""
A script to automatically create train and test files, together with the labels and the names for the test files. The data needs to be stored in the folder
src/data_grabbing/raw_data as .zip files skeleton_action_recognition_test_split.zip and skeleton_action_recognition_train_split.zip. 

If run the first time, it creates npy files for the raw data and the train labels, as well as a pkl file containing the test file names in 
src/data_grabbing/refactored_data. This step gets skipped if run again.

Afterwards, the data gets preprocessed (see pre_normaliser.py for details) in accordance to the following flags, and stored in the folder 
src/data_grabbing/preprocessed_data as train_process_data + timestamp.npy (shape: (N,C,T,V,M)), test_process_data + timestamp.npy (shape: (N,C,T,V,M)), 
train_labels + timestamp.nyp (shape: (N)) and test_filenames + timestamp.pkl (string list of length N).
"""


pad = True                  #True to pad null frames, False to do nothing
centre = 1                  #0 to do nothing, 1 to centre the data frame-wise, 2 to centre the data sample-wise
rotate = 0                  #does not work at the moment, rotate = 0, 1 or 2 will not do anything
switchBody = True           #True to fix bodies frequently changing positions, False to do nothing
eliminateSpikes = True      #True to eliminate frames, which move too fast compared to the ones before and after, False to do nothing
scale = 2                   #0 to do nothing, 1 to scale the data frame-wise, 2 to scale it sample-wise. If no scaling is applied, eliminateSpikes does not work and should be False!
parallel = True             #True to use parallel processing, False to not use it
smoothen = True             #True to apply a Savgol filter to smoothen the data, False to do nothing
setPerson0 = 3              #0 to do nothing, 1 to set the person with higher average speed to be person 0. 2 to set the left-most person to be person 0. 3 to add mirrored samples to the data.
confidence = 2              #0 to not augment the data. 1 to augment it in C=4 with the speed of each joint in each frame, compared to the last and next frame. 2 to apply a logarithmic sigmoid functions to the velocities.


now = datetime.now()
date = now.strftime("%d-%m-%Y-%H-%M")

#creating train files:
print("Creating train files:")
pre_normaliser = preNormaliser(pad=pad, centre=centre, rotate=rotate, switchBody =switchBody, eliminateSpikes = eliminateSpikes,
                               scale = scale, parallel = parallel, smoothen = smoothen, setPerson0 = setPerson0, confidence = confidence, flag = 'train')
data = pre_normaliser.train_prenorm_data
labels = pre_normaliser.train_prenorm_label

print( r"Saving preprocessed training data in src/data_grabbing/preprocessed_data" )
np.save(r"src/data_grabbing/preprocessed_data/train_process_data"+date+".npy", data)
np.save(r"src/data_grabbing/preprocessed_data/train_labels"+date+".npy", labels)

#creating test files:
print("\n\n Creating test files:")
pre_normaliser = preNormaliser(pad=pad, centre=centre, rotate=rotate, switchBody=switchBody, eliminateSpikes=eliminateSpikes,
                               scale=scale,parallel=parallel, smoothen=smoothen, setPerson0=setPerson0, confidence=confidence, flag='test')
data = pre_normaliser.test_prenorm_data
names = pre_normaliser.test_filenames

print( r"Saving preprocessed training data in src/data_grabbing/preprocessed_data" )
np.save(r"src/data_grabbing/preprocessed_data/test_process_data"+date+".npy", data)
with open(r"src/data_grabbing/preprocessed_data/test_filenames"+date+".pkl", 'wb') as f:
    pickle.dump(pre_normaliser.test_filenames, f)


