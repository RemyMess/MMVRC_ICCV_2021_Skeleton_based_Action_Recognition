import os
import numpy as np
import pickle
from sklearn.model_selection import StratifiedKFold


class DataGrabber:

    '''
        Extracts the raw data from the path into the attributes train_data, train_label as np.arrays of shapes (N,C,T,V,M) 
        ... alias (sample,coordinate, frame alias time, joint alias vertex, person ID) and (N,).
        The attribute dataset_folder_path specifies the raw data path.
        The attribute k_fold_idx_dict is a nested dictionary specifying the indices of cross_val_fold_num=5 stratified
        ... folds of the (input) data, split into a training and a validation set. It has the structure [N_fold][z] where
        ... N_fold ranges over '0',...,str(cross_val_fold_num-1), z ranges over 'train', 'val'.
    '''
    
    def __init__(self):
        self.dataset_folder_path = os.path.dirname(__file__) + "/raw_data/"

    def load_raw_data(self, flag = 'train'):
        data_uav = np.load(os.path.join(self.dataset_folder_path,'{}_raw_data.npy'.format(flag)),mmap_mode='r') # mmap_mode = None (not 'r')
        #N,C,T,V,M = data_uav.shape
        #print(N,C,T,V,M)

        with open(os.path.join(self.dataset_folder_path,'{}_label.pkl'.format(flag)), 'rb') as f:
            sample_name, label_uav = pickle.load(f)
        
        label_uav = np.array(label_uav)
        #print(label_uav.shape)
        return data_uav,label_uav

    def fold_dict(self,data=None,labels=None,cross_val_fold_num=5):
        if data is None and labels is None:
            data_uav, label_uav = self._load_uav_data()
        else:
            data_uav, label_uav = data, labels
        # length_uav = lengths   ER: ???

        class_num = 155

        print('Check data imbalance...')
        class_cnt = np.zeros(class_num)
        for l in label_uav:
            class_cnt[l] += 1
        print(class_cnt)
        print('Avg sample num: ',class_cnt.mean())
        print('Max sample num: ',class_cnt.max())
        print('Min sample num: ',class_cnt.min())

        k_fold = StratifiedKFold(cross_val_fold_num)
        k_fold.get_n_splits(data_uav,label_uav)
        k_fold_idx_dict = dict()

        print('Create {}-fold for cross validation...'.format(cross_val_fold_num))
        for k, (train_idx, val_idx) in enumerate(k_fold.split(data_uav,label_uav)):
            k_fold_idx_dict.update({str(k):{'train':train_idx, 'val':val_idx}})
            print(k+1,'- fold:','Trainset size: ',len(train_idx),' Valset size: ',len(val_idx))
        return k_fold_idx_dict     

    def fold(self,data,labels,k_fold_idx_dict,fold_idx=0,debug=False):
        '''
            Will only be applied if data is to be loaded. Loads the data and the fold indices from the prenormaliser.
            Reshapes the data in order to be compatible with PSFDataset.
        '''
        isFastTestSubset = debug # --- for preliminary testing purposes only
        
        train_index = k_fold_idx_dict[str(fold_idx)]['train'] if not isFastTestSubset else k_fold_idx_dict[str(fold_idx)]['train'][::100]
        val_index = k_fold_idx_dict[str(fold_idx)]['val'] if not isFastTestSubset else k_fold_idx_dict[str(fold_idx)]['val'][::100]

        train_data = data[train_index]
        train_label = labels[train_index]
        
        val_data = data[val_index]
        val_label = labels[val_index]
        
        print(train_data.shape, val_data.shape)
        
        return train_data, train_label, val_data, val_label

    
if __name__ == "__main__":
    obj = DataGrabber()
