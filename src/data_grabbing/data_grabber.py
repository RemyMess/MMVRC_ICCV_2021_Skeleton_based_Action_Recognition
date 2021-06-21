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
        #self.dataset_folder_path = "".join(__file__.split("/")[:-1]) + "/raw_data/" --- OLD VERSION, had a bug
        self.dataset_folder_path = os.path.dirname(__file__) + "/raw_data/"
        self.train_data, self.train_label = self._load_uav_data()
        self.fold_idx_dict = self._cross_validation_fold(cross_val_fold_num=5)

    def _load_uav_data(self, flag = 'train'):
        data_uav = np.load(os.path.join(self.dataset_folder_path,'{}_raw_data.npy'.format(flag)))#mmap_mode = None, 'r‘
        #N,C,T,V,M = data_uav.shape
        #print(N,C,T,V,M)

        with open(os.path.join(self.dataset_folder_path,'{}_label.pkl'.format(flag)), 'rb') as f:
            sample_name, label_uav = pickle.load(f)
        
        label_uav = np.array(label_uav)
        #print(label_uav.shape)
        return data_uav,label_uav

    def _cross_validation_fold(self,data_input=None,cross_val_fold_num=5):
        if data_input is None: 
            data_uav, label_uav = self.train_data, self.train_label
        elif data_input=='load':
            data_uav, label_uav = self._load_uav_data()
        else:
            data_uav, label_uav = data_input
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

    '''
    def _get_data_batch(self):
        raise NotImplementedError

    def _get_all_data(self):
        raise NotImplementedError
        
    ER ???
    '''
    
if __name__ == "__main__":
    obj = DataGrabber()
