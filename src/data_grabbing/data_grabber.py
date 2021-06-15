import os
import numpy as np
import pickle
from tqdm import tqdm
import math
from joblib import Parallel, delayed

# TODO: debug (follow Weixin's original code)


class DataGrabber:
    '''
        raw data stored in MMVRC_ICCV_2021_Skeleton_based_Action_Recognition/data_grabbing/raw_data
    '''
    def __init__(self):
        self.dataset_folder_path = "".join(__file__.split("/")[:-1]) + "/raw_data/"
        self.train_data, self.train_label = self._cross_validation()

    def _load_uav_data(self, flag = 'train'):
        data_uav = np.load(os.path.join(self.dataset_folder_path,'{}_raw_data.npy'.format(flag)))#mmap_mode = None, 'r‘
        N,C,T,V,M = data_uav.shape
        #print(N,C,T,V,M)

        with open(os.path.join(self.dataset_folder_path,'{}_label.pkl'.format(flag)), 'rb') as f:
            sample_name, label_uav = pickle.load(f)
        
        label_uav = np.array(label_uav)
        #print(label_uav.shape)
        return data_uav,label_uav

    def _cross_validation(self):
        data_uav, label_uav = self._load_uav_data()
        length_uav = lengths

        class_num = 155
        cross_val_fold_num = 5

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
        for k, (train, val) in enumerate(k_fold.split(data_uav,label_uav)):
            k_fold_idx_dict.update({str(k):{'train':train, 'val':val}})
            print(k+1,'- fold:','Trainset size: ',len(train),' Valset size: ',len(val))

    def _get_data_batch(self):
        raise NotImplementedError

    def _get_all_data(self):
        raise NotImplementedError

if __name__ == "__main__":
    obj = DataGrabber()
