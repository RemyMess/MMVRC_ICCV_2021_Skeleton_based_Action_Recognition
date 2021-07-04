import os
import numpy as np
import pickle
from numpy.lib.stride_tricks import as_strided
from sklearn.model_selection import StratifiedKFold
import glob
from tqdm import tqdm
import re
from zipfile import ZipFile

MAX_BODY_TRUE = 2
MAX_BODY_KINECT = 4
NUM_JOINT = 17
MAX_FRAME = 601

FILENAME_REGEX = r'P\d+S\d+G\d+B\d+H\d+UC\d+LC\d+A(\d+)R\d+_\d+'


class DataGrabber:

    '''
        Extracts the raw data from the path into the attributes train_data, train_label as np.arrays of shapes (N,C,T,V,M) 
        ... alias (sample,coordinate, frame alias time, joint alias vertex, person ID) and (N,).
        The attribute raw_dataset_folder_path specifies the raw data path.
        The attribute k_fold_idx_dict is a nested dictionary specifying the indices of cross_val_fold_num=5 stratified
        ... folds of the (input) data, split into a training and a validation set. It has the structure [N_fold][z] where
        ... N_fold ranges over '0',...,str(cross_val_fold_num-1), z ranges over 'train', 'val'.
    '''
    
    def __init__(self, given_flag = 'train'):
        self.raw_dataset_folder_path = os.path.join(os.path.dirname(__file__), "raw_data")
        self.refactored_dataset_folder_path = os.path.join(os.path.dirname(__file__), "refactored_data")

        # A. Check that raw data has been added
        self.check_raw_data_has_been_added()
        # B. Refactor data
        for flag in ['test', 'train']:
            # if no refactored file, factor.
            if not self.data_has_already_been_refactored(flag):
                print(f"DataGrabber: ({flag} data) refactoring the raw {flag} data.")
                self.gen_refactored_data_from_raw(flag)
            else:
                print(f"DataGrabber: ({flag} data) the raw {flag} data has already been refactored. Skipping this step.")

        # C. Load
        print(f"DataGrabber: loading refactored test and train data.")
        self.train_data, self.train_label = self._load_uav_data(given_flag)

        if given_flag == 'train':
            self.fold_idx_dict = self.cross_validation_fold(cross_val_fold_num=5)

    def data_has_already_been_refactored(self, flag):
        return os.path.isfile(os.path.join(self.refactored_dataset_folder_path, self._refactored_data_filename(flag)))

    def check_raw_data_has_been_added(self):
        for flag in ['test', 'train']:
            raw_folder_name = f"skeleton_action_recognition_{flag}_split"
            raw_folder_path = os.path.join(self.raw_dataset_folder_path, raw_folder_name)
            if not os.path.isdir(raw_folder_path):
                if not os.path.isdir(raw_folder_path + ".zip"):
                    print(f"DataGrabber: {flag} zip files detected. Unzipping data.")
                    with ZipFile(raw_folder_path + ".zip", 'r') as zipObj:
                        zipObj.extractall(raw_folder_path)
                else:
                    raise Exception(f"DataGrabber: competition data has not been loaded (no folder '{raw_folder_path}'). Please do the setup in the /datagrabber/README.md.")

    def _load_uav_data(self, flag = 'train'):
        refactored_data_path = os.path.join(self.refactored_dataset_folder_path, f'{self._refactored_data_filename(flag)}')
        data_uav = np.load(refactored_data_path)#mmap_mode = None, 'r‘
        #N,C,T,V,M = data_uav.shape
        #print(N,C,T,V,M)


        # print("inside1")
        # print(refactored_data_path)
        # print(data_uav)

        if flag == 'train':
            with open(os.path.join(self.refactored_dataset_folder_path,'{}_label.pkl'.format(flag)), 'rb') as f:
                sample_name, label_uav = pickle.load(f)

            label_uav = np.array(label_uav)

        else:
            with open(os.path.join(self.refactored_dataset_folder_path, '{}_label.pkl'.format(flag)), 'rb') as f:
                sample_name = pickle.load(f)

            label_uav = sample_name
        # print("inside2")
        # print(self.refactored_dataset_folder_path,'{}_label.pkl'.format(flag))
        # print(label_uav)

        #print(label_uav.shape)
        return data_uav, label_uav

    def _refactored_data_filename(self, split):
        return f"{split}_raw_data.npy"

    def gen_refactored_data_from_raw(self, split: str):
        if split not in ["test", "train"]:
            raise Exception("_gen_data_from_raw (DataGrabber): split can only take values in ['test', 'train']")

        out_folder_path = self.refactored_dataset_folder_path
        raw_data_folder_path = os.path.join(self.raw_dataset_folder_path, f"skeleton_action_recognition_{split}_split", split)

        skeleton_filenames = [os.path.basename(f) for f in glob.glob(os.path.join(raw_data_folder_path, "**.txt"), recursive=True)]

        sample_name = []
        for basename in skeleton_filenames:
            filename = os.path.join(raw_data_folder_path, basename)
            if not os.path.exists(filename):
                raise OSError('%s does not exist!' %filename)
            sample_name.append(filename)

        data = np.zeros((len(sample_name), 3, MAX_FRAME, NUM_JOINT, MAX_BODY_TRUE), dtype=np.float32)
        for i, s in enumerate(tqdm(sample_name)):
            sample = self._read_xyz(s, max_body=MAX_BODY_KINECT, num_joint=NUM_JOINT)
            data[i, :, 0:sample.shape[1], :, :] = sample
        #data = pre_normalization(data)

        np.save('{}/{}_raw_data.npy'.format(out_folder_path, split), data)

        if split != 'test':
            sample_label = []
            for basename in skeleton_filenames:
                label = int(re.match(FILENAME_REGEX, basename).groups()[0])
                sample_label.append(label)

            with open('{}/{}_label.pkl'.format(out_folder_path, split), 'wb') as f:
                pickle.dump((sample_name, list(sample_label)), f)

        if split == 'test':
            with open('{}/{}_label.pkl'.format(out_folder_path, split), 'wb') as f:
                pickle.dump(sample_name, f)

    def _read_skeleton_filter(self, file):
        with open(file, 'r') as f:
            skeleton_sequence = {}
            skeleton_sequence['numFrame'] = int(f.readline())
            skeleton_sequence['frameInfo'] = []
            # num_body = 0
            for t in range(skeleton_sequence['numFrame']):
                frame_info = {}
                frame_info['numBody'] = int(f.readline())
                frame_info['bodyInfo'] = []

                for m in range(frame_info['numBody']):
                    body_info = {}
                    body_info_key = [
                        'bodyID', 'clipedEdges', 'handLeftConfidence',
                        'handLeftState', 'handRightConfidence', 'handRightState',
                        'isResticted', 'leanX', 'leanY', 'trackingState'
                    ]
                    body_info = {
                        k: float(v)
                        for k, v in zip(body_info_key, f.readline().split())
                    }
                    body_info['numJoint'] = int(f.readline())
                    body_info['jointInfo'] = []
                    for v in range(body_info['numJoint']):
                        joint_info_key = [
                            'x', 'y', 'z', 'depthX', 'depthY', 'colorX', 'colorY',
                            'orientationW', 'orientationX', 'orientationY',
                            'orientationZ', 'trackingState'
                        ]
                        joint_info = {
                            k: float(v)
                            for k, v in zip(joint_info_key, f.readline().split())
                        }
                        body_info['jointInfo'].append(joint_info)
                    frame_info['bodyInfo'].append(body_info)
                skeleton_sequence['frameInfo'].append(frame_info)

        return skeleton_sequence

    def _get_nonzero_std(self, s):  # tvc
        index = s.sum(-1).sum(-1) != 0  # select valid frames
        s = s[index]
        if len(s) != 0:
            s = s[:, :, 0].std() + s[:, :, 1].std() + s[:, :, 2].std() # three channels
        else:
            s = 0
        return s

    def _read_xyz(self, file, max_body, num_joint):
        seq_info = self._read_skeleton_filter(file)
        data = np.zeros((max_body, seq_info['numFrame'], num_joint, 3))
        for n, f in enumerate(seq_info['frameInfo']):
            for m, b in enumerate(f['bodyInfo']):
                for j, v in enumerate(b['jointInfo']):
                    if m < max_body and j < num_joint:
                        data[m, n, j, :] = [v['x'], v['y'], v['z']]
                    else:
                        pass

        # select two max energy body
        energy = np.array([self._get_nonzero_std(x) for x in data])
        index = energy.argsort()[::-1][0:MAX_BODY_TRUE]
        data = data[index]

        data = data.transpose(3, 1, 2, 0)
        return data

    def cross_validation_fold(self, data_input=None, cross_val_fold_num=5):
        if data_input is None: 
            data_uav, label_uav = self.train_data, self.train_label
        elif data_input=='load':
            data_uav, label_uav = self._load_uav_data()
        else:
            data_uav, label_uav = data_input
        # length_uav = lengths   ER: ???

        print("wesh")
        print(label_uav)


        class_num = 155

        print('DataGrabber (_cross_validation_fold): Check data imbalance...')
        class_cnt = np.zeros(class_num)
        for l in label_uav:
            class_cnt[l] += 1
        print(class_cnt)
        print('DataGrabber (_cross_validation_fold): Avg sample num: ',class_cnt.mean())
        print('DataGrabber (_cross_validation_fold): Max sample num: ',class_cnt.max())
        print('DataGrabber (_cross_validation_fold): Min sample num: ',class_cnt.min())

        k_fold = StratifiedKFold(cross_val_fold_num)
        k_fold.get_n_splits(data_uav,label_uav)
        k_fold_idx_dict = dict()

        print('DataGrabber (_cross_validation_fold): Create {}-fold for cross validation...'.format(cross_val_fold_num))
        for k, (train_idx, val_idx) in enumerate(k_fold.split(data_uav,label_uav)):
            k_fold_idx_dict.update({str(k):{'train':train_idx, 'val':val_idx}})
            print(k+1,'- fold:','Trainset size: ',len(train_idx),' Valset size: ',len(val_idx))
        return k_fold_idx_dict
    
if __name__ == "__main__":
    obj = DataGrabber()
