from psfdataset import transforms, PSFDataset, PSFZippedDataset
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import os

class LinConvData:
    N_JOINTS = 17
    N_AXES = 2
    N_PERSONS = 2
    N_TIMESTEPS = 305
    N_CLASSES=155

    def __init__(self, pre_normaliser,transform='landmarks',load_transform=True,debug=False):
        '''
            A preNormaliser object is defined if data is not to be loaded. The path signature features are
            ... directly created using 'self.sig_transform' and saved in 'self.sig_data' (a pair (trainingset, valset)
            ... of class PSFDataset or PSFZippedDataset.
        '''
        self.transform = transform
        self.load_transform = load_transform
        self._debug = debug
        if not load_transform:
            self.pre_normaliser = pre_normaliser
        if self.transform=='landmarks':
            self.train_data, self.train_label, self.val_data, self.val_label=self.load_landmarks()
        elif self.transform=='pairs':
            self.sig_data = self.sig_transform()


    def _load_data(self,fold_idx=0):
        '''
            Will only be applied if data is to be loaded. Loads the data and the fold indices from the prenormaliser.

        '''
        isFastTestSubset = self._debug # --- for preliminary testing purposes only

        data_uav = self.pre_normaliser.train_prenorm_data
        ## data_uav = np.transpose(self.pre_normaliser.train_prenorm_data,(0,2,4,3,1)) # to (N, T, M, V, C) --- for  reshaping see below; this has been changed ER
        label_uav = self.pre_normaliser.train_prenorm_label
        ## data_uav = data_uav.reshape(*data_uav.shape[:2], -1, *data_uav.shape[4:]) --- for reshaping see below; this has been changed ER
        print('Prenormalized data shape:',data_uav.shape)
        print('Prenormalized label shape',label_uav.shape)

        fold_idx_dict = self.pre_normaliser.data_grabber.fold_idx_dict
        train_index = fold_idx_dict[str(fold_idx)]['train'] if not isFastTestSubset else fold_idx_dict[str(fold_idx)]['train'][::100]
        val_index = fold_idx_dict[str(fold_idx)]['val'] if not isFastTestSubset else fold_idx_dict[str(fold_idx)]['val'][::100]

        train_data = data_uav[train_index]
        train_label = label_uav[train_index]
        # train_length = length_uav[train_index]   # see Weixin's notebook: can be used to add the temporal length of a movie as a feature; then self._get_iter(...) has to be adapted

        val_data = data_uav[val_index]
        val_label = label_uav[val_index]
        # val_length = length_uav[val_index]   # see above

        print('Test split data shape', train_data.shape, val_data.shape)

        return train_data, train_label, val_data, val_label

    def _get_iter(self,data,label):
        """
        Returns iterators over the requested
        subset of the data.
        """
        return iter(list(zip(data, label)))

    def load_landmarks(self):
        '''
            Either loads the landmark data from path, or creates from pre_normaliser and saves to file
            The path is ~/scr/preprocessing/LinConvData.
        '''

        path = os.path.dirname(__file__) + "/LinConvData/"
        if self.load_transform:
            print('Load data')
            train_data=np.load(path+'uav_train_landmarks.npy')
            train_label_trans=np.load(path+'uav_train_landmarks_labels.npy')

            val_data=np.load(path+'uav_val_landmarks.npy')
            val_label_trans=np.load(path+'uav_val_landmarks_labels.npy')

            print('reshaped data',train_data.shape, val_data.shape)
            print('to shape',val_data.shape[:3]+(LinConvData.N_JOINTS*LinConvData.N_PERSONS,))

        else:
            print('Create data')

            train_data, train_label, val_data, val_label = self._load_data()
            # transforming data [N,C,T,V,M] to [N,T,V,M,C] to [N,T,C,V*M]
            train_data = np.transpose(train_data,(0,2,1,3,4))
            train_data=train_data.reshape(train_data.shape[:3] + (LinConvData.N_JOINTS*LinConvData.N_PERSONS,))
            val_data = np.transpose(val_data,(0,2,1,3,4))
            val_data=val_data.reshape(val_data.shape[:3]+(LinConvData.N_JOINTS*LinConvData.N_PERSONS,))
            train_label_trans = to_categorical(train_label,self.N_CLASSES)
            val_label_trans =to_categorical(val_label,self.N_CLASSES)

            print('reshaped data',train_data.shape, val_data.shape)
            print('to shape',val_data.shape[:3]+(LinConvData.N_JOINTS*LinConvData.N_PERSONS,))

            print('Save landmark data')
            np.save(path+'uav_train_landmarks.npy', train_data)
            np.save(path+'uav_train_landmarks_labels.npy', train_label_trans)

            np.save(path+'uav_val_landmarks.npy', val_data)
            np.save(path+'uav_val_landmarks_labels.npy', val_label_trans)


        return train_data,train_label_trans, val_data, val_label_trans

    def sig_transform(self):
        '''
            Either loads the data from path, or creates the desired PSFDataset object (trainingset, validation set). These
            ... objects contain the desired PSF transform. Then trainingset and validation set are filled with the data so that
            ... they are ready to be used by torch for training and validation. In the second case, data will be saved
            ... in the path. The path is ~/scr/preprocessing/PSF.
        '''

        path = os.path.dirname(__file__) + "/LinConvData/"
        if self.load_transform:
            print('Load data')
            trainingset = PSFDataset()
            valset = PSFDataset()
            trainingset.load(path+"uav_train_"+self.transform)
            valset.load(path+"uav_val_"+self.transform)
        else:
            print('Create data')
            if self.transform=='pairs':
                tr = transforms.Compose([
                        transforms.spatial.Tuples(2)
                        
                ])
                trainingset = PSFDataset(transform=tr)
                valset = PSFDataset(transform=tr)

                train_data, train_label, val_data, val_label = self._load_data()
                # transforming data [N,C,T,V,M] to [N,T,V,M,C] to [N,T,V*M,C]
                train_data = np.transpose(train_data,(0,2,3,4,1))
                train_data = train_data.reshape(*train_data.shape[:2],-1,*train_data.shape[4:])
                val_data = np.transpose(val_data,(0,2,3,4,1))
                val_data = val_data.reshape(*val_data.shape[:2],-1,*val_data.shape[4:])
                if self._debug:
                # consider only the 5 first joints of the first person
                    train_data = train_data[:,:,:5,:]
                    val_data = val_data[:,:,:5,:]
                print('Training data shape:',train_data.shape)
                print('Validation data shape:',val_data.shape)
                trainingset.fill_from_iterator(self._get_iter(train_data, train_label))
                valset.fill_from_iterator(self._get_iter(val_data, val_label))
            else:
                raise NotImplementedError

            print('Save PSF.')
            trainingset.save(path+"uav_train_"+self.transform)
            valset.save(path+"uav_val_"+self.transform)

        print("Number of trainingset elements:", len(trainingset))
        print("Number of validation set elements", len(valset))
        print("Dimension of feature vector:", trainingset.get_data_dimension())

        return trainingset, valset
