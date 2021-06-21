from psfdataset import transforms, PSFDataset, PSFZippedDataset
import numpy as np
import os

class AugmentedSigTransformer:
    def __init__(self, pre_normaliser,transform='example',load_transform=True,debug=False):
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
        self.sig_data = self.sig_transform()

        
    def _load_data(self,fold_idx=0):
        '''
            Will only be applied if data is to be loaded. Loads the data and the fold indices from the prenormaliser.
            Reshapes the data in order to be compatible with PSFDataset.
        '''
        isFastTestSubset = self._debug # --- for preliminary testing purposes only

        data_uav = self.pre_normaliser.train_prenorm_data
        ## data_uav = np.transpose(self.pre_normaliser.train_prenorm_data,(0,2,4,3,1)) # to (N, T, M, V, C) --- for  reshaping see below; this has been changed ER
        label_uav = self.pre_normaliser.train_prenorm_label
        ## data_uav = data_uav.reshape(*data_uav.shape[:2], -1, *data_uav.shape[4:]) --- for reshaping see below; this has been changed ER
        print('Data shape:',data_uav.shape)
        
        fold_idx_dict = self.pre_normaliser.data_grabber.fold_idx_dict
        train_index = fold_idx_dict[str(fold_idx)]['train'] if not isFastTestSubset else fold_idx_dict[str(fold_idx)]['train'][::100]
        val_index = fold_idx_dict[str(fold_idx)]['val'] if not isFastTestSubset else fold_idx_dict[str(fold_idx)]['val'][::100]

        train_data = data_uav[train_index]
        train_label = label_uav[train_index]
        # train_length = length_uav[train_index]   # see Weixin's notebook: can be used to add the temporal length of a movie as a feature; then self._get_iter(...) has to be adapted
        
        val_data = data_uav[val_index]
        val_label = label_uav[val_index]
        # val_length = length_uav[val_index]   # see above
        
        print(train_data.shape, val_data.shape)
        
        return train_data, train_label, val_data, val_label

    def _get_iter(self,data,label): 
        """
        Returns iterators over the requested
        subset of the data.
        """
        return iter(list(zip(data, label)))

    def sig_transform(self):
        '''
            Either loads the data from path, or creates the desired PSFDataset object (trainingset, validation set). These  
            ... objects contain the desired PSF transform. Then trainingset and validation set are filled with the data so that
            ... they are ready to be used by torch for training and validation. In the second, case data will be saved
            ... in the path. The path is ~/scr/preprocessing/PSF.
        '''
             
        path = os.path.dirname(__file__) + "/PSF/"
        if self.load_transform:
            print('Load PSF.')
            trainingset = PSFDataset()
            valset = PSFDataset()
            trainingset.load(path+"uav_train_"+self.transform)
            valset.load(path+"uav_val_"+self.transform)
        else:
            print('Create PSF.')
            if self.transform=='example':
                tr = transforms.Compose([
                        transforms.spatial.Tuples(2),
                        transforms.SpatioTemporalPath(),
                        transforms.temporal.MultiDelayedTransformation(2),
                        transforms.temporal.DyadicPathSignatures(dyadic_levels=1,
                                                                signature_level=3)
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
