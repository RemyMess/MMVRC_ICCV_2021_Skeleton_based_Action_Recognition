from src.algos.sig_classifier import UAVDataset,SigClassifier
from src.preprocessing.pre_normaliser import preNormaliser
from src.data_grabbing.data_grabber import DataGrabber
import os
import numpy as np


class SkeletonBasedActionRecognition:
    '''
        A class for skeleton-based human action recognition on UAV.
        Two methods: 1) '__init__': data preprocessing or loading, building the classifier. 
        ... A priori, several classifiers can be chosen and the device can be specified. 2) 'run' contains training
        ... and validation, as well as printing the confusion matrix.
    '''
    def __init__(self,algo='SigClassifier',batch_size=64,lr=0.005,epochs=20,load_preprocessed_data=False,pad=True,
                 centre=1,rotate=0,switchBody =True,eliminateSpikes = True,scale = 2,parallel = True,
                 smoothen = False,setPerson0 = 2,confidence = 0,data_in_mem=False,path=None,debug=False):
            
            #algo='SigClassifier',batch_size=20,lr=0.0001,epochs=20,transform='example',load_data=True,
            #device='cuda',pad=True,centre=True,rotate=True,debug=False):
        '''
            · Inputs:
            algo: the classifier algorithm to be used. Algorithms can be added as a .py-script in ~/src/algos/. The args
            ... of __init__ may be adapted accordingly. Further algos should also rely on the preNormaliser class. There
            ... should be the option the load the data from a file; see load_data.
            batch_size: batch size for SGD-type optimizers
            lr: learning rate, if applicable
            epochs: epochs, training iterations, if applicable
            transform: a string indicating the set of path signature features to be extracted by sig_data_wrapper. 'example'
            ... is the simple one-shot transform (from the demo notebook).
            preprocess_data: whether to preprocess the data and save it, or to load it. The corresponding path is ~/src/preprocessed_data
            device: the device to be used by torch for training. Change to 'cpu' if no gpu with cuda is available.

            pad: whether to perform the padding procedure (replacing zero frames with valid frames) or not,
            ... see ~/src/pre_processing/pre_normaliser.py
            centre: whether to centre the skeleta around the first person's torso or not, 0 = not, 1 = frame-wise,
            ... 2 = sample-wise, see ~/src/pre_processing/pre_normaliser.py
            rotate: whether to rotate the skeleta such that the vertebrae and clavicles are aligned with the z- and x-axes
            ... or not --DOES NOT WORK RIGHT NOW
            switchBody: whether to switch the bodies in order to reduce the total energy or not,
            ... see ~/src/pre_processing/pre_normaliser.py
            eliminateSpikes: whether to try to find outlying frames and replace them with valid frames,
            ... see ~/src/pre_processing/pre_normaliser.py
            scale: whether to scale the data to fit into the unit square (preserves proportions), 0 = not, 1 = frame-wise,
            ... 2 = sample-wise, see ~/src/pre_processing/pre_normaliser.py
            smoothen: whether to apply a savgol filter to smoothen the data or not, see ~/src/pre_processing/pre_normaliser.py
            
            path: where to save preprocessed data, if None, then it is ./src/preprocessed_data/
            debug: if True, only a small fraction of the frames and joints will be considered.
            
            · Commands:
            Defines a suitable preNormaliser object, if data is not to be loaded.
            Defines the signature data wrapper (an AugmentedSigTransformer object) and the sig_classifier (a SigClassifier
            ... object) for the 'SigClassifier' algo with the specified parameters, see above
        '''
        
        # 1) Define path
        print('Define path.')
        self.data_in_mem = data_in_mem
        if path is None:
            self.path = os.path.join(os.path.dirname(__file__),'preprocessed_data/')
        else:
            self.path = path

        # 2) Define data set (including loading the data and, if not loaded, preprocessing)
        print('\nDefine data...')
        data_grabber = DataGrabber()
        if not load_preprocessed_data:
            print('... preprocessing data,')
            X,Y = data_grabber.load_raw_data('train')
            pre_normaliser = preNormaliser(pad=pad,centre=centre,rotate=rotate,switchBody=switchBody,
                                           eliminateSpikes=eliminateSpikes,scale=scale,parallel=parallel,
                                           smoothen=smoothen,setPerson0=setPerson0,confidence=confidence)
            X,Y = pre_normaliser.pre_normalization(X,Y)
            print('... saving preprocessed data,')
            np.save(os.path.join(self.path,'preprocessed_features.npy'),X)
            np.save(os.path.join(self.path,'labels.npy'),Y)
            print('... preprocessed data saved,')
        else:
            print('... loading preprocessed data,')
            X = np.load(os.path.join(self.path,'preprocessed_features.npy'),mmap_mode='r')
            Y = np.load(os.path.join(self.path,'labels.npy'))
            print('... preprocessed data loaded,')

        print('... defining a folding dict,')
        fold_dict = data_grabber.fold_dict(X,Y)
        n_fold = 0   # could be an input to self.__init__, or one could loop over it in range(5)
        if not debug:
            train_indices = fold_dict[str(n_fold)]['train']
            val_indices = fold_dict[str(n_fold)]['val']
        else:
            train_indices = fold_dict[str(n_fold)]['train'][::100]
            val_indices = fold_dict[str(n_fold)]['val'][::100]

        if data_in_mem:
            print('... defining training and validation data sets.')
            training_set = UAVDataset((X[train_indices],Y[train_indices]),data_in_mem=data_in_mem)
            val_set = UAVDataset((X[val_indices],Y[val_indices]),data_in_mem=data_in_mem)       
        else:
            if not load_preprocessed_data:
                print('... saving data individually,')
                for n in range(Y.shape[0]):
                    datum = np.array([X[n],Y[n]],dtype=np.ndarray)
                    np.save(self.path+'datum_'+str(n),datum)
            X,Y = None, None

            print('... defining training and validation data sets.')
            training_set = UAVDataset((self.path,train_indices),data_in_mem=data_in_mem)
            val_set = UAVDataset((self.path,val_indices),data_in_mem=data_in_mem)             

        # 3) Define algo
        print('\nDefine algo...')
        self.algo = algo
        if algo=='SigClassifier':                        
            self.sig_classifier = SigClassifier(batch_size=batch_size,lr=lr,epochs=epochs,debug=debug)
            print('... building the model.')
            self.sig_classifier.build(training_set,val_set)
        else:
            raise NotImplementedError

    def __call__(self,flag='TEST'):
        '''
            Performs training and validation (and prediction soon). Prints the confusion matrix with respect to the validation set if render_plot == True.
        '''
        if self.algo=='SigClassifier':
            print('Fitting the model {}.'.format(flag))
            self.sig_classifier.fit(flag)
        else:
            raise NotImplementedError

if __name__ == "__main__":
    print('\n   ---   Run 0.   ---\n')
    obj = SkeletonBasedActionRecognition()
    obj()
    print('\n   ---   Run 0 finished.   ---\n')
