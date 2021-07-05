from src.algos.sig_classifier import UAVDataset,SigClassifier
from src.preprocessing.pre_normaliser import preNormaliser
from src.preprocessing.augmented_sig_transformer import AugmentedSigTransformer
from src.data_grabbing.data_grabber import DataGrabber
from src.csv_writer import csv_writer
import os
import numpy as np
import torch

class SkeletonBasedActionRecognition:
    '''
        A class for skeleton-based human action recognition on UAV.
        Two methods: 1) '__init__': data preprocessing or loading, building the classifier. 
        ... A priori, several classifiers can be chosen and the device can be specified. 2) 'run' contains training
        ... and validation, as well as printing the confusion matrix.
    '''
    def __init__(self,training=True,algo='SigClassifier.LinearNet',load_model_weights=False,
                 batch_size=128,lr=0.005,epochs=100,double_precision=0,
                 transform='example7',load_PSF=True,Gaussian_noise=1,flipping=0,
                 load_preprocessed_data=True,pad=1,centre=1,rotate=0,switchBody=1,eliminateSpikes=1,scale=2,
                 parallel=0,smoothen=0,setPerson0=3,confidence=2,
                 data_in_mem=False,path=None,debug=False):
            
            #algo='SigClassifier',batch_size=20,lr=0.0001,epochs=20,transform='example',load_data=True,
            #device='cuda',pad=True,centre=True,rotate=True,debug=False):
        '''
            · Inputs:
            training=True means training, else: evaluation / testing
            algo: the classifier algorithm to be used. Algorithms can be added as a .py-script in ~/src/algos/. The args
            ... of __init__ may be adapted accordingly. Further algos should also rely on the preNormaliser class. There
            ... should be the option the load the data from a file; see load_data.
            batch_size: batch size for SGD-type optimizers
            lr: learning rate, if applicable
            double_precision: whether to use float64/double or float32/float
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
            flag: string of form 'f1.f2' where f1 is a flag for preprocessing, f2 is a flag for the algo
            debug: if True, only a small fraction of the frames and joints will be considered.
            dtype: No other option than float32 available for now

            · Commands:
            Defines a suitable preNormaliser object, if data is not to be loaded.
            Defines the signature data wrapper (an AugmentedSigTransformer object) and the sig_classifier (a SigClassifier
            ... object) for the 'SigClassifier' algo with the specified parameters, see above
        '''
        self.training = training
        mode = 'train' if training else 'test'
        self.double = double_precision
        self.load_model_weights = load_model_weights or not training
        self.algo_class, self.algo_spec = algo.split('.')
        self.data_in_mem = data_in_mem
        self.pre_process_flag = 'Pad{}Cen{}Rot{}SwB{}ESp{}Sca{}Par{}Smo{}sPz{}Con{}'.format(pad,
                                                                                            centre,
                                                                                            rotate,
                                                                                            switchBody,
                                                                                            eliminateSpikes,
                                                                                            scale,
                                                                                            parallel,
                                                                                            smoothen,
                                                                                            setPerson0,
                                                                                            confidence)
        self.hyperpar_flag = 'BS{}LR{}Epo{}DPr{}'.format(batch_size,lr,epochs,double_precision)
        if self.algo_spec=='LinearNet':
            self.transforms_flag = transform+'GN{}Flp{}'.format(Gaussian_noise,flipping)
        else:
            self.transforms_flag = 'GN{}Flp{}'.format(Gaussian_noise,flipping)
        self.flag = "Model:{}.{}_Par:{}_PreProc:{}_TraFo:{}".format(
                self.algo_class,
                self.algo_spec,
                self.hyperpar_flag,
                self.pre_process_flag,
                self.transforms_flag)

        if self.algo_class == 'SigClassifier':
        
            # 1) Define path
            print('Define path for preprocessed data.')
            if path is None:
                path = os.path.dirname(__file__)
            self.path = os.path.join(path,'preprocessed_data')

            # 2) Define data set (including loading the data, preprocessing (if applicable), PSF calculation (if applicable))
            print('\nDefine data...')
            data_grabber = DataGrabber()
            landmarks_path = os.path.join(self.path,'landmarks',self.pre_process_flag)
            if not load_preprocessed_data:
                print('... preprocessing data,')
                X,Y = data_grabber.load_raw_data(mode)
                pre_normaliser = preNormaliser(pad=pad,centre=centre,rotate=rotate,switchBody=switchBody,
                                               eliminateSpikes=eliminateSpikes,scale=scale,parallel=parallel,
                                               smoothen=smoothen,setPerson0=setPerson0,confidence=confidence)
                X,Y = pre_normaliser.pre_normalization(X,Y)
                print('... saving preprocessed landmark data,')
                os.makedirs(landmarks_path,exist_ok=True)
                np.save(os.path.join(landmarks_path,mode+'_features.npy'),X)
                np.save(os.path.join(landmarks_path,mode+'_labels.npy'),Y)
                print('... preprocessed landmark data saved,')
            else:
                print('... loading preprocessed landmark data,')
                X = np.load(os.path.join(landmarks_path,mode+'_features.npy'),mmap_mode='r')
                Y = np.load(os.path.join(landmarks_path,mode+'_labels.npy'),allow_pickle=True)
                print('... preprocessed landmark data loaded,')
            if not double_precision:
                X = X.astype(np.float32)

            if training:
                print('... defining a folding dict,')
                fold_dict = data_grabber.fold_dict(X,Y)
                n_fold = 0   # could be an input to self.__init__, or one could loop over it in range(5)
                if not debug:
                    # take [::10] temporary for hyperparameter testing
                    train_indices = fold_dict[str(n_fold)]['train']
                    val_indices = fold_dict[str(n_fold)]['val']
                else:
                    train_indices = fold_dict[str(n_fold)]['train'][::100]
                    val_indices = fold_dict[str(n_fold)]['val'][::100]

            if self.algo_spec == 'LinearNet':
                print('... defining training and validation data sets.')
                augmented_sig_transformer = AugmentedSigTransformer(transform=transform,
                                                                    load_transform=load_PSF,
                                                                    N_randomsubset=50,
                                                                    path=os.path.join(self.path,
                                                                                      'path_signatures',
                                                                                      self.pre_process_flag+'_'+
                                                                                      transform),
                                                                    debug=debug)
                # data_in_mem==True; you need huge memory (>200 GB) to handle this;
                # ... that is because psfdataset saves the data as attributes of the PSFDataset instances;
                # ... this could be circumvented by changing the psfdataset code and editing the __getitem__ method instead
                # ... also, the code would be more transparent if data augmentation could be included in __getitem__
                if training:
                    training_set,val_set = augmented_sig_transformer(X[train_indices],
                                                                     Y[train_indices],
                                                                     X[val_indices],
                                                                     Y[val_indices])
                    self.LinearNet_data_dimension = training_set.get_data_dimension()
                else:
                    Y = np.zeros(X.shape[0])
                    test_set = augmented_sig_transformer(val_data=X,val_label=Y)
                    self.LinearNet_data_dimension = test_set.get_data_dimension()
                X,Y = None,None
            else:
                if data_in_mem:
                    print('... defining training and validation data sets.')
                    training_set = UAVDataset((X[train_indices],Y[train_indices]),
                                              self.pre_process_flag,
                                              data_in_mem=data_in_mem)
                    val_set = UAVDataset((X[val_indices],Y[val_indices]),
                                         self.pre_process_flag,
                                         data_in_mem=data_in_mem)       
                else:
                    if not load_preprocessed_data:
                        print('... saving data individually,')
                        for n in range(Y.shape[0]):
                            datum = np.array([X[n],Y[n]],dtype=np.ndarray)
                            np.save(os.path.join(landmarks_path,
                                                 'datum_'+str(n)),
                                    datum)
                    X,Y = None, None

                    print('... defining training and validation data sets.')
                    training_set = UAVDataset((self.path,train_indices),
                                              self.pre_process_flag,
                                              data_in_mem=data_in_mem)
                    val_set = UAVDataset((self.path,val_indices),
                                         self.pre_process_flag,
                                         data_in_mem=data_in_mem)

            # 3) Define algo
            print('\nDefine algo...')
            if self.algo_class=='SigClassifier':
                if self.algo_spec=='LinearNet':
                    if Gaussian_noise==0:
                        augment_std = None
                    else:
                        augment_std = 0.02 * Gaussian_noise
                    self.sig_classifier = SigClassifier(self.algo_spec,flag=self.flag,double_precision=double_precision,
                                                        data_dimension=self.LinearNet_data_dimension,augment_std=augment_std,
                                                        batch_size=batch_size,lr=lr,epochs=epochs,debug=debug)
                elif self.algo_spec=='DeepSigTrafo':
                    self.sig_classifier = SigClassifier(self.algo_spec,double_precision=double_precision,
                                                        sig_depth=3,
                                                        batch_size=batch_size,lr=lr,epochs=epochs,debug=debug)
                print('... building the model.')
                if training:
                    self.sig_classifier.build(training_set,val_set)
                else:
                    self.sig_classifier.build(val_set=test_set)
            else:
                raise NotImplementedError

        else:
            raise NotImplementedError

    def __call__(self):
        '''
            Performs training and validation (and prediction soon). Prints the confusion matrix with respect to the validation set if render_plot == True.
        '''
        if self.algo_class=='SigClassifier':
            path = os.path.join(os.path.dirname(__file__),'models')
            if self.load_model_weights or not self.training:
                if self.pre_process_flag[-5] == 3:
                    load_pre_process_flag = self.pre_process_flag[:-5] + '2' + self.pre_process_flag[-4:]
                    load_flag = "Model:{}.{}_Par:{}_PreProc:{}_TraFo:{}".format(
                        self.algo_class,
                        self.algo_spec,
                        self.hyperpar_flag,
                        load_pre_process_flag,
                        self.transforms_flag)
                self.sig_classifier.load(os.path.join(path,load_flag))
            os.makedirs(path,exist_ok=True)
            if self.training:
                print('\n >>> Fitting {}. <<<'.format(self.flag))  
                self.sig_classifier.fit(plot_path=path)
            else:
                predictions = self.sig_classifier.predict().astype(np.int16)
                csv_writer(predictions,os.path.join(path,self.flag)+'_predictions.csv')
            # Save model data
            if self.training:
                self.sig_classifier.save(os.path.join(path,self.flag))
        else:
            raise NotImplementedError

if __name__ == "__main__":
    '''
    print('\n   ---   Run 1.   ---\n')
    obj = SkeletonBasedActionRecognition(training=1,load_PSF=False,debug=True)
    obj()
    obj = SkeletonBasedActionRecognition(training=0,load_PSF=False,debug=True,setPerson0=2)
    obj()
    print('\n   ---   Run 1 finished.   ---\n')
    '''
    print('\n   ---   Run 2.   ---\n')
    obj = SkeletonBasedActionRecognition(training=1,load_preprocessed_data=False,load_PSF=False)
    obj()
    obj = SkeletonBasedActionRecognition(training=0,load_preprocessed_data=False,load_PSF=False)
    obj()
    print('\n   ---   Run 2 finished.   ---\n')
    print('\n   ---   Run 3.   ---\n')
    obj = SkeletonBasedActionRecognition(training=1,lr=0.02)
    obj()
    obj = SkeletonBasedActionRecognition(training=0,lr=0.02)
    obj()
    print('\n   ---   Run 3 finished.   ---\n')
    print('\n   ---   Run 4.   ---\n')
    obj = SkeletonBasedActionRecognition(training=1,lr=0.001)
    obj()
    obj = SkeletonBasedActionRecognition(training=0,lr=0.001)
    obj()
    print('\n   ---   Run 4 finished.   ---\n')
    print('\n   ---   Run 5.   ---\n')
    obj = SkeletonBasedActionRecognition(training=1,batch_size=256)
    obj()
    obj = SkeletonBasedActionRecognition(training=0,batch_size=256)
    obj()
    print('\n   ---   Run 5 finished.   ---\n')
    print('\n   ---   Run 6.   ---\n')
    obj = SkeletonBasedActionRecognition(training=1,double_precision=1)
    obj()
    obj = SkeletonBasedActionRecognition(training=0,double_precision=1)
    obj()
    print('\n   ---   Run 6 finished.   ---\n')
