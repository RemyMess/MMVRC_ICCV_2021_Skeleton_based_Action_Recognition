from src.preprocessing.augmented_sig_transformer import AugmentedSigTransformer
from src.algos.sig_classifier import SigClassifier
from src.preprocessing.pre_normaliser import preNormaliser
from src.algos.sigRNNlinconv import SigRNNLinConv
from src.preprocessing.linconv_data import LinConvData

class SkeletonBasedActionRecognition:
    '''
        A class for skeleton-based human action recognition on UAV.
        Two methods: 1) '__init__': data preprocessing or loading, building the classifier.
        ... A priori, several classifiers can be chosen and the device can be specified. 2) 'run' contains training
        ... and validation, as well as printing the confusion matrix.
    '''
    def __init__(self,algo='SigClassifier',batch_size=20,lr=0.0001,epochs=20,transform='example',load_data=False,
            device='cuda',pad=True,centre=1,rotate=1,switchBody =True, eliminateSpikes = True, scale = 2, smoothen = True,debug=False):

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
            load_data: whether to load the data or to create and save it. The corresponding path is ~/src/pre_processing/PSF/.
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

            debug: if True, only a small fraction of the frames and joints will be considered.

            · Commands:
            Defines a suitable preNormaliser object, if data is not to be loaded.
            Defines the signature data wrapper (an AugmentedSigTransformer object) and the sig_classifier (a SigClassifier
            ... object) for the 'SigClassifier' algo with the specified parameters, see above
        '''

        if not load_data:
            print('Init pre_normaliser')
            pre_normaliser = preNormaliser(pad=pad,centre=centre,rotate=rotate,switchBody=switchBody,eliminateSpikes=eliminateSpikes,scale=scale,parallel=False,smoothen=smoothen)
            print('Init pre_normaliser finished')
        else:
            pre_normaliser = None

        self.algo = algo

        if algo=='SigClassifier':
            sig_data_wrapper = AugmentedSigTransformer(pre_normaliser=pre_normaliser,transform=transform,
                    load_transform=load_data,debug=debug)
            self.sig_classifier = SigClassifier(
                    sig_data_wrapper=sig_data_wrapper,device=device,batch_size=batch_size,lr=lr,epochs=epochs)

        if self.algo=='logsigrnn_linconv' :
            sig_data_wrapper = LinConvData(pre_normaliser=pre_normaliser,transform=transform, load_transform=load_data,debug=debug)
            self.sig_classifier = SigRNNLinConv(sig_data_wrapper=sig_data_wrapper,batch_size=batch_size,lr=lr,epochs=epochs)

        else:
            raise NotImplementedError

    def run(self, render_plot: bool = True):
        '''
            Performs training and validation. Prints the confusion matrix with respect to the validation set if
            ... render_plot == True.
        '''
        if self.algo=='SigClassifier':
            self.sig_classifier.run(render_plot=render_plot)
        if self.algo=='logsigrnn_linconv':
            self.sig_classifier.run(render_plot=render_plot)
        else:
            raise NotImplementedError

if __name__ == "__main__":
    print('Init obj.')
    obj = SkeletonBasedActionRecognition(algo='logsigrnn_linconv',batch_size=256,lr=0.001,epochs=60,transform='landmarks',load_data=False,
            device='cuda',pad=True,centre=1,rotate=1,switchBody =True, eliminateSpikes = True, scale = 2, smoothen = False, debug=False)
    print('Run obj.')
    obj.run(render_plot=True)
