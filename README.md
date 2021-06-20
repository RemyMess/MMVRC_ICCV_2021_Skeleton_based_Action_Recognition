# MMVRC_ICCV_2021_Skeleton_based_Action_Recognition

Solution to task 2 in MMVRC competition. Statement of that task is the following:

"Skeleton-based action recognition which aims to recognize human behaviors given skeletal data has been attracting more and more attentions, as skeletons are concise and powerful representations for human behaviors. However, existing works lack viewpoints from the UAV perspectives, which are important in many real-world application scenarios, such as catastrophe rescue, city surveillance and wild patrol. These are some situations where humans are out of reach but monitoring and understanding humans' behaviors are highly demanded."


Source: https://sutdcv.github.io/multi-modal-video-reasoning/#/datasets
Dataset paper: https://arxiv.org/abs/2104.00946


## Installation

1. Create your environment and build dependences
```bash
cd build;
./create_env.sh;
source .env_mmvrc/bin/activate;
./build.sh;
```
2. Run
```python
python main.py
```

## Structure of main.py

A class for skeleton-based human action recognition on UAV.
        Two methods: 1) '__init__': data preprocessing or loading, building the classifier. 
        ... A priori, several classifiers can be chosen and the device can be specified. 2) 'run' contains training
        ... and validation, as well as printing the confusion matrix.

The init method:    
        – Inputs:
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
            pad: whether to perform the padding procedure (addressing shifted started, missing frames, asynchronous ending
            ... in case of two-player scenes) or not, see ~/src/pre_processing/pre_normaliser.py
            centre: whether to centre the skeleta around the first person's torso or not
            rotate: whether to rotate the skeleta such that the vertebrae and clavicles are aligned with the z- and x-axes
            ... or not
            debug: if True, only a small fraction of the frames will be considered.
            
        – Commands:
            Defines a suitable preNormaliser object, if data is not to be loaded.
            Defines the signature data wrapper (an AugmentedSigTransformer object) and the sig_classifier (a SigClassifier
            ... object) for the 'SigClassifier' algo with the specified parameters, see above
            
The run method:
        Performs training and validation. Prints the confusion matrix with respect to the validation set if 
            ... render_plot == True.

## License
[MIT](https://choosealicense.com/licenses/mit/)


## Authors and acknowledgment
Elena Gal, Philip Jettkant, Konstantin St, Hannes Kern, Sebastian Ertel, Simon Breneis, Emanuel Rapsch, Weile Weng, Weixin Yang, Terry Lyons, Remy Messadene

## Project status
On-going
