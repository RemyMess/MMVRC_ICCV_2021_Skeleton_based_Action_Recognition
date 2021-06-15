from psfdataset import transforms

# TODO: debug (follow Weixin's original code)

class AugmentedSigTransformer:
    def __init__(self, preNormalisedDataWrapper):
        self.preNormalisedDataWrapper = preNormalisedDataWrapper
        self.train_sig_data, self.train_sig_label = self.transform() # TO DEBUG

    def get_iter(data,label): 
        """
        Returns iterators over the requested
        subset of the data.
        """
        return iter(list(zip(data, label)))

    def transform(self):
        tr = transforms.Compose([
                transforms.spatial.Crop(),
                transforms.spatial.Normalize(),
                transforms.spatial.Tuples(2),
                transforms.SpatioTemporalPath(),
                transforms.temporal.MultiDelayedTransformation(2),
                transforms.temporal.DyadicPathSignatures(dyadic_levels=1,
                                                        signature_level=3)
        ])

        isFastTestSubset = True

        data_uav = np.transpose(self.preNormalisedDataWrapper.train_prenorm_data,(0,2,4,3,1))
        #if isFastTestSubset: data_uav = data_uav[:,:305]
        data_uav = data_uav.reshape(*data_uav.shape[:2], -1, *data_uav.shape[4:])
        print(data_uav.shape)

        fold_idx = 0
        train_index = k_fold_idx_dict[str(fold_idx)]['train'] if not isFastTestSubset else k_fold_idx_dict[str(fold_idx)]['train'][::100]
        val_index = k_fold_idx_dict[str(fold_idx)]['val'] if not isFastTestSubset else k_fold_idx_dict[str(fold_idx)]['val'][::100]

        train_data = data_uav[train_index]
        print(train_data.shape)
        train_label = label_uav[train_index]
        train_length = length_uav[train_index]

        val_data = data_uav[val_index]
        print(val_data.shape)
        val_label = label_uav[val_index]
        val_length = length_uav[val_index]

        trainingset = PSFDataset(transform=self.tr)
        valset = PSFDataset(transform=self.tr)

        trainingset.fill_from_iterator(self.get_iter(train_data, train_label))
        valset.fill_from_iterator(self.get_iter(val_data, val_label))

        print("Number of trainingset elements:", len(trainingset))
        print("Number of testset elements", len(valset))
        print("Dimension of feature vector:", trainingset.get_data_dimension())

        #trainingset.save("uav_train")
        #testset.save("uav_test")
        
        return _good_value # TODO: to debug