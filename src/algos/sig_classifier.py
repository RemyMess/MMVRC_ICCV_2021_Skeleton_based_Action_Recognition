import torch

class LinearNet(torch.nn.Module):
    '''
        Two linear layers with Dropout.
    '''
    def __init__(self, input_dim, output_dim=155,intermediate_dim=512):
        torch.nn.Module.__init__(self)
        # for hyperpar testing only: intermediate_dim = 256
        self.linear1 = torch.nn.Linear(input_dim,intermediate_dim)
        self.Dropout = torch.nn.Dropout(0.5)
        #self.BN = torch.nn.BatchNorm1d(intermediate_dim)
        self.linear2 = torch.nn.Linear(intermediate_dim,output_dim)
        
    def forward(self, x):
        x = self.linear1(x)
        x = self.Dropout(x)
        #x = self.BN(x)
        x = self.linear2(x)
        return x

    #----------------------------------------

import signatory

class TJSigNet(torch.nn.Module):
    def __init__(self,sig_depth,debug=False):
        super(TJSigNet,self).__init__()
        if debug:
            sig_depth = 2
        self.signature = []
        for v in range(17):
            self.signature.append(signatory.Signature(depth=sig_depth))
        # channels = 3 because output size of the neural net is 3, no time-augmentation, no original duplicate
        sig_channels = signatory.signature_channels(channels=3,
                                                    depth=sig_depth)
        intermediate_width = 512 if not debug else 256
        self.linear1 = torch.nn.Linear(34*sig_channels, intermediate_width)
        self.dropout = torch.nn.Dropout(p=0.5)
        self.linear2 = torch.nn.Linear(intermediate_width,155)

    def forward(self,inp):
        # inp is tensor of shape (N,C,T,V,M), M=2, V=17
        m = 0
        xs = []
        for v in range(17):
            for m in range(2):
                x = inp[:,:,:,v,m].transpose(1,2)
                x = self.signature[v](x)
                xs.append(x)
        x = torch.cat(xs,1)
        x = self.linear1(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x                

    # -----------------------------

import itertools
    
class SigNetClassic(torch.nn.Module):
    def __init__(self,*args,d_tuples=(2,3),debug=False):
        super(SigNet,self).__init__()
        self.d_tuples = d_tuples
        if debug:
            pass
        pass

    def forward(self,x):
        # x=input is a tensor of shape (N,C,T,V,M), M=2, V=17, N: batch size
        N,C,T,V,M = x.shape
        x_copy = x
        
        # tuples: input: (N,C,T,V,M) ---> output: (N,d_tuple,-1)
        # format of x_unmixed_tuples: [M,d_tuples], containes tensors of shape (N,d_tuple*,-1)
        x_unmixed_tuples = []
        x_mixed_tuples = []
        for d in self.d_tuples:
            unmixed_tuples1, unmixed_tuples2, mixed_tuples = self._get_tuples(range(34),d,17)
            x_personal_tuples = []
            for m in range(M):
                pass
        x_unmixed_tuples = []
        for m in range(M):
            x_personal_tuples = []
            for d in self.d_tuples:
                x_personal_tuples.append(...)
            x_unmixed_tuples.append(x_personal_tuples)

        x_mixed_tuples = []
        for d in self.d_tuples:
            x_mixed_tuples.append(...)

    def _get_tuples(source,degree: int,separation):
        # only implemented for m = 2
        
        # source: an iterable of length N, degree: the degree of tuples to be returned,
        # separation: an index of source
        # returns: list of all unmixed tuples until separation-1,
        # ... list of all unmixed tuples from separation on,
        # ... list of all mixed tuples
        
        all_tuples = itertools.combinations(source)
        unmixed_tuples1 = []
        unmixed_tuples2 = []
        mixed_tuples = []
        for tuple in all_tuples:
            below, above = False, False
            for element in tuple:
                if element < separation:
                    below = True
                else:
                    above = True
                if below and above:
                    break
            if below and above:
                mixed_tuples.append(tuple)
            elif below:
                unmixed_tuples1.append(tuple)
            else:
                unmixed_tuples2.append(tuple)
        return unmixed_tuples1, unmixed_tuples2, mixed_tuples
                    
    def _convolve_joints_person(self,x,m):
        # x: input tensor of shape (N,C,T,V), m: person ID
        # returns: tensor of shape (N,C,T,V'), V' << V
        x_C = []
        for c in range(C):
            x_C.append(self.augment1_person(x[:,c,:,:]))
        x_C = torch.cat(x_C).transpose(0,1)
        return x_C

    def _convolve_interacting_joints(self,x):
        # x: input tensor of shape (N,C,T,V*M)
        # returns: tensor of shape (N,C,T,V''), V'' << 2*V
        x_C = []
        for c in range(C):
            x_C.append(self.augment1_interaction(x[:,c,:,:]))
        x_C = torch.cat(x_C).transpose(0,1)
        return x_C

    # ----------------------------
'''    
class SigNetNotSoClassic(torch.nn.Module):
    def __init__(self,*args,d_tuples=(2,3),debug=False):
        super(SigNet,self).__init__()
        self.d_tuple = d_tuple
        if debug:
            pass
        pass

    def forward(self,x):
        # x=input is a tensor of shape (N,C,T,V,M), M=2, V=17, N: batch size
        N,C,T,V,M = x.shape
        x_copy = x
        
        # convolve joints (uniformly over C): a) for each person, b) for the interaction between both persons
        # result: x_Spat_Conv, a tensor of shape (N,C,T,V*)
        x_M = []
        for m in range(M):
            x_M.append(self._convolve_joints_person(x[:,:,:,:,m],m))
        x_inter = torch.reshape(x,(N,C,T,V*M))
        x_inter = self._convolve_interacting_joints(x_inter)
        x_M.append(x_inter)
        x_SpatConv = torch.cat(x_M,axis=3)

        # create self.d_tuple-tuples of x_SpatConv and compute their signatures
        x.reshape((N,C,T,V*M))
        

    def _convolve_joints_person(self,x,m):
        # x: input tensor of shape (N,C,T,V), m: person ID
        # returns: tensor of shape (N,C,T,V'), V' << V
        x_C = []
        for c in range(C):
            x_C.append(self.augment1_person(x[:,c,:,:]))
        x_C = torch.cat(x_C).transpose(0,1)
        return x_C

    def _convolve_interacting_joints(self,x):
        # x: input tensor of shape (N,C,T,V*M)
        # returns: tensor of shape (N,C,T,V''), V'' << 2*V
        x_C = []
        for c in range(C):
            x_C.append(self.augment1_interaction(x[:,c,:,:]))
        x_C = torch.cat(x_C).transpose(0,1)
        return x_C
'''
    # ----------------------------
    # ----------------------------

from torch.utils.data import Dataset
import os

class UAVDataset(Dataset):
    ''' 
transform: specifies data augmentation transforms, e.g. Gaussian noise, flipping
'''
    def __init__(self,data,pre_process_flag,transform=None,data_in_mem=False):
        super(UAVDataset,self).__init__()
        if data_in_mem:
            self.X = data[0]
            self.Y = data[1]
        else:
            self.path = data[0]
            self.indices = data[1]
        self._pre_process_flag = pre_process_flag
        self.data_in_mem = data_in_mem
        
    def __len__(self):
        if self.data_in_mem:
            return len(self.Y)
        else:
            return len(self.indices)

    def __getitem__(self,n):
        if self.data_in_mem:
            x,y = self.X[n], self.Y[n]
        else:
            n = self.indices[n]
            datum = np.load(os.path.join(self.path,'landmarks',self._pre_process_flag,'datum_{}.npy'.format(n)),allow_pickle=True)
            x,y = datum[0],datum[1]
        x = torch.tensor(x,dtype=torch.float)
        y = torch.tensor(y,dtype=torch.float)
        return x,y

    def get_labels(self):
        if self.data_in_mem:
            return self.Y
        else:
            Y_full = np.load(os.path.join(self.path,'landmarks',self._pre_process_flag,'labels.npy'))
            return Y_full[self.indices]

    # ---------------------------------------

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from tqdm import tqdm, trange
import matplotlib.pyplot as plt



BATCH_SIZE = 64
LR = 0.005
EPOCHS = 20
class SigClassifier:
    '''
        '__init__' requires an AugmentedSigTransformer object as input, it defines the model as a LinearClassifierNet object. 
        ... The model is set up. There, the AugmentedSigTransformer object yields the "wrapped" training and validation sets  
        ... and the result is fed into the appropriate torch data loaders.
        'run' performs the training and, if desired, printing of the confusion matrix. The mechanics of training and valing
        ... per iteration (epoch) are outsourced into separate methods ('train_network', 'val_network').
    '''
    def __init__(self,algo_spec,flag='test',double_precision=True,batch_size=BATCH_SIZE,lr=LR,epochs=EPOCHS,debug=False,**model_kwargs):
        # augment_std: data augmentation in training loop for path signature features,
        # multiplicative Gaussion noise with std augment_std; inactive if None
        if algo_spec=='LinearNet':
            self._augment_std = model_kwargs['augment_std']
            data_dimension = model_kwargs['data_dimension']
            self.model = LinearNet(data_dimension)
        elif algo_spec=='DeepSigTrafo':
            sig_depth = model_kwargs['sig_depth']
            self.model = TJSigNet(sig_depth)
        self.dtype = torch.double if double_precision else torch.float
        class_num = 155
        self.double = double_precision
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs if not debug else 2
        self.flag = flag
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        torch.manual_seed(0)
        np.random.seed(0)
        torch.cuda.manual_seed_all(0)
        self._debug = debug

    def build(self,training_set=None,val_set=None):
        self.model.to(device=self.device,dtype=self.dtype)
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        if training_set is not None:
            self._training_loader = torch.utils.data.DataLoader(training_set,
                                                                batch_size=self.batch_size,
                                                                shuffle=True)
        self._val_loader = torch.utils.data.DataLoader(val_set,
                                                       batch_size=2000,
                                                       shuffle=False)
        self._val_labels = val_set.get_labels()

    def _train_network(self,epoch):
        self.model.train()
        cumulated_loss = 0.0
        #alpha = 0.00001
        for i, data in tqdm(enumerate(self._training_loader), desc="Epoch " + str(epoch+1), leave=False):
            inputs, labels = data[0].to(self.device,dtype=self.dtype), data[1].to(self.device,dtype=torch.long)
            self.optimizer.zero_grad()
            if self._augment_std is not None:
                ones = torch.ones(inputs.shape,dtype=self.dtype,device=self.device)
                inputs *= torch.randn(size=inputs.shape,dtype=self.dtype,device=self.device)*self._augment_std + 1.
            outputs = self.model(inputs)
            #regularization_loss = 0
            #for param in model.parameters(): regularization_loss += torch.sum(torch.abs(param))
            loss = self.loss_fn(outputs, labels)# + alpha * regularization_loss
            loss.backward()
            self.optimizer.step()
            cumulated_loss += loss.item()
        return (cumulated_loss / len(self._training_loader))

    def _val_network(self):
        cumulated_outputs = np.array([])
        cumulated_loss = 0.0
        self.model.eval()
        with torch.no_grad():
            for data in tqdm(self._val_loader,desc='Validation',leave=False):
                inputs, labels = data[0].to(self.device,dtype=self.dtype), data[1].to(self.device,dtype=torch.long)
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, labels)
                cumulated_loss += loss.item()

                outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1)
                cumulated_outputs = np.concatenate((cumulated_outputs, outputs.cpu().numpy()), axis=None)
            val_loss = cumulated_loss / len(self._val_loader)
            return val_loss, cumulated_outputs

    def fit(self,plot_path=None):
        # Print initial accuracy before training
        __,outputs = self._val_network()
        acc = accuracy_score(self._val_labels, outputs)
        print("Initial accuracy:", acc)

        for epoch in tqdm(range(self.epochs), desc="Training"):
            train_loss = self._train_network(epoch)
            val_loss, outputs = self._val_network()
            
            acc = accuracy_score(self._val_labels, outputs)
            print("\nEpoch:", epoch+1, "\ttrain_loss:", train_loss, "\tval_loss:", val_loss, "\tAccuracy:", acc)

        if plot_path is not None:
            fig = plt.figure(figsize=(15, 15))
            cm = confusion_matrix(self._val_labels, outputs)
            ax = fig.add_subplot(111)
            cm_display = ConfusionMatrixDisplay(cm).plot(ax=ax,xticks_rotation="vertical")
            plt.savefig(os.path.join(plot_path,'confusion_matrix_{}.png'.format(self.flag)),dpi=300,bbox_inches='tight')
            plt.show()

    def predict(self):
        _,outputs = self._val_network()
        return outputs
    
    def save(self,path):
        torch.save(self.model.state_dict(),path+'.pt')

    def load(self,path):
        self.model.load_state_dict(torch.load(path+'.pt'))
