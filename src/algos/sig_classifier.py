import torch


BATCH_SIZE = 20
LR = 0.0001
EPOCHS = 20

class LinearClassifierNet(torch.nn.Module):
    '''
        A linear layer.
    '''
    def __init__(self, input_dim, output_dim):
        torch.nn.Module.__init__(self)
        self.linear = torch.nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        x = self.linear(x)
        return x

class SigClassifier:
    '''
        '__init__' requires an AugmentedSigTransformer object as input, it defines the model as a LinearClassifierNet object. 
        ... The model is set up. There, the AugmentedSigTransformer object yields the "wrapped" training and validation sets  
        ... and the result is fed into the appropriate torch data loaders.
        'run' performs the training and, if desired, printing of the confusion matrix. The mechanics of training and testing
        ... per iteration (epoch) are outsourced into local methods ('train_network', 'test_network').
    '''
    def __init__(self,sig_data_wrapper,device='cuda',batch_size=BATCH_SIZE,lr=LR,epochs=EPOCHS):
        self.sig_data_wrapper = sig_data_wrapper
        self.model = LinearClassifierNet(input_dim=trainingset.get_data_dimension(), output_dim=class_num)
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self._setup_model()
        self._device = torch.device(device)

    def _setup_model(self):
        self.model.to(device=self._device, dtype=torch.double)

        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

	trainingset,valset = self.sig_data_wrapper.sig_data
        self.training_loader = torch.utils.data.DataLoader(trainingset,
                                                    batch_size=self.batch_size,
                                                    shuffle=True,
                                                    num_workers=2)
        self.test_loader = torch.utils.data.DataLoader(valset,
                                                batch_size=1,
                                                shuffle=False,
                                                num_workers=1)

    def train_network(epoch):
        self.model.train()
        cumulated_loss = 0.0
        #alpha = 0.00001
        for i, data in tqdm(enumerate(self.training_loader), desc="Epoch " + str(epoch), leave=False):
            inputs, labels = data[0].abs().to(self._device), data[1].to(self._device, dtype=torch.long)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            #regularization_loss = 0
            #for param in model.parameters(): regularization_loss += torch.sum(torch.abs(param))
            loss = self.loss_fn(outputs, labels)# + alpha * regularization_loss
            loss.backward()
            self.optimizer.step()
            cumulated_loss += loss.item()
        return (cumulated_loss / len(self.training_loader))

    def test_network():
        cumulated_outputs = np.array([])
        cumulated_loss = 0.0
        self.model.eval()
        with torch.no_grad():
            for data in self.test_loader:
                inputs, labels = data[0].to(self._device), data[1].to(self._device, dtype=torch.long)
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, labels)
                cumulated_loss += loss.item()

                outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1)
                cumulated_outputs = np.concatenate((cumulated_outputs, outputs.cpu().numpy()), axis=None)
            test_loss = cumulated_loss / len(self.test_loader)
            return test_loss, cumulated_outputs

    def run(self, render_plot: bool = False):
        # Print initial accuracy before training
        __, outputs = test_network()
        acc = accuracy_score(valset.get_labels(), outputs)
        print("Initial accuracy:", acc)

        for epoch in trange(self.epochs, desc="Training"):
            train_loss = train_network(epoch)
            test_loss, outputs = test_network()
            
            acc = accuracy_score(valset.get_labels(), outputs)
            print("epoch:", epoch+1, "\ttrain_loss:", train_loss, "\ttest_loss:", test_loss, "\tAccuracy:", acc)

        if render_plot:
            fig = plt.figure(figsize=(15, 15))
            cm = confusion_matrix(valset.get_labels(), outputs)
            ax = fig.add_subplot(111)
            cm_display = ConfusionMatrixDisplay(cm,
                                                display_labels=np.arange(155)).plot(ax=ax,
                                                                                xticks_rotation="vertical")
    
