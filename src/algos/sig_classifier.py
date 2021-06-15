import torch

# TODO: debug


BATCH_SIZE = 20
LR = 0.0001
EPOCHS = 20

class LinearClassifierNet(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        torch.nn.Module.__init__(self)
        self.linear = torch.nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        x = self.linear(x)
        return x

class SigClassifier:
    def __init__(SigDataWrapper):
        self.SigDataWrapper = SigDataWrapper
        self.model = LinearClassifierNet(input_dim=trainingset.get_data_dimension(), output_dim=class_num)
        self._setup_model()

    def _setup_model(self):
        model.to(device=device, dtype=torch.double)


        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)


        training_loader = torch.utils.data.DataLoader(trainingset,
                                                    batch_size=BATCH_SIZE,
                                                    shuffle=True,
                                                    num_workers=2)
        test_loader = torch.utils.data.DataLoader(valset,
                                                batch_size=1,
                                                shuffle=False,
                                                num_workers=1)

    def train_network(epoch):
        self.model.train()
        cumulated_loss = 0.0
        #alpha = 0.00001
        for i, data in tqdm(enumerate(training_loader), desc="Epoch " + str(epoch), leave=False):
            inputs, labels = data[0].abs().to(device), data[1].to(device, dtype=torch.long)
            optimizer.zero_grad()
            outputs = model(inputs)
            #regularization_loss = 0
            #for param in model.parameters(): regularization_loss += torch.sum(torch.abs(param))
            loss = loss_fn(outputs, labels)# + alpha * regularization_loss
            loss.backward()
            optimizer.step()
            cumulated_loss += loss.item()
        return (cumulated_loss / len(training_loader))

    def test_network():
        cumulated_outputs = np.array([])
        cumulated_loss = 0.0
        model.eval()
        with torch.no_grad():
            for data in test_loader:
                inputs, labels = data[0].to(device), data[1].to(device, dtype=torch.long)
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
                cumulated_loss += loss.item()

                outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1)
                cumulated_outputs = np.concatenate((cumulated_outputs, outputs.cpu().numpy()), axis=None)
            test_loss = cumulated_loss / len(test_loader)
            return test_loss, cumulated_outputs

    def run(self, render_plot: bool = False):
        # Print initial accuracy before training
        __, outputs = test_network()
        acc = accuracy_score(valset.get_labels(), outputs)
        print("Initial accuracy:", acc)

        for epoch in trange(EPOCHS, desc="Training"):
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
    