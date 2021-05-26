from torch import nn
import torch.optim as optim
import torch
import dataset_type_2
from torch.utils.data import DataLoader

from sklearn import metrics

class Trainer:
    def __init__(self, model, learning_rate, optimizer=None, criterion=None):

        # Select device
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")


        self.model = model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()

        
    def fit(self, vir_train_set, bac_train_set, batch_size, n_epochs):
        """ Train model """
        # create DataLoader
        train_forward_dataset = dataset_type_2.OneHotEncDataset(bac_train_set, vir_train_set, is_reverse = False)
        train_reverse_dataset = dataset_type_2.OneHotEncDataset(bac_train_set, vir_train_set, is_reverse = True)

        train_loader_args = dict(shuffle=True, batch_size=batch_size)

        train_forward_loader = DataLoader(train_forward_dataset, **train_loader_args)
        train_reverse_loader = DataLoader(train_reverse_dataset, **train_loader_args)

        self.model.train()
        for epoch in range(n_epochs):
            print(F"epoch: {epoch}")
            epoch_loss = 0
            for (x_batch_forward, y_batch),(x_batch_reverse,_) in zip(train_forward_loader, train_reverse_loader):
                self.optimizer.zero_grad()
                output = self.model(x_batch_forward.to(self.device), x_batch_reverse.to(self.device))
                loss = self.criterion(output,y_batch.to(self.device))
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

        print("Fit completed. Model parameters:")
        for param in self.model.parameters():
            print(param)

    def predict_proba(self, test_forward_loader, test_reverse_loader):
        self.model.eval()
        with torch.no_grad():
            all_outputs = torch.tensor([], dtype=torch.float32).to(self.device)

            for (x_batch_forward, y_batch),(x_batch_reverse,_) in zip(test_forward_loader, test_reverse_loader):
                output = self.model(x_batch_forward.to(self.device),
                    x_batch_reverse.to(self.device))
                all_outputs = torch.cat((all_outputs,output),0)

        return all_outputs

    def get_model_scores(self, vir_test_set, bac_test_set, batch_size):
        test_forward_dataset = dataset_type_2.OneHotEncDataset(bac_test_set, vir_test_set, is_reverse = False)
        test_reverse_dataset = dataset_type_2.OneHotEncDataset(bac_test_set, vir_test_set, is_reverse = True)

        test_loader_args = dict(shuffle=False, batch_size=batch_size)
        
        test_forward_loader = DataLoader(test_forward_dataset, **test_loader_args)
        test_reverse_loader = DataLoader(test_reverse_dataset, **test_loader_args)
        pred = self.predict_proba(test_forward_loader, test_reverse_loader) # return from device!!!

        pred_classes = torch.max(pred.data,1)[1]

        pred=pred.numpy()[:,1]
        pred_classes = pred_classes.numpy()
        labels = test_forward_dataset.Y.numpy()

        roc = metrics.roc_auc_score(labels, pred)

    


        precision, recall, f1, _ = metrics.precision_recall_fscore_support(labels, pred_classes, average='binary')
        f1 = metrics.f1_score(labels, pred_classes)
        accuracy = metrics.accuracy_score(labels, pred_classes)

        return [roc, precision, recall, f1, accuracy]

