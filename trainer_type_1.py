from torch import nn
import torch.optim as optim
import torch
import dataset_type_1
from torch.utils.data import DataLoader
from sklearn import metrics
import logging
import sys

class Trainer:
    def __init__(self, model_forward, model_reverse, fr_result, learning_rate, optimizer=None, criterion=None):

        # Select device
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")


        self.model_forward = model_forward.to(self.device)
        self.model_reverse = model_reverse.to(self.device)
        self.fr_result = fr_result
        self.optimizer_forward = optim.Adam(self.model_forward.parameters(), lr=learning_rate)
        self.optimizer_reverse = optim.Adam(self.model_reverse.parameters(), lr=learning_rate)
        self.criterion_forward = nn.CrossEntropyLoss()
        self.criterion_reverse= nn.CrossEntropyLoss()

    def fit_strand_specific(self, model, train_dataloader, optimizer, criterion, n_epochs):
        model.train()
        for epoch in range(n_epochs):
            logging.debug(F"epoch: {epoch}")
            epoch_loss = 0
            for x_batch, y_batch in train_dataloader:
                optimizer.zero_grad()
                output = model(x_batch.to(self.device))
                loss = criterion(output,y_batch.to(self.device))
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

        logging.debug("Fit completed. Model parameters:")
        for param in model.parameters():
            logging.debug(param)


    def fit(self, vir_train_set, bac_train_set, batch_size, n_epochs):
        """ Train model """
        # create DataLoader
        train_forward_dataset = dataset_type_1.OneHotEncDataset(bac_train_set, vir_train_set, is_reverse = False)
        train_reverse_dataset = dataset_type_1.OneHotEncDataset(bac_train_set, vir_train_set, is_reverse = True)

        train_loader_args = dict(shuffle=True, batch_size=batch_size)

        train_forward_loader = DataLoader(train_forward_dataset, **train_loader_args)
        train_reverse_loader = DataLoader(train_reverse_dataset, **train_loader_args)

        self.fit_strand_specific(self.model_forward,
            train_forward_loader,
            self.optimizer_forward,
            self.criterion_forward,
            n_epochs)

        self.fit_strand_specific(self.model_reverse,
            train_reverse_loader,
            self.optimizer_reverse,
            self.criterion_reverse,
            n_epochs)

    def get_model_scores(self, vir_test_set, bac_test_set, batch_size):
        test_forward_dataset = dataset_type_1.OneHotEncDataset(bac_test_set, vir_test_set, is_reverse = False)
        test_reverse_dataset = dataset_type_1.OneHotEncDataset(bac_test_set, vir_test_set, is_reverse = True)

        test_loader_args = dict(shuffle=False, batch_size=batch_size)
        
        test_forward_loader = DataLoader(test_forward_dataset, **test_loader_args)
        test_reverse_loader = DataLoader(test_reverse_dataset, **test_loader_args)
        pred = self.predict_proba(test_forward_loader, test_reverse_loader)

        pred_classes = torch.max(pred.data,1)[1]

        pred=pred.numpy()[:,1]
        pred_classes = pred_classes.numpy()
        labels = test_forward_dataset.Y.numpy()

        roc = metrics.roc_auc_score(labels, pred)

        precision, recall, f1, _ = metrics.precision_recall_fscore_support(labels, pred_classes, average='binary')
        f1 = metrics.f1_score(labels, pred_classes)
        accuracy = metrics.accuracy_score(labels, pred_classes)

        return [roc, precision, recall, f1, accuracy]

    

    def predict_proba(self, test_forward_loader, test_reverse_loader):
        self.model_forward.eval()
        self.model_reverse.eval()

        with torch.no_grad():
            all_outputs_forward = torch.tensor([], dtype=torch.float32)
            all_outputs_reverse = torch.tensor([], dtype=torch.float32)

            for x_batch, y_batch in test_forward_loader:
                output = self.model_forward(x_batch)
                all_outputs_forward = torch.cat((all_outputs_forward,output),0)

            for x_batch, y_batch in test_reverse_loader:
                output = self.model_reverse(x_batch)
                all_outputs_reverse = torch.cat((all_outputs_reverse,output),0)

            if self.fr_result == "average":
                result = (all_outputs_reverse + all_outputs_forward)/2
            elif self.fr_result == "max":
                result = max(all_outputs_reverse, all_outputs_forward)
            else:
                logging.error("Unknown fr_result")
                sys.exit()

        return result

    #def predict(self, test_dataloader):
        #output_proba = self.predict_proba(test_dataloader)
        #return torch.max(output_proba.data,1)[1]

    #def predict_proba_tensor(self,T):
        #self.model.eval()
        #with torch.no_grad():
        #    output=self.model(T)
        #return output