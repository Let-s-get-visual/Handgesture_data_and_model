import torch
import torch.nn as nn
import torch.nn.functional as F


class Engine:
    def __init__(self, model, optimizer, device):
        self.model = model
        self.optimizer = optimizer
        self.device = device

    @staticmethod
    def loss_function(outputs, targets):
        loss = nn.NLLLoss()
        outputs = F.log_softmax(outputs, dim=1)
        return loss(outputs, targets)

    def train(self, data_loader):
        self.model.train()
        final_loss = 0
        final_acc = 0

        for data in data_loader:
            self.optimizer.zero_grad()
            inputs = data[0].to(self.device)
            targets = data[1].to(self.device).to(torch.long)

            outputs = self.model(inputs)
            pred = torch.argmax(outputs, dim=1, keepdim=False)
            acc = torch.sum(pred == targets.view(-1).data) / pred.shape[0]

            outputs = outputs.to(self.device)
            loss = self.loss_function(outputs, targets)
            loss.backward()

            self.optimizer.step()
            final_loss += loss.item()
            final_acc += acc.item()
        epoch_loss = final_loss / len(data_loader)
        epoch_accuracy = final_acc / len(data_loader)
        return epoch_loss, epoch_accuracy

    def eval(self, data_loader):
        self.model.eval()
        final_loss = 0
        final_acc = 0

        with torch.no_grad():

            for data in data_loader:
                inputs = data[0].to(self.device)
                targets = data[1].to(self.device).to(torch.long)

                outputs = self.model(inputs)
                outputs = outputs.to(self.device)
                pred = torch.argmax(outputs, dim=1, keepdim=False)
                acc = torch.sum(pred == targets.view(-1).data) / pred.shape[0]

                loss = self.loss_function(outputs, targets)
                final_loss += loss.item()
                final_acc += acc.item()
        epoch_loss = final_loss / len(data_loader)
        epoch_accuracy = final_acc / len(data_loader)
        return epoch_loss, epoch_accuracy
