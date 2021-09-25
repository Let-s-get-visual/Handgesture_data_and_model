import time
from time import ctime

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from engine import Engine
from cnn_model import CNN
import data_handler_CNN as dhC


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 100


def run_training(epochs, save_model=False, save_fig=False, early_stopping=True):

    data_loaders = {x: DataLoader(dhC.image_datasets[x],
                                  batch_size=1024,
                                  shuffle=(x == 'train')) for x in ['train', 'validation']}

    model = CNN()

    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    engine = Engine(model=model,
                    optimizer=optimizer,
                    device=DEVICE)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=engine.optimizer,
                                                        milestones=[k for k in range(20, epochs, 20)],
                                                        gamma=0.1)
    best_loss = np.inf
    best_accuracy = 0

    all_train_losses = []
    all_val_losses = []
    all_train_accuracy = []
    all_val_accuracy = []

    early_stopping_iter = 20
    early_stopping_counter = 0

    for epoch in range(epochs):
        train_loss, train_accuracy = engine.train(data_loaders["train"])
        valid_loss, valid_accuracy = engine.eval(data_loaders["validation"])

        lr_scheduler.step()

        all_train_losses.append(train_loss)
        all_val_losses.append(valid_loss)
        all_train_accuracy.append(train_accuracy)
        all_val_accuracy.append(valid_accuracy)

        print(f"Epoch: {epoch}")
        print("-"*20)
        print(f"Train loss: {train_loss}, Valid loss: {valid_loss}")
        print(f"Train accuracy: {train_accuracy}, Valid accuracy: {valid_accuracy}")
        print()

        if valid_accuracy > best_accuracy:
            best_accuracy = valid_accuracy
            if save_model:
                torch.save(model.state_dict(), "modelDict_best_accuracy.pth")

        if valid_loss < best_loss:
            best_loss = valid_loss
            if save_model:
                torch.save(model.state_dict(), "modelDict_best_loss.pth")

        else:
            early_stopping_counter += 1

        if early_stopping_counter > early_stopping_iter and early_stopping:
            print(f"STOPPED EARLY AT {epoch}!!!")
            break

    if save_fig:
        plt.figure(figsize=(10, 8))
        plt.plot(all_train_losses)
        plt.plot(all_val_losses)
        plt.savefig("Losses.png")

        plt.figure(figsize=(10, 8))
        plt.plot(all_train_accuracy)
        plt.plot(all_val_accuracy)
        plt.savefig("Accuracy.png")

    return best_loss, best_accuracy


if __name__ == "__main__":

    start = time.time()
    print(f"Started at: {ctime(start)}")
    print("-" * 10 + "*" * 3 + "-" * 10)
    print()

    loss, accuracy = run_training(EPOCHS,
                                  save_model=True,
                                  save_fig=True,
                                  early_stopping=False)

    print(f"Loss: {loss}")
    print(f"Accuracy: {accuracy}")
    print("-" * 10 + "*" * 3 + "-" * 10)
    print()

    print(f"Ended at: {ctime(time.time())}")

    duration = time.time() - start

    print(f"Trained in: {duration / 60}")


