# -*- coding: utf-8 -*-

import os
import time
import datetime
import shutil

import numpy as np
import pandas as pd
import torch
from torch import optim, nn


class Trainer(object):
    def __init__(self, params):

        training_params = params["training"]
        
        self.lr = training_params["lr"]
        self.max_epoch = training_params["epoch"]
        self.beta_1 = training_params["beta_1"]
        self.checkpoint_root = training_params["checkpoint_root"]
        self.checkpoint_epoch = training_params["checkpoint"]
        if training_params["use_cuda"]:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
    
    def train(self, model, result_dir, train_dataloader, val_dataloader=None):
        print("create data loader")
        model.to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=self.lr, betas=(self.beta_1, 0.999))
        
        criterion = nn.CrossEntropyLoss()

        start_time = time.time()
        train_loss_list = []
        train_acc_list = []
        val_loss_list = []
        val_acc_list = []
        print("training starts")
        for epoch in range(1, self.max_epoch+1):
            train_loss = 0
            train_acc = 0
            val_loss = 0
            val_acc = 0
            for i, (images, labels) in enumerate(train_dataloader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                pred = model(images)
                loss = criterion(pred, labels)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                ps = torch.exp(pred)
                equality = (labels.data == ps.max(dim=1)[1])
                train_acc += equality.type(torch.FloatTensor).mean().item()
            train_loss /= len(train_dataloader)
            train_acc /= len(train_dataloader)

            if val_dataloader is not None and epoch % self.checkpoint_epoch == 0:
                model.eval()
                with torch.no_grad():
                    for i, (images, labels) in enumerate(val_dataloader):
                        images = images.to(self.device)
                        labels = labels.to(self.device)
                        pred = model(images)
                        loss = criterion(pred, labels)
                        val_loss += loss.item()
                        ps = torch.exp(pred)
                        equality = (labels.data == ps.max(dim=1)[1])
                        val_acc += equality.type(torch.FloatTensor).mean().item()
                    val_loss /= len(val_dataloader)
                    val_acc /= len(val_dataloader)
                    del pred
                model.train()
            else:
                val_loss = np.nan
                val_acc = np.nan

            train_loss_list.append(train_loss)
            train_acc_list.append(train_acc)
            val_loss_list.append(val_loss)
            val_acc_list.append(val_acc)

            if epoch % self.checkpoint_epoch == 0:
                path = f"{result_dir}/{epoch:0>4}.pth"
                torch.save(model.state_dict(), path)

            elapsed = time.time() - start_time
            print(f"epoch{epoch} done. "
                  f"train_loss:{train_loss:.4f}, train_acc: {train_acc:.4f}, "
                  f"val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}"
                  f" ({elapsed:.4f} sec)")
        result_df = pd.DataFrame({"epoch":list(range(1, self.max_epoch+1)),
                                  "train_loss": train_loss_list,
                                  "val_loss": val_loss_list,
                                  "train_acc": train_acc_list,
                                  "val_acc": val_acc_list})
        result_df.to_csv(f"{result_dir}/result.csv", index=False)
        path = f"{result_dir}/{self.max_epoch:0>4}.pth"
        torch.save(model.state_dict(), path)
        path_optim = f"{result_dir}/{self.max_epoch:0>4}_optim.pth"
        torch.save(optimizer.state_dict(), path_optim)
        print(f"training ends ({elapsed} sec)")
        return model