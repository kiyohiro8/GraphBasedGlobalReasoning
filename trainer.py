# -*- coding: utf-8 -*-

import os
import time
import datetime
import shutil
import json

import numpy as np
import pandas as pd
import torch
from torch import optim, nn
from torch.optim.lr_scheduler import CosineAnnealingLR

from losses import FocalLoss


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
        optimizer = optim.Adam(model.parameters(), lr=self.lr, betas=(self.beta_1, 0.999), weight_decay=5e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)
        #class_weights = train_dataloader.dataset.class_weights.to(self.device)
        #criterion = nn.CrossEntropyLoss(weight=class_weights)
        criterion = FocalLoss()
        start_time = time.time()
        train_loss_list = []
        train_acc_list = []
        val_iou_list = []
        val_iou_dict_dict = {}

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

            model.eval()
            confusion_dict = calculate_confusion(model, val_dataloader, self.device)
            iou_dict = calculate_IoU(confusion_dict)
            val_acc = iou_dict["mean"]
            val_iou_dict_dict.update({epoch: iou_dict})
            model.train()

            train_loss_list.append(train_loss)
            train_acc_list.append(train_acc)
            val_iou_list.append(val_acc)
            
            with open(f"{result_dir}/class_iou.json", "w") as f:
                json.dump(val_iou_dict_dict, f)

            if epoch % self.checkpoint_epoch == 0:
                path = f"{result_dir}/{epoch:0>4}.pth"
                torch.save(model.state_dict(), path)

            scheduler.step()

            elapsed = time.time() - start_time
            print(f"epoch{epoch} done. "
                  f"train_loss:{train_loss:.4f}, train_acc: {train_acc:.4f}, "
                  f"val_acc: {val_acc:.4f}"
                  f" ({elapsed:.4f} sec)")
        result_df = pd.DataFrame({"epoch":list(range(1, self.max_epoch+1)),
                                  "train_loss": train_loss_list,
                                  "train_acc": train_acc_list,
                                  "val_acc": val_iou_list})
        result_df.to_csv(f"{result_dir}/result.csv", index=False)
        path = f"{result_dir}/{self.max_epoch:0>4}.pth"
        torch.save(model.state_dict(), path)
        path_optim = f"{result_dir}/{self.max_epoch:0>4}_optim.pth"
        torch.save(optimizer.state_dict(), path_optim)
        with open(f"{result_dir}/class_iou.json", "w") as f:
            json.dump(val_iou_dict_dict, f)
        print(f"training ends ({elapsed} sec)")
        return model

def calculate_confusion(model, dataloader, device, num_class=34):
    result_dict = {}
    
    for i in range(num_class):
        result_dict.update({i: {"TP":0,
                                "FPFN": 0}})
    model.eval()
    with torch.no_grad():
        for i, (images, labels) in enumerate(dataloader):
            images = images.to(device)
            labels = labels.cpu().numpy()
            
            pred = model(images).cpu().detach().numpy()
            pred = np.argmax(pred, 1)
            for i in range(num_class):

                TP = np.logical_and(labels==i, pred==i).sum()
                FPFN = np.logical_xor(labels==i, pred==i).sum()
                result_dict[i]["TP"] += TP
                result_dict[i]["FPFN"] += FPFN

            del pred
    
    return result_dict

def calculate_IoU(confusion_dict):
    iou_dict = {}
    mean_IoU = 0
    for i in confusion_dict.keys():
        TP = confusion_dict[i]["TP"]
        FPFN = confusion_dict[i]["FPFN"]
        if TP+FPFN != 0:
            IoU = TP/(TP+FPFN)
        else:
            IoU = 0
        iou_dict.update({i: IoU})
        mean_IoU += IoU
        
    mean_IoU /= len(iou_dict.keys())
    iou_dict.update({"mean": mean_IoU})
    return iou_dict