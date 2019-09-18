# -*- coding: utf-8 -*-

import sys
import yaml
import datetime
import os

from models import FCNwithGloRe
from trainer import Trainer
from utils import get_dataloader

def train(params):
    model = FCNwithGloRe(params)
    trainer = Trainer(params)
    image_size = params["common"]["image_size"][0]
    train_data_path = params["common"]["train_data_path"]
    val_data_path = params["common"]["val_data_path"]
    train_batch_size = params["common"]["train_batch_size"]
    val_batch_size = params["common"]["val_batch_size"]
    num_class = params["common"]["num_class"]
    train_dataloader = get_dataloader(train_data_path, train_batch_size, num_class, image_size, is_train=True)
    val_dataloder = get_dataloader(val_data_path, val_batch_size, num_class, is_train=False)

    dt_now = datetime.datetime.now()
    result_dir = f"./result/{dt_now.year}{dt_now.month:0>2}{dt_now.day:0>2}-{dt_now.hour:0>2}{dt_now.minute:0>2}/"

    os.makedirs(result_dir, exist_ok=True)

    with open(f"{result_dir}/params.yaml", "w") as f:
        f.write(yaml.dump(params, default_flow_style=False))

    trainer.train(model, result_dir, train_dataloader=train_dataloader, val_dataloader=val_dataloder)

def predict():
    pass


if __name__=="__main__":

    args = sys.argv
    mode = args[1]
    param_file = args[2]

    assert mode in ["train", "predict", "calculate_iou"]

    if mode == "train":
        with open(param_file, "r") as f:
            params = yaml.load(f)
        train(params)

    elif mode == "predict":
        model_path = sys.args[3]


