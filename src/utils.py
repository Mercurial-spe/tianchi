import numpy as np
import cv2
import os
import time
from datetime import datetime
import torch
import torch.nn as nn

def is_valid_image(image_path):
    if not os.path.exists(image_path):
        return False
    image = cv2.imread(image_path)
    return image is not None

def validate(val_loader, model, criterion):
    # switch to evaluate mode
    model.eval()

    total_acc = 0
    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            input = input.cuda()
            target = target.cuda()

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            total_acc += (output.argmax(1).long() == target.long()).sum().item()
    return total_acc / len(val_loader.dataset)

def train(train_loader, model, criterion, optimizer, epoch):
    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        output = model(input)
        loss = criterion(output, target)
        acc1 = (output.argmax(1).long() == target.long()).sum().item()

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print(datetime.now(), loss.item(), acc1 / input.size(0))

def predict(test_loader, model, criterion):
    # switch to evaluate mode
    model.eval()
    pred = [] 
    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(test_loader):
            input = input.cuda()
            target = target.cuda()

            # compute output
            output = model(input)
            loss = criterion(output, target)

            pred += list(output.argmax(1).long().cpu().numpy())
    return pred 