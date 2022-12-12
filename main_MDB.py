"""
-*- coding: utf-8 -*-

@Time    : 2021/10/21 10:17

@Author  : Wang Ziming

@Email   : zi_ming_wang@outlook.com

@File    : main_TD.py
"""
import os
import sys
import scipy.io as io
import numpy as np
import torch
import os
from tqdm import tqdm
from model.util import setup_seed
from torch.utils.tensorboard import SummaryWriter
from model.ptn_loader import sPtn
from model.surrogate_act import SurrogateHeaviside
from model.snn import STCADenseLayer, ReadoutLayer, SNN

# num of neurons in the first and the last layer
num_in = 384
num_class = 10
dt = 3e-3


def getMDB(root_path, train_batch_size=1, val_batch_size=1, mode='spike'):
    data = io.loadmat(root_path)
    train_source = data['TrainData']
    test_source = data['TestData']
    train_data = sPtn(dataSource=train_source['ptn'][0, 0], labelSource=train_source['Labels'][0, 0].squeeze(),
                      TmaxSource=train_source['Tmax'][0, 0], mode='neuron',
                      dt=dt)
    test_data = sPtn(dataSource=test_source['ptn'][0, 0], labelSource=test_source['Labels'][0, 0].squeeze(),
                     TmaxSource=test_source['Tmax'][0, 0],
                     mode='neuron',
                     dt=dt)
    train_ldr = torch.utils.data.DataLoader(dataset=train_data, batch_size=train_batch_size, shuffle=True,
                                            pin_memory=False, num_workers=0)
    test_ldr = torch.utils.data.DataLoader(dataset=test_data, batch_size=val_batch_size, shuffle=False,
                                           pin_memory=False, num_workers=0)
    return train_ldr, test_ldr


def runTrain(train_ldr, optimizer, snn, evaluator):
    global device, dtype
    loss_record = []
    predict_tot = []
    label_tot = []
    snn.train()
    for idx, (ptns, labels) in enumerate(train_ldr):
        ptns = ptns.permute([0, 2, 1]).contiguous().to(device, dtype)
        # print(ptns.shape)
        labels = labels.to(device)
        # print(labels.dtype,labels.shape)
        optimizer.zero_grad()
        output, _ = snn(ptns)
        # target = torch.nn.functional.one_hot(labels)
        # print(target.shape)
        loss = evaluator(output, labels)
        loss.backward()
        torch.nn.utils.clip_grad_value_(snn.parameters(), 5)
        optimizer.step()
        snn.clamp()
        predict = torch.argmax(output, axis=1)
        loss_record.append(loss.detach().cpu())
        # print(torch.mean((predict==labels).float()))
        predict_tot.append(predict)
        # print(predict.shape)
        label_tot.append(labels)
        # print(torch.mean((predict == labels).float()))
    predict_tot = torch.cat(predict_tot)
    label_tot = torch.cat(label_tot)
    train_acc = torch.mean((predict_tot == label_tot).float())
    train_loss = torch.tensor(loss_record).sum() / len(label_tot)
    return train_acc, train_loss


def runTest(val_ldr, snn, evaluator):
    global test_trace, device, dtype
    snn.eval()
    with torch.no_grad():
        loss_record = []
        predict_tot = []
        label_tot = []
        for idx, (ptns, labels) in enumerate(val_ldr):
            ptns = ptns.permute([0, 2, 1]).contiguous().to(device, dtype)
            labels = labels.to(device)
            output, _ = snn(ptns)
            loss = evaluator(output, labels)
            loss_record.append(loss)
            # snn.clamp()
            predict = torch.argmax(output, axis=1)
            predict_tot.append(predict)
            label_tot.append(labels)
        predict_tot = torch.cat(predict_tot)
        label_tot = torch.cat(label_tot)
        val_acc = torch.mean((predict_tot == label_tot).float())
        val_loss = torch.tensor(loss_record).sum() / len(label_tot)
    return val_acc, val_loss


def main():
    model_list = [name for name in os.listdir(save_dir) if name.startswith(prefix)]
    file = prefix + 'cas_' + str(len(model_list))
    save_path = os.path.join(save_dir, file)
    train_ldr, val_ldr = getMDB(root_path='./data/M10_Data.mat', train_batch_size=64, val_batch_size=100)
    print('Finish loading MedlyDB from: ', save_path)
    # STCA full connected neural network
    layers = []
    # SurrogateHeaviside.sigma = 2
    spike_fn = SurrogateHeaviside.apply
    layers.append(STCADenseLayer(structure[0], structure[1], spike_fn, w_init_mean, w_init_std, recurrent=False,
                                 lateral_connections=False, fc_drop=0.5))
    # layers.append(SpikingDenseLayer(structure[1],structure[2],spike_fn,w_init_mean,w_init_std,recurrent=False,lateral_connections=True))
    layers.append(ReadoutLayer(structure[1], structure[2], w_init_mean, w_init_std))

    snn = SNN(layers).to(device, dtype)
    # optimizer = RAdam(snn.parameters(), lr=3e-4)
    # optimizer = torch.optim.AdamW(snn.parameters(), lr=3e-4, weight_decay=1e-3)
    # optimizer = torch.optim.AdamW(snn.parameters(), lr=3e-4)
    optimizer = torch.optim.Adam(snn.parameters(), lr=3e-4, amsgrad=False)
    # optimizer = torch.optim.AdamW(snn.parameters(), lr=3e-4, amsgrad=True)
    evaluator = torch.nn.CrossEntropyLoss()

    # define some hyperparameter and holk for a run
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    best_acc = 0
    test_trace = []
    train_trace = []
    loss_trace = []
    if (not os.path.exists(log_dir)):
        os.mkdir(log_dir)
    start_epoch = 0

    # resume
    if (load_if):
        state = torch.load(os.path.join(save_dir, file + '.t7'))
        snn.load_state_dict(state['best_net'])
        start_epoch = state['best_epoch']
        train_trace = state['traces']['train']
        test_trace = state['traces']['test']
        loss_trace = state['traces']['loss']

    # run  forward - backward
    for epoch in tqdm(range(start_epoch, start_epoch + epoch_num)):
        train_acc, train_loss = runTrain(train_ldr, optimizer, snn, evaluator)
        train_trace.append(train_acc)
        loss_trace.append(train_loss)
        print('\ntrain record: ', train_loss, train_acc)

        val_acc, val_loss = runTest(val_ldr, snn, evaluator)
        test_trace.append(val_acc)
        print('validation record:', val_loss, val_acc)
        if (val_acc > best_acc):
            best_acc = val_acc
            print('Saving model..  with acc {0} in the epoch {1}'.format(best_acc, epoch))
            state = {
                'best_acc': best_acc,
                'best_epoch': epoch,
                'best_net': snn.state_dict(),
                'traces': {'train': train_trace, 'test': test_trace, 'loss': loss_trace},
                # 'raw_pth': ptn_path
            }
            torch.save(state, os.path.join(save_path + '.t7'))


if __name__ == '__main__':
    # global config
    setup_seed(20)
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float
    structure = [num_in, 700, num_class]
    w_init_std = 0.1
    w_init_mean = 0.
    epoch_num = 60
    load_if = False
    save_dir = './checkpoints'
    log_dir = './log'
    prefix = 'MDB10_SPG_'
    print(device, dtype)
    main()
