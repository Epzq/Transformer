import math
import torch
import torch.nn as nn

import numpy as np

from torch.nn.utils.rnn import pad_sequence

import matplotlib.pyplot as plt

import argparse

import time

import random

from BatchGen import BatchGenerator,LabelsDataset
from Transformer import Seq2Seq,Encoder,Decoder
from Trainer import train,evaluate,accuracy

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser()
parser.add_argument('--action', default='train')
parser.add_argument('--dataset', default="breakfast")
parser.add_argument('--Epoch', default = '100',type=int)
parser.add_argument('--LR', default = '0.0005',type=float)
parser.add_argument('--Batch', default = '1',type=int)

args = parser.parse_args()
#device = torch.device("cpu")

vid_list_file = "./"+args.dataset+"/splits/train.split1.bundle"
vid_list_file_tst = "./"+args.dataset+"/splits/test.split1.bundle"

gt_path = "./"+args.dataset+"/groundTruth/"

arrays= "./"+args.dataset+"/predictions/"


train_pth = "./"+args.dataset+"/train"
test_pth = "./"+args.dataset+"/test"

mapping_file = "./"+args.dataset+"/mapping.txt"

file_ptr = open(mapping_file, 'r') 
actions = file_ptr.read().split('\n')[:-1]
actions_dict=dict()
for a in actions:
    actions_dict[a.split()[1]] = (int(a.split()[0]))

SOS_index = actions_dict['SOS']
EOS_index = actions_dict['EOS']
    
batch = args.Batch

def collate_fn_padd(batch):
    '''
    Padds batch of variable length

    note: it converts things ToTensor manually here since the ToTensor transform
    assume it takes in images rather than arbitrary tensors.
    '''
    ## get sequence lengths
    ## padd
    batch_input = [ t[0] for t in batch]
    batch_input = torch.nn.utils.rnn.pad_sequence(batch_input,batch_first=True)
    batch_targets = [ t[1] for t in batch]
    batch_targets = torch.nn.utils.rnn.pad_sequence(batch_targets,batch_first=True)
    return batch_input,batch_targets   


Batch_gen =  BatchGenerator(actions_dict,gt_path,vid_list_file,vid_list_file_tst,arrays,SOS_index,EOS_index)
input20,tar2010,tar2020,tar2030,tar2050,input30,tar3010,tar3020,tar3030,tar3050 = Batch_gen.GT_train_data()  
input20_tst,tar2010_tst,tar2020_tst,tar2030_tst,tar2050_tst,input30_tst,tar3010_tst,tar3020_tst,tar3030_tst,tar3050_tst=Batch_gen.GT_test_data()    
input20_pred,tar2010_pred,tar2020_pred,tar2030_pred,tar2050_pred,input30_pred,tar3010_pred,tar3020_pred,tar3030_pred,tar3050_pred=Batch_gen.ASRF_pred()



train_20to10_dataset = LabelsDataset(input20,tar2010)
train_20to20_dataset = LabelsDataset(input20,tar2020)
train_20to30_dataset = LabelsDataset(input20,tar2030)
train_20to50_dataset = LabelsDataset(input20,tar2050)
train_30to10_dataset = LabelsDataset(input30,tar3010)
train_30to20_dataset = LabelsDataset(input30,tar3020)
train_30to30_dataset = LabelsDataset(input30,tar3030)
train_30to50_dataset = LabelsDataset(input30,tar3050)
test_20to10_dataset = LabelsDataset(input20_tst,tar2010_tst)
test_20to20_dataset = LabelsDataset(input20_tst,tar2020_tst)
test_20to30_dataset = LabelsDataset(input20_tst,tar2030_tst)
test_20to50_dataset = LabelsDataset(input20_tst,tar2050_tst)
test_30to10_dataset = LabelsDataset(input30_tst,tar3010_tst)
test_30to20_dataset = LabelsDataset(input30_tst,tar3020_tst)
test_30to30_dataset = LabelsDataset(input30_tst,tar3030_tst)
test_30to50_dataset = LabelsDataset(input30_tst,tar3050_tst)
pred_20to10_dataset = LabelsDataset(input20_pred,tar2010_pred)
pred_20to20_dataset = LabelsDataset(input20_pred,tar2020_pred)
pred_20to30_dataset = LabelsDataset(input20_pred,tar2030_pred)
pred_20to50_dataset = LabelsDataset(input20_pred,tar2050_pred)
pred_30to10_dataset = LabelsDataset(input30_pred,tar3010_pred)
pred_30to20_dataset = LabelsDataset(input30_pred,tar3020_pred)
pred_30to30_dataset = LabelsDataset(input30_pred,tar3030_pred)
pred_30to50_dataset = LabelsDataset(input30_pred,tar3050_pred)

train_20to10_loader = torch.utils.data.DataLoader(train_20to10_dataset,batch_size=batch,shuffle=True, collate_fn=collate_fn_padd)
train_20to20_loader = torch.utils.data.DataLoader(train_20to20_dataset,batch_size=batch,shuffle=True, collate_fn=collate_fn_padd)
train_20to30_loader = torch.utils.data.DataLoader(train_20to30_dataset,batch_size=batch,shuffle=True, collate_fn=collate_fn_padd)
train_20to50_loader = torch.utils.data.DataLoader(train_20to50_dataset,batch_size=batch,shuffle=True, collate_fn=collate_fn_padd)

train_30to10_loader = torch.utils.data.DataLoader(train_30to10_dataset,batch_size=batch,shuffle=True, collate_fn=collate_fn_padd)
train_30to20_loader = torch.utils.data.DataLoader(train_30to20_dataset,batch_size=batch,shuffle=True, collate_fn=collate_fn_padd)
train_30to30_loader = torch.utils.data.DataLoader(train_30to30_dataset,batch_size=batch,shuffle=True, collate_fn=collate_fn_padd)
train_30to50_loader = torch.utils.data.DataLoader(train_30to50_dataset,batch_size=batch,shuffle=True, collate_fn=collate_fn_padd)

test_20to10_loader = torch.utils.data.DataLoader(test_20to10_dataset,batch_size=batch,shuffle=True, collate_fn=collate_fn_padd)
test_20to20_loader = torch.utils.data.DataLoader(test_20to20_dataset,batch_size=batch,shuffle=True, collate_fn=collate_fn_padd)
test_20to30_loader = torch.utils.data.DataLoader(test_20to30_dataset,batch_size=batch,shuffle=True, collate_fn=collate_fn_padd)
test_20to50_loader = torch.utils.data.DataLoader(test_20to50_dataset,batch_size=batch,shuffle=True, collate_fn=collate_fn_padd)

test_30to10_loader = torch.utils.data.DataLoader(test_30to10_dataset,batch_size=batch,shuffle=True, collate_fn=collate_fn_padd)
test_30to20_loader = torch.utils.data.DataLoader(test_30to20_dataset,batch_size=batch,shuffle=True, collate_fn=collate_fn_padd)
test_30to30_loader = torch.utils.data.DataLoader(test_30to30_dataset,batch_size=batch,shuffle=True, collate_fn=collate_fn_padd)
test_30to50_loader = torch.utils.data.DataLoader(test_30to50_dataset,batch_size=batch,shuffle=True, collate_fn=collate_fn_padd)

acc_20to10_loader = torch.utils.data.DataLoader(test_20to10_dataset,batch_size=1,shuffle=True, collate_fn=collate_fn_padd)
acc_20to20_loader = torch.utils.data.DataLoader(test_20to20_dataset,batch_size=1,shuffle=True, collate_fn=collate_fn_padd)
acc_20to30_loader = torch.utils.data.DataLoader(test_20to30_dataset,batch_size=1,shuffle=True, collate_fn=collate_fn_padd)
acc_20to50_loader = torch.utils.data.DataLoader(test_20to50_dataset,batch_size=1,shuffle=False, collate_fn=collate_fn_padd)

acc_30to10_loader = torch.utils.data.DataLoader(test_30to10_dataset,batch_size=1,shuffle=True, collate_fn=collate_fn_padd)
acc_30to20_loader = torch.utils.data.DataLoader(test_30to20_dataset,batch_size=1,shuffle=True, collate_fn=collate_fn_padd)
acc_30to30_loader = torch.utils.data.DataLoader(test_30to30_dataset,batch_size=1,shuffle=True, collate_fn=collate_fn_padd)
acc_30to50_loader = torch.utils.data.DataLoader(test_30to50_dataset,batch_size=1,shuffle=True, collate_fn=collate_fn_padd)

pred_20to10_loader = torch.utils.data.DataLoader(pred_20to10_dataset,batch_size=batch,shuffle=True, collate_fn=collate_fn_padd)
pred_20to20_loader = torch.utils.data.DataLoader(pred_20to20_dataset,batch_size=batch,shuffle=True, collate_fn=collate_fn_padd)
pred_20to30_loader = torch.utils.data.DataLoader(pred_20to30_dataset,batch_size=batch,shuffle=True, collate_fn=collate_fn_padd)
pred_20to50_loader = torch.utils.data.DataLoader(pred_20to50_dataset,batch_size=batch,shuffle=True, collate_fn=collate_fn_padd)

pred_30to10_loader = torch.utils.data.DataLoader(pred_30to10_dataset,batch_size=batch,shuffle=True, collate_fn=collate_fn_padd)
pred_30to20_loader = torch.utils.data.DataLoader(pred_30to20_dataset,batch_size=batch,shuffle=True, collate_fn=collate_fn_padd)
pred_30to30_loader = torch.utils.data.DataLoader(pred_30to30_dataset,batch_size=batch,shuffle=True, collate_fn=collate_fn_padd)
pred_30to50_loader = torch.utils.data.DataLoader(pred_30to50_dataset,batch_size=batch,shuffle=True, collate_fn=collate_fn_padd)



INPUT_DIM = len(actions_dict)+1
OUTPUT_DIM = INPUT_DIM
HID_DIM = 64
ENC_LAYERS = 1
DEC_LAYERS = 1
ENC_HEADS = 4
DEC_HEADS = 4
ENC_PF_DIM = 64
DEC_PF_DIM = 64
ENC_DROPOUT = 0.1
DEC_DROPOUT = 0.1

enc = Encoder(INPUT_DIM, 
              HID_DIM, 
              ENC_LAYERS, 
              ENC_HEADS, 
              ENC_PF_DIM, 
              ENC_DROPOUT, 
              device)

dec = Decoder(OUTPUT_DIM, 
              HID_DIM, 
              DEC_LAYERS, 
              DEC_HEADS, 
              DEC_PF_DIM, 
              DEC_DROPOUT, 
              device)

model = Seq2Seq(enc, dec, 0, 0, device)
model.to(device)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(model):,} trainable parameters')

def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)
        
model.apply(initialize_weights);

LEARNING_RATE = args.LR
optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)

criterion = nn.CrossEntropyLoss(ignore_index=0)
duration_loss=nn.MSELoss()
rel=nn.ReLU()
sig=nn.Sigmoid()

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

N_EPOCHS = args.Epoch
CLIP = 1

Epoch_tacc=[]
Epoch_tloss=[]
Epoch_vloss=[]
Epoch_tacc=[]
Epoch_test=[]
for epoch in range(N_EPOCHS):
    
    
    start_time = time.time()
    train_loss = train(model,train_20to50_loader, optimizer, criterion,duration_loss,sig,CLIP,device)
    #valid_loss = evaluate(model, test_20to50_loader, criterion,duration_loss,sig,device)
    #train_acc = accuracy(train_20to50_loader,model,device,sig)
    epoch_acc = accuracy(acc_20to50_loader,model,device,sig,SOS_index)
    print(epoch_acc)
    end_time = time.time()
    
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
    
    print(f'Epoch: {epoch+1:2} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    #print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
    #print(f'\tTrain acc: {train_acc:.5f} | Val Acc: {epoch_acc:.5f}')
    Epoch_tloss.append(train_loss)
    #Epoch_vloss.append(valid_loss)
    #Epoch_tacc.append(train_acc)
    Epoch_test.append(epoch_acc)
