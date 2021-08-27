import torch.nn as nn
import numpy as np
from sklearn.metrics import top_k_accuracy_score

from dataloader import *
from model import *


def train(dataset,split,n_epochs,h_dim,z_dim,n_layers,n_obs,ground_truth = True):
    trainset = NextActionDataset('train',dataset, split, n_obs,ground_truth=True)
    trainloader = DataLoader(trainset, batch_size=32, shuffle=True, num_workers=0)

    model = CVRAE_next_action(trainset.n_acts, h_dim, z_dim, n_layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)

    cross_entropy = nn.CrossEntropyLoss()

    running_kld = 0.0
    running_act_loss = 0.0
    for epoch in range(1, n_epochs + 1):
        print('training epoch {}'.format(epoch))
        running_loss = 0.0
        running_kld = 0.0
        running_act_loss = 0.0

        for epoch_steps, data in enumerate(trainloader):
            optimizer.zero_grad()
            x,y = data

            pred_act_prob, kl_loss = model(x)
            act_loss = cross_entropy(pred_act_prob, y)

            loss = act_loss + kl_loss

            loss.backward()
            optimizer.step()

            running_loss += loss
            running_kld += kl_loss.item()
            running_act_loss += act_loss.item()

        epoch_loss = running_loss / epoch_steps
        epoch_kld = running_kld / epoch_steps
        epoch_act_loss = running_act_loss / epoch_steps

        print('epoch {} completed'.format(epoch))
        print("[%d] kld: %.3f, act_loss: %.3f" % (epoch,
                                                  epoch_kld,
                                                  epoch_act_loss))
        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        },'saved_models/cvrae_h_{}_z_{}_rnn_layers_{}_next_action_{}_{}.pth.tar'.format(model.h_dim, model.z_dim, model.n_layers, dataset, split))

#train('50salads',1,30,64,64,2,3)

def eval(dataset,split,h_dim,z_dim,n_layers,n_obs,top_k,model_path=False,ground_truth = True):
    testset = NextActionDataset('test', dataset='50salads', split=1, n_obs=3, ground_truth=True)
    testloader = DataLoader(testset, batch_size=1, shuffle=False)

    if model_path:
        checkpoint = torch.load(model_path)
        model.state_dict(checkpoint['state_dict'])

    y_true = np.zeros(len(testset))
    y_pred = np.zeros((len(testset), testset.n_acts))

    for i, data in enumerate(testloader):
        with torch.no_grad():
            x,y = data
            pred_act,_ = model(x)
            #print(pred_act)
            y_true[i] = y
            y_pred[i] = pred_act

    return top_k_accuracy_score(y_true, y_pred, k=top_k,labels=np.array(list(testset.act2ix.values())))


eval('50salads',1,64,64,2,3,1)
