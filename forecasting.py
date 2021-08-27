import torch.nn as nn
import numpy as np
from sklearn.metrics import balanced_accuracy_score

from model import *
from dataloader import *

torch.autograd.set_detect_anomaly(True)

n_epochs = 1

def train(dataset,split,n_epochs,h_dim,z_dim,n_layers, ground_truth = True):

    trainset = ActionForecastingDataset(dataset,split,ground_truth)
    trainloader = DataLoader(trainset, batch_size=1, collate_fn = generate_train_batch, shuffle=True, num_workers=0)

    model = CVRAE_forecasting(trainset.n_acts,h_dim,z_dim,n_layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)

    kld_losses = []
    act_losses = []
    dur_losses = []

    cross_entropy = nn.CrossEntropyLoss()
    gaussian_nll = nn.GaussianNLLLoss()

    for epoch in range(1, n_epochs + 1):
        print('training epoch {}'.format(epoch))
        running_loss = 0.0
        running_kld = 0.0
        running_act_loss = 0.0
        running_dur_loss = 0.0

        for epoch_steps, data in enumerate(trainloader):
            optimizer.zero_grad()
            obs_acts,obs_durs = data[0]

            obs_dur_norm = (torch.FloatTensor(obs_durs) - trainset.dur_mean) / trainset.dur_std

            pred_act_prob, pred_dur_params, kl_loss = model(obs_acts, obs_dur_norm)
            target_act,target_dur = data[1]
            tar_dur_norm = (torch.FloatTensor(target_dur) - trainset.dur_mean) / trainset.dur_std

            act_loss = cross_entropy(pred_act_prob, torch.tensor(target_act))
            dur_loss = gaussian_nll(pred_dur_params[0],tar_dur_norm , pred_dur_params[-1])

            loss = act_loss + dur_loss + kl_loss
            loss.backward()
            optimizer.step()

            running_kld += kl_loss.item()
            running_act_loss += act_loss.item()
            running_dur_loss += dur_loss.item()

        epoch_kld = running_kld / epoch_steps
        epoch_act_loss = running_act_loss / epoch_steps
        epoch_dur_loss = running_dur_loss / epoch_steps

        print('epoch {} completed'.format(epoch))
        print("[%d] Train loss: %.3f, kld: %.3f, act_loss: %.3f, dur_loss: %.3f" % (epoch, epoch_loss, epoch_kld, epoch_act_loss, epoch_dur_loss))

        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        },'saved_models/cvrae_h_{}_z_{}_rnn_layers_{}_forecasting_{}_{}.pth.tar'.format(model.h_dim, model.z_dim, model.n_layers, dataset, split))



def eval(dataset,split,obs_percent,pred_percent,h_dim,z_dim,n_layers,model_path=False,ground_truth = True):
    testset = ActionForecastingDataset(dataset, split, obs_percent, train_or_test = 'test', ground_truth=ground_truth)
    testloader = DataLoader(testset, batch_size=1, shuffle=False)


    model = CVRAE_forecasting(testset.n_acts, h_dim, z_dim, n_layers)

    if model_path:
        checkpoint = torch.load(model_path)
        model.state_dict(checkpoint['state_dict'])

    targets = []
    preds = []

    for i, data in enumerate(testloader):
        obs,target = data

        obs_acts, obs_durs = obs
        obs_dur_norm = (torch.FloatTensor(obs_durs) - testset.dur_mean) / testset.dur_std

        target_acts, target_durs = target
        total_dur = sum(target_durs)

        pred_acts, pred_durs = model.generate(obs_acts, obs_dur_norm, total_dur, testset.dur_mean,testset.dur_std)
        pred_durs = [dur if dur >= 0 else 1 for dur in pred_durs]

        pred_framewise = list(np.repeat(pred_acts, pred_durs, axis=0))

        target_framewise = list(np.repeat(torch.tensor(target_acts), torch.tensor(target_durs), axis=0))

        target_framewise = target_framewise[:int(total_dur* pred_percent)]

        if len(pred_framewise) > len(target_framewise):
            pred_framewise = pred_framewise[:len(target_framewise)]

        elif len(pred_framewise) < len(target_framewise):
            diff = len(target_framewise) - len(pred_framewise)
            pred_framewise.extend([pred_framewise[-1]] * diff)

        preds.append(pred_framewise)
        targets.append(target_framewise)

    preds_flatten = [act for pred in preds for act in pred]
    targets_flatten = [act for target in targets for act in target]

    accuracy = round(balanced_accuracy_score(targets_flatten, preds_flatten), 5)
    return accuracy
