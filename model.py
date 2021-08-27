import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions.normal import Normal
from torch.distributions.one_hot_categorical import OneHotCategorical

class CVRAE_forecasting(nn.Module):
    def __init__(self, act_dim, h_dim, z_dim, n_layers):
        super().__init__()

        self.act_dim = act_dim
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.n_layers = n_layers

        self.x_enc = nn.Linear(act_dim + 1, h_dim)
        self.act_enc =  nn.Linear(act_dim, h_dim)
        self.hidden_to_z = nn.Linear(h_dim, z_dim)
        self.phi_x_hidden_to_enc = nn.Linear(h_dim + h_dim, h_dim)
        self.hidden_to_prior = nn.Linear(h_dim, h_dim)
        self.z_to_phi_z = nn.Linear(z_dim, h_dim)
        self.phi_z_hidden_to_dec = nn.Linear(h_dim + h_dim, h_dim)
        self.dec_to_act = nn.Linear(h_dim, act_dim)
        self.dur_decoder = nn.Linear(h_dim + h_dim, 1)

        self.rnn = nn.GRU(h_dim + h_dim, h_dim, n_layers)

        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()

        self.cross_entropy = nn.CrossEntropyLoss()
        self.gaussian_nll = nn.GaussianNLLLoss(full=True)

    def forward(self, act,dur):
        act_one_hot = F.one_hot(torch.tensor(act),num_classes = self.act_dim).type(torch.long)
        x = torch.transpose(torch.cat([act_one_hot,dur.unsqueeze(-1)],-1),0,1)

        hidden = self.init_hidden(x)
        kld = 0
        for t in range(x.size(0)):
            phi_x_t = self.relu(self.x_enc(x[t]))

            enc_t = self.relu(self.phi_x_hidden_to_enc(torch.cat([phi_x_t, hidden[-1]], -1)))
            enc_mean_t = self.hidden_to_z(enc_t)
            enc_std_t = self.softplus(self.hidden_to_z(enc_t))

            prior_t = self.relu(self.hidden_to_prior(hidden))
            prior_mean_t = self.hidden_to_z(prior_t)
            prior_std_t = self.softplus(self.hidden_to_z(prior_t))

            kld += self._kld_gauss(enc_mean_t, enc_std_t, prior_mean_t, prior_std_t)

            z_t = self._reparameterized_sample(enc_mean_t, enc_std_t)
            phi_z_t = self.relu(self.z_to_phi_z(z_t))

            _, hidden = self.rnn(torch.cat([phi_x_t.unsqueeze(0), phi_z_t.unsqueeze(0)], -1), hidden)

        prior_t = self.relu(self.hidden_to_prior(hidden[-1]))
        prior_mean_t = self.hidden_to_z(prior_t)
        prior_std_t = self.softplus(self.hidden_to_z(prior_t))

        z_t = self._reparameterized_sample(prior_mean_t, prior_std_t)
        phi_z_t = self.relu(self.z_to_phi_z(z_t))

        dec_t = self.relu(self.phi_z_hidden_to_dec(torch.cat([phi_z_t, hidden[-1]], -1)))

        unnorm_prob_pred_act = self.dec_to_act(dec_t)

        fut_act_encoded = self.act_enc(unnorm_prob_pred_act)
        dur_dec = self.dur_decoder(torch.cat([fut_act_encoded.clone(), dec_t], -1))

        pred_dur_mean = self.sigmoid(dur_dec)
        pred_dur_std = self.softplus(dur_dec)

        return unnorm_prob_pred_act, (pred_dur_mean,pred_dur_std), kld

    def generate(self, act, dur, total_dur='dur', mean='mean_dur', std='std_dur'):
        with torch.no_grad():
            act_one_hot = F.one_hot(torch.tensor(act),num_classes = self.act_dim).type(torch.long)
            x = torch.cat([act_one_hot,dur.unsqueeze(-1)],-1).unsqueeze(1)

            hidden = self.init_hidden(x)
            pred_acts = []
            pred_durations = []
            pred_duration_so_far = 0

            for t in range(x.size(0)):
                phi_x_t = self.relu(self.x_enc(x[t]))

                enc_t = self.relu(self.phi_x_hidden_to_enc(torch.cat([phi_x_t, hidden[-1]], -1)))
                enc_mean_t = self.hidden_to_z(enc_t)
                enc_std_t = self.softplus(self.hidden_to_z(enc_t))

                z_t = self._reparameterized_sample(enc_mean_t, enc_std_t)
                phi_z_t = self.relu(self.z_to_phi_z(z_t))

                _, hidden = self.rnn(
                    torch.cat([phi_x_t.unsqueeze(0), phi_z_t.unsqueeze(0)], -1), hidden)


            # start generation by sampling duration of last observed action
            dec_t = self.relu(self.phi_z_hidden_to_dec(torch.cat([phi_z_t, hidden[-1]], -1)))
            last_obs_act_encoded = self.act_enc(x[-1,:,:self.act_dim])

            pred_dur_mean = self.sigmoid(
                self.dur_decoder(torch.cat([last_obs_act_encoded, dec_t], -1)))
            pred_dur_std = self.softplus(
                self.dur_decoder(torch.cat([last_obs_act_encoded, dec_t], -1)))

            last_obs_act_dur_pred_normalised = Normal(pred_dur_mean, pred_dur_std).sample((50,))
            last_obs_act_dur_pred = torch.mean(last_obs_act_dur_pred_normalised) * std + mean
            #last_obs_act_dur_pred_unnormalised = pred_dur_mean * std + mean

            last_obs_act_dur = x[-1,:,-1] * std + mean

            if last_obs_act_dur_pred > last_obs_act_dur:
                dur[t] = last_obs_act_dur_pred
                pred_acts.append(torch.argmax(act[t], -1).item())

                first_pred_act_dur = (last_obs_act_dur_pred - last_obs_act_dur)
                pred_durations.append(first_pred_act_dur.item())
                pred_duration_so_far += first_pred_act_dur

            while True:
                prior_t = self.relu(self.hidden_to_prior(hidden[-1]))
                prior_mean_t = self.hidden_to_z(prior_t)
                prior_std_t = self.softplus(self.hidden_to_z(prior_t))

                z_t = self._reparameterized_sample(prior_mean_t, prior_std_t)
                phi_z_t = self.relu(self.z_to_phi_z(z_t))

                dec_t = self.relu(self.phi_z_hidden_to_dec(torch.cat([phi_z_t, hidden[-1]], -1)))

                #print(Categorical(self.softmax(self.dec_to_act(dec_t))).sample((10,)))
                act_samples_t = OneHotCategorical(self.softmax(self.dec_to_act(dec_t))).sample((50,))

                act_sample_histogram = torch.histc((act_samples_t == 1).nonzero()[:, -1].type(torch.float),
                                                   bins=self.act_dim, min=0, max=self.act_dim - 1)

                pred_act = torch.argmax(act_sample_histogram, -1)

                #pred_act = torch.argmax(self.softmax(self.dec_to_act(dec_t)))

                pred_act_one_hot = torch.nn.functional.one_hot(pred_act, num_classes=self.act_dim).type(torch.float).unsqueeze(0)

                pred_dur_mean = self.sigmoid(
                    self.dur_decoder(torch.cat([self.act_enc(pred_act_one_hot), dec_t], -1)))
                pred_dur_std = self.softplus(
                    self.dur_decoder(torch.cat([self.act_enc(pred_act_one_hot), dec_t], -1)))

                pred_dur = torch.mean(Normal(pred_dur_mean, pred_dur_std).sample((50,)), 0, True)[-1]
                #print(pred_dur)
                pred_dur_unnorm = pred_dur.item() * std + mean
                #pred_act_dur_unnormalised = pred_dur_mean[0][0].item() * std + mean

                phi_x_t = self.relu(self.x_enc(torch.cat([pred_act_one_hot,pred_dur],-1))).unsqueeze(0)

                _, hidden = self.rnn(torch.cat([phi_x_t, phi_z_t.unsqueeze(0)], -1), hidden)

                if pred_acts:
                    if pred_acts[-1] != pred_act:
                        pred_acts.append(pred_act.item())
                        pred_durations.append(pred_dur_unnorm)

                    elif pred_acts[-1] == pred_act:
                        pred_durations[-1] += pred_dur_unnorm

                elif not pred_acts:
                    pred_acts.append(pred_act.item())
                    pred_durations.append(pred_dur_unnorm)

                pred_duration_so_far += pred_dur_unnorm
                if pred_duration_so_far >= total_dur:
                    break

            return pred_acts, pred_durations

    def init_hidden(self, obs_seq):
        return torch.randn(self.n_layers, obs_seq.size(1), self.h_dim)

    def _reparameterized_sample(self, mean, std):
        """using std to sample"""
        eps = torch.randn(std.shape, requires_grad=True)
        return eps.mul(std).add_(mean)

    def _kld_gauss(self, mean_1, std_1, mean_2, std_2):
        """Using std to compute KLD"""
        kld_element = (2 * torch.log(std_2) - 2 * torch.log(std_1) +
                       (std_1.pow(2) + (mean_1 - mean_2).pow(2)) /
                       std_2.pow(2) - 1)
        return .5 * torch.sum(kld_element)


class CVRAE_next_action(nn.Module):
    def __init__(self, act_dim, h_dim, z_dim, n_layers):
        super().__init__()
        self.act_dim = act_dim
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.n_layers = n_layers

        self.phi_x = nn.Sequential(nn.Linear(self.act_dim, h_dim),
                                   nn.ReLU())
        self.phi_encoder = nn.Sequential(nn.Linear(h_dim + h_dim, h_dim),
                                         nn.ReLU(),
                                         nn.Linear(h_dim, z_dim))
        self.phi_prior = nn.Sequential(nn.Linear(h_dim, h_dim),
                                       nn.ReLU(),
                                       nn.Linear(h_dim, z_dim))
        self.phi_z = nn.Sequential(nn.Linear(z_dim, h_dim),
                                   nn.ReLU())
        self.phi_decoder = nn.Sequential(nn.Linear(h_dim + h_dim, h_dim),
                                         nn.ReLU())

        self.phi_act = nn.Linear(act_dim, h_dim)

        self.act_decoder = nn.Linear(h_dim, act_dim)
        self.dur_decoder = nn.Linear(h_dim + h_dim, 1)

        self.rnn = nn.GRU(h_dim + h_dim, h_dim, n_layers)

        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # encode observations
        x_one_hot = torch.transpose(F.one_hot(x,num_classes = self.act_dim).type(torch.float),0,1)
        h = self.init_hidden(x_one_hot)
        kld = 0
        print(x_one_hot.shape)
        for t in range(x_one_hot.size(0)):
            x_t = x_one_hot[t,:,:]

            #print(self.phi_x(x_t).shape)
            enc_t = self.phi_encoder(torch.cat([self.phi_x(x_t), h[-1]], -1))

            enc_mean_t = enc_t
            enc_std_t = self.softplus(enc_t)

            prior_t = self.phi_prior(h[-1])
            prior_mean_t = prior_t
            prior_std_t = self.softplus(prior_t)

            kld += self._kld_gauss(enc_mean_t, enc_std_t, prior_mean_t, prior_std_t)

            z_t = self._reparameterized_sample(enc_mean_t, enc_std_t)

            _, h = self.rnn(torch.cat([self.phi_x(x_t).unsqueeze(0), self.phi_z(z_t).unsqueeze(0)], -1), h)

        prior_t = self.phi_prior(h[-1])
        prior_mean_t = prior_t
        prior_std_t = self.softplus(prior_t)

        z_t = self._reparameterized_sample(prior_mean_t, prior_std_t)

        dec_t = self.phi_decoder(torch.cat([self.phi_z(z_t), h[-1]], -1))

        pred_act = self.act_decoder(dec_t)  # unnormalised probability

        return pred_act, kld

    def init_hidden(self, x):
        return torch.randn(self.n_layers, x.size(1), self.h_dim)

    def _reparameterized_sample(self, mean, std):
        """using std to sample"""
        eps = torch.randn(std.shape, requires_grad=True)
        return eps.mul(std).add_(mean)

    def _kld_gauss(self, mean_1, std_1, mean_2, std_2):
        """Using std to compute KLD"""
        kld_element = (2 * torch.log(std_2) - 2 * torch.log(std_1) +
                       (std_1.pow(2) + (mean_1 - mean_2).pow(2)) /
                       std_2.pow(2) - 1)
        return .5 * torch.sum(kld_element)
