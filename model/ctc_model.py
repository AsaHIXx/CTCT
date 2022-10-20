
import torch.nn as nn
import torch.nn.functional as F
import torch


class LabelPredictor(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(LabelPredictor, self).__init__()
        self.fc = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU())
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.out_dim = out_dim

    def forward(self, x):
        latent = self.fc(x)
        x = F.relu(self.fc2(latent))

        return x, latent, F.softmax(x, dim=1)

    def add_units(self, n_new):
        '''
        n_new : integer variable counting the neurons you want to add
        '''

        # take a copy of the current weights stored in self._fc
        current = self.fc2.weight.data
        #         print('Current', current.shape)
        current_bias = self.fc2.bias.data  # Only used at the end of the post

        # randomly initialize a tensor with the size of the wanted layer
        hl_input = torch.zeros([n_new, current.shape[1]])
        nn.init.xavier_uniform_(hl_input, gain=nn.init.calculate_gain('relu'))

        # concatenate the old weights with the new weights
        new_wi = torch.cat([current, hl_input], dim=0)

        # reset weight and grad variables to new size
        self.fc2 = nn.Linear(current.shape[1], self.out_dim + n_new)  # 2 is the size of my output layer

        # set the weight data to new values
        self.fc2.weight = torch.nn.Parameter(new_wi)


class FeatureExtractor_fc(nn.Module):
    def __init__(self, num_inputs, embed_size):
        super(FeatureExtractor_fc, self).__init__()
        self.in_features = embed_size
        self.feature_layers = nn.Sequential(
            nn.Linear(num_inputs, 1024),
            nn.LeakyReLU(0.02),
            nn.Dropout(p=0.4),

            nn.Linear(1024, 512),
            # nn.BatchNorm1d(512),

            nn.ReLU(),
            nn.Dropout(p=0.4),

            nn.Linear(512, self.in_features),
            nn.ReLU(),

        )

    def output_num(self):
        return self.in_features

    def forward(self, x, is_dec=False):
        extrac_f = self.feature_layers(x)
        return extrac_f


class Ctcnet(nn.Module):
    def __init__(self, num_inputs, embed_size, class_out):
        super(Ctcnet, self).__init__()
        self.num_inputs = num_inputs
        self.embed_size = embed_size
        self.class_out = class_out
        self.features = FeatureExtractor_fc(num_inputs, embed_size)
        self.classifier = LabelPredictor(self.embed_size, 100, self.class_out)

    def forward(self, x):
        feat = self.features(x)
        prob, latent, softmax_res = self.classifier(feat)
        bottleneck = feat
        return feat, bottleneck, prob, softmax_res, latent, feat, feat

    def optim_parameters(self, lr):
        d = [{'params': self.features.parameters(), 'lr': lr},
             {'params': self.classifier.parameters(), 'lr': lr * 10}]
        return d
