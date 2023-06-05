from collections import OrderedDict

import torch
import torch.nn as nn


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def initialize(self, m):
        if isinstance(m, (nn.Conv1d)):
            # nn.init.xavier_uniform_(m.weight)
            # if m.bias is not None:
            #     nn.init.xavier_uniform_(m.bias)

            nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")


class SampleCNN(Model):
    def __init__(self, strides, supervised, out_dim):
        super(SampleCNN, self).__init__()

        self.strides = strides
        self.supervised = supervised
        self.sequential = [
            nn.Sequential(
                nn.Conv1d(1, 128, kernel_size=3, stride=3, padding=0),
                nn.BatchNorm1d(128),
                nn.ReLU(),
            )
        ]

        self.hidden = [
            [128, 128],
            [128, 128],
            [128, 256],
            [256, 256],
            [256, 256],
            [256, 256],
            [256, 256],
            [256, 256],
            [256, 512],
        ]

        assert len(self.hidden) == len(
            self.strides
        ), "Number of hidden layers and strides are not equal"
        for stride, (h_in, h_out) in zip(self.strides, self.hidden):
            self.sequential.append(
                nn.Sequential(
                    nn.Conv1d(h_in, h_out, kernel_size=stride, stride=1, padding=1),
                    nn.BatchNorm1d(h_out),
                    nn.ReLU(),
                    nn.MaxPool1d(stride, stride=stride),
                )
            )

        # 1 x 512
        self.sequential.append(
            nn.Sequential(
                nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm1d(512),
                nn.ReLU(),
            )
        )

        self.sequential = nn.Sequential(*self.sequential)

        if self.supervised:
            self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(512, out_dim)

    def forward(self, x):
        out = self.sequential(x)
        if self.supervised:
            out = self.dropout(out)

        out = out.reshape(x.shape[0], out.size(1) * out.size(2))
        logit = self.fc(out)
        return logit


def load_encoder_checkpoint(checkpoint_path: str, output_dim: int) -> OrderedDict:
    state_dict = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    if "pytorch-lightning_version" in state_dict.keys():
        new_state_dict = OrderedDict(
            {
                k.replace("model.encoder.", ""): v
                for k, v in state_dict["state_dict"].items()
                if "model.encoder." in k
            }
        )
    else:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if "encoder." in k:
                new_state_dict[k.replace("encoder.", "")] = v

    new_state_dict["fc.weight"] = torch.zeros(output_dim, 512)
    new_state_dict["fc.bias"] = torch.zeros(output_dim)
    return new_state_dict
    
