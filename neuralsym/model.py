import torch
import torch.nn as nn
import torch.nn.functional as F


class Highway(nn.Module):
    def __init__(self, size, num_layers, f, dropout, head=False, input_size=None):
        super(Highway, self).__init__()
        self.num_layers = num_layers
        self.f = f
        self.dropout = nn.Dropout(dropout)
        if head:
            assert input_size is not None
            self.nonlinear = nn.ModuleList([nn.Linear(input_size, size)])
            self.linear = nn.ModuleList([nn.Linear(input_size, size)])
            self.gate = nn.ModuleList([nn.Linear(input_size, size)])
        else:
            self.nonlinear = nn.ModuleList(
                [nn.Linear(size, size) for _ in range(num_layers)])
            self.linear = nn.ModuleList(
                [nn.Linear(size, size) for _ in range(num_layers)])
            self.gate = nn.ModuleList([nn.Linear(size, size)
                                      for _ in range(num_layers)])

    def forward(self, x):
        for layer in range(self.num_layers):
            gate = torch.sigmoid(self.gate[layer](x))

            nonlinear = self.f(self.nonlinear[layer](x))
            linear = self.linear[layer](x)

            x = gate * nonlinear + (1 - gate) * linear
            x = self.dropout(x)
        return x


class NN_Highway(nn.Module):
    def __init__(self, output_size, size=512, num_layers_body=5,
                 dropout_head=0.3, dropout_body=0.1,
                 f=F.elu, input_size=32681):
        super(NN_Highway, self).__init__()
        self.highway_head = Highway(
            size=size, num_layers=1, f=f, dropout=dropout_head,
            head=True, input_size=input_size
        )
        if num_layers_body <= 0:
            self.highway_body = None
        else:
            self.highway_body = Highway(
                size=size, num_layers=num_layers_body,
                f=f, dropout=dropout_body
            )

        self.classifier = nn.Linear(size, output_size)

    def forward(self, fp):
        if self.highway_body:
            embedding = self.highway_body(self.highway_head(fp))
        else:
            embedding = self.highway_head(fp)
        return self.classifier(embedding).squeeze(dim=1)


class NN_FC(nn.Module):
    def __init__(self, output_size, size=512,
                 dropout=0.1, input_size=32681):
        super(NN_FC, self).__init__()
        self.fc = nn.Sequential(*
                                [nn.Linear(input_size, size),
                                 nn.ELU(),
                                    nn.Dropout(dropout)]
                                )
        self.classifier = nn.Linear(size, output_size)

    def forward(self, fp):
        embedding = self.fc(fp)
        return self.classifier(embedding).squeeze(dim=1)
