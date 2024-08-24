import torch.nn as nn
import dgl

class GNN(nn.Module):
    def __init__(self, in_dim, out_dim, features, num_layers, num_classes):
        super(GNN, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_classes = num_classes
        self.features = features
        self.num_layers = num_layers

        self.conv_list = nn.ModuleList()
        self.conv_list.append(dgl.nn.GraphConv(self.in_dim, self.out_dim))
        for i in range(self.num_layers-1):
            self.conv_list.append(dgl.nn.GraphConv(self.out_dim, self.out_dim))

        self.mlp = nn.Linear(self.out_dim, num_classes)

    def forward(self, graph, nodes):
        feat = self.features(nodes)
        if isinstance(graph, list):
            for layer, block in zip(self.conv_list, graph):
                feat = layer(block, feat)
        else:
            feat = self.conv_list[0](graph, feat)

        logits = self.mlp(feat)

        return feat, logits