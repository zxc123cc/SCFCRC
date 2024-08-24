import os
import numpy as np
import torch
from torch.utils.data import Dataset, Subset, DataLoader


class SequenceDataset(Dataset):
    def __init__(self, seq_data, labels,priori_data=None):
        super(SequenceDataset, self).__init__()
        # (N,S,E)
        self.data = seq_data
        self.labels = labels
        self.priori_data = priori_data
        self.n_samples = labels.shape[0]
        self.nodes = torch.arange(self.n_samples).cuda()

    def __getitem__(self, idx: torch.LongTensor):
        # batch_seq = torch.index_select(self.data, 1, idx)
        batch_seq = self.data[idx]
        batch_labels = self.labels[idx]

        if self.priori_data is not None and self.priori_data[idx][-1] < 0.9:
            priori = self.priori_data[idx]
        # if self.priori_data is not None:
        #     priori = self.priori_data[idx]
        else:
            priori = torch.tensor([0.2,0.2,0.2,0.4])
        batch_nodes = self.nodes[idx]
        # print(priori)
        return batch_seq, batch_labels, priori, batch_nodes

    def __len__(self):
        return self.n_samples


def split_dataset(data):
    seq_data, labels, priori_data, train_nid, val_nid, test_nid, *infos = data
    dataset = SequenceDataset(seq_data, labels, priori_data)
    train_set = Subset(dataset, train_nid)
    val_set = Subset(dataset, val_nid)
    test_set = Subset(dataset, test_nid)
    return train_set, val_set, test_set


def load_sequence_data(args):
    file_dir = os.path.join(args['base_dir'], args['dataset'])

    # 读取dataset info array
    info_name = f"{args['dataset']}_infos_" \
                f"{args['train_size']}_{args['val_size']}_{args['seed']}.npz"
    info_file = os.path.join(file_dir, info_name)
    info_data = np.load(info_file)

    # labels, train_nid, val_nid, infos=[feat_dim, n_classes, n_relations]
    labels = torch.LongTensor(info_data['label'])
    train_nid = torch.LongTensor(info_data['train_nid'])
    val_nid = torch.LongTensor(info_data['val_nid'])
    test_nid = torch.LongTensor(info_data['test_nid'])
    infos = info_data['infos']
    feat_dim, n_classes, n_relations = tuple(infos)
    n_nodes = labels.shape[0]
    seq_len = n_relations * (1 + 1 + args['n_hops'] * (2*n_classes + 1))
    # seq_len = n_relations * (1 + args['n_hops'] * (n_classes+1))

    flag_1 = 'grp_norm' if args['grp_norm'] else 'no_grp_norm'
    flag_2 = 'norm_feat' if args['norm_feat'] else 'no_norm_feat'

    file_name = f"{args['dataset']}_{flag_1}_{flag_2}_{args['n_hops']}_" \
                f"{args['train_size']}_{args['val_size']}_{args['seed']}_with_risk_seed717_add_featmlp_duibi3.npy"
    seq_file = os.path.join(file_dir, file_name)

    sequence_array = np.memmap(seq_file, dtype=np.float32, mode='r+', shape=(n_nodes, seq_len, feat_dim))

    priori_array = np.memmap(os.path.join(file_dir, 'priori_tmp.npy'), dtype=np.float32, mode='r+', shape=(n_nodes, 4))

    seq_data = torch.tensor(sequence_array)
    priori_data = torch.tensor(priori_array)

    print(f"[Global] Dataset <{args['dataset']}> Overview\n"
          f"\tEntire (postive/total) {torch.sum(labels):>6} / {labels.shape[0]:<6}\n"
          f"\tTrain  (postive/total) {torch.sum(labels[train_nid]):>6} / {labels[train_nid].shape[0]:<6}\n"
          f"\tValid  (postive/total) {torch.sum(labels[val_nid]):>6} / {labels[val_nid].shape[0]:<6}\n"
          f"\tTest   (postive/total) {torch.sum(labels[test_nid]):>6} / {labels[test_nid].shape[0]:<6}\n")

    return seq_data, labels,priori_data, train_nid, val_nid, test_nid, feat_dim, n_classes, n_relations
