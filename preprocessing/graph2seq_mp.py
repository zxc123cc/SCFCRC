import argparse
import os
import numpy as np
import torch
import sys

sys.path.append('../')
import data_utils
from fcf import get_filtered_feature, get_filtered_feature_with_contrastive
from utils import utility


def save_sequence(args, data):
    pass


def graph2seq(pid, args, st, ed, nids, graph_data, sequence_array, priori_array, gnn_ckpt):
    filtered_features, pseudo_labels = get_filtered_feature_with_contrastive(args, device='cuda', do_train=True,
                                                                             ckpt_file=gnn_ckpt)
    seq_loader = data_utils.GroupFeatureSequenceLoader(graph_data, fanouts=args['fanouts'],
                                                       grp_norm=args['grp_norm'],
                                                       filtered_features=filtered_features,
                                                       pseudo_labels=pseudo_labels
                                                       )
    nids = torch.from_numpy(nids)
    # 可以改成多进程
    seq_feat, priori = seq_loader.load_batch(nids, pid=pid)
    sequence_array[st:ed] = seq_feat.numpy()
    priori_array[st:ed] = priori.numpy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='graph2seq')
    parser.add_argument('--dataset', type=str, default='yelp',
                        help='Dataset name, [amazon, yelp, BF10M]')

    parser.add_argument('--train_size', type=float, default=0.4,
                        help='Train size of nodes.')
    parser.add_argument('--val_size', type=float, default=0.1,
                        help='Val size of nodes.')
    parser.add_argument('--seed', type=int, default=717,
                        help='Collecting neighbots in n hops.')

    parser.add_argument('--norm_feat', action='store_true', default=False,
                        help='Using group norm, default False')
    parser.add_argument('--grp_norm', action='store_true', default=False,
                        help='Using group norm, default False')
    parser.add_argument('--force_reload', action='store_true', default=False,
                        help='Using group norm, default False')

    parser.add_argument('--add_self_loop', action='store_true', default=False,
                        help='add self-loop to all the nodes')

    #     parser.add_argument('--n_hops', type=int, default=1,
    #                         help='Collecting neighbots in n hops.')
    parser.add_argument('--fanouts', type=int, default=[-1, -1], nargs='+',
                        help='Sampling neighbors, default [-1] means full neighbors')


    parser.add_argument('--base_dir', type=str, default='./raw_data',
                        help='Directory for loading graph data.')
    parser.add_argument('--save_dir', type=str, default='mp_output',
                        help='Directory for saving the processed sequence data.')
    parser.add_argument('--n_workers', type=int, default=1,
                        help='Using n processes.')
    # BF10M base_dir = /home/work/wangyuchen09/Projects/fraud_detection/dataset/fraud_20211209
    args = vars(parser.parse_args())
    print(args)

    utility.setup_seed(717)

    # 读取data
    graph_data = data_utils.prepare_data(args, add_self_loop=args['add_self_loop'])

    # 基本数据
    g = graph_data.graph
    n_classes = graph_data.n_classes
    feat_dim = graph_data.feat_dim
    n_relations = graph_data.n_relations
    n_groups = 2 * n_classes + 1
    n_hops = len(args['fanouts'])
    n_nodes = g.num_nodes()

    seq_len = n_relations * (n_hops * n_groups + 1 + 1)

    all_nid = g.nodes()
    file_dir = os.path.join(args['save_dir'], args['dataset'])
    os.makedirs(file_dir, exist_ok=True)

    # 聚合序列文件
    flag_1 = 'grp_norm' if args['grp_norm'] else 'no_grp_norm'
    flag_2 = 'norm_feat' if args['norm_feat'] else 'no_norm_feat'
    #     flag_3 = 'self_loop' if args['add_self_loop'] elsr 'no_self_loop'
    file_name = f"{args['dataset']}_{flag_1}_{flag_2}_{n_hops}_" \
                f"{args['train_size']}_{args['val_size']}_{args['seed']}_seed717_contrast_featmlp_ic_pc--z.npy"
    seq_file = os.path.join(file_dir, file_name)
    print(f"Saving seq_file to {seq_file}")
    sequence_array = np.memmap(seq_file, dtype=np.float32, mode='w+', shape=(n_nodes, seq_len, feat_dim))
    priori_array = np.memmap(os.path.join(file_dir, f"{args['dataset']}_priori_pc.npy"), dtype=np.float32, mode='w+',
                             shape=(n_nodes, 4))

    procs = []
    n_workers = args['n_workers']

    nids = g.nodes().numpy()
    graph2seq(0, args, 0, len(nids), nids[:len(nids)], graph_data, sequence_array, priori_array,
              gnn_ckpt=f'../logs/checkpoints/{args["dataset"]}/gnn_ic_pc.bin')

    sequence_array.flush()
    priori_array.flush()
