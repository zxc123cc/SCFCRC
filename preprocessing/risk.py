import torch
import torch.nn as nn
from tqdm import tqdm
import sys
sys.path.append('..')
from modules import pseudo_label_methods,graph_models
import dgl
from dgl.dataloading import MultiLayerFullNeighborSampler, NodeDataLoader
import data_utils
from torch.nn import functional as F
from utils import metrics

def get_risk_feature(args, device='cpu',do_train=True,ckpt_file=''):
    graph_data = data_utils.load_graph(dataset_name=args['dataset'], raw_dir='./raw_data/',
                                       train_size=args['train_size'], val_size=args['val_size'],
                                       seed=args['seed'], norm=args['norm_feat'],
                                       force_reload=args['force_reload'])

    graph = graph_data
    num_classes = 2

    # retrieve labels of ground truth
    labels = graph.ndata["label"].to(device)

    # Extract node features
    feat = graph.ndata["feature"].to(device)

    # retrieve masks for train/validation/test
    train_mask = graph.ndata["train_mask"]
    train_idx = torch.nonzero(train_mask, as_tuple=False).squeeze(1).to(device)
    graph = graph.to(device)

    # 标签传播
    g_h = dgl.to_homogeneous(graph)
    g_h = dgl.add_self_loop(g_h)
    lp = pseudo_label_methods.LabelPropagation(10, 0.5, reset=True)
    pseudo_labels = lp(g_h, labels, mask=train_idx).argmax(dim=1)

    # print(len(labels))
    # print(torch.sum(train_mask))
    # print(torch.sum(labels[val_idx] == presudo_labels[val_idx]).item() / (len(val_idx)))
    # exit()

    # 特征无关密集层
    features = nn.Embedding(feat.shape[0], feat.shape[1])
    features.weight = nn.Parameter(feat, requires_grad=False)
    # MLP = MLP_(features=features, input_dim=feat.shape[1], output_dim=args.embed_dim)

    gh_sampler = MultiLayerFullNeighborSampler(1)
    #  NodeDataLoader可以以小批次的形式对一个节点的集合进行迭代。
    gh_dataloader = NodeDataLoader(g_h, g_h.nodes(), gh_sampler, device=device,
                                   use_ddp=False, batch_size=512, shuffle=True, drop_last=False,
                                   num_workers=0)

    out_dim = feat.shape[1]
    # 全局风险
    simple_gnn = graph_models.GNN(in_dim=feat.shape[1], out_dim=out_dim,
                                 features=features, num_classes=num_classes,
                                 num_layers=1)
    simple_gnn = simple_gnn.to(device)
    if do_train:
        _, cnt = torch.unique(labels, return_counts=True)
        simple_gnn_loss = nn.CrossEntropyLoss(weight=1 / cnt)
        simple_gnn_opt = torch.optim.Adam(simple_gnn.parameters(), lr=0.005, weight_decay=1e-3)
        best_score = 0
        for epoch in range(100):
            simple_gnn.train()
            for i, (input_nodes, output_nodes, blocks) in enumerate(tqdm(gh_dataloader, desc=f"epoch: {epoch}")):
                batch_labels = pseudo_labels[output_nodes]
                blocks = [block.to(device) for block in blocks]
                _, logit = simple_gnn(blocks, input_nodes)
                loss = simple_gnn_loss(logit, batch_labels)
                simple_gnn_opt.zero_grad()
                loss.backward()
                simple_gnn_opt.step()
            # val
            simple_gnn.eval()
            prob_list = []
            label_list = []
            for i, (input_nodes, output_nodes, blocks) in enumerate(tqdm(gh_dataloader)):
                blocks = [block.to(device) for block in blocks]
                feat, logits = simple_gnn(blocks, input_nodes)
                # preds = torch.argmax(logits,dim=-1)
                batch_labels = pseudo_labels[output_nodes]
                prob_list.append(logits.cpu())
                label_list.append(batch_labels.cpu())
            probs = torch.cat(prob_list, dim=0)
            eval_labels = torch.cat(label_list, dim=0)
            te_true, te_prob, te_pred = metrics.convert_probs(eval_labels, probs, threshold_moving=False, thres=0.5)
            results = metrics.eval_model(te_true, te_prob, te_pred)
            if results.f1_macro > best_score:
                best_score = results.f1_macro
                state_dict = simple_gnn.state_dict()
                torch.save(state_dict, ckpt_file)
    else:
        simple_gnn.load_state_dict(torch.load(ckpt_file, map_location='cpu'))
    risk_feat = torch.zeros(len(train_mask), out_dim).float().to(device)
    print(best_score)
    # gh_dataloader = NodeDataLoader(g_h, g_h.nodes(), gh_sampler, device=device,
    #                                use_ddp=False, batch_size=512, shuffle=True, drop_last=False,
    #                                num_workers=0)
    simple_gnn.eval()

    # pseudo_labels = torch.zeros_like(labels)
    for i, (input_nodes, output_nodes, blocks) in enumerate(tqdm(gh_dataloader)):
        blocks = [block.to(device) for block in blocks]
        feat, logits = simple_gnn(blocks, input_nodes)
        # preds = torch.argmax(logits,dim=-1)
        # print(preds.sum(dim=0))
        # pseudo_labels[output_nodes] = preds
        risk_feat[output_nodes] = feat.detach()
    # pseudo_labels[train_idx] = labels[train_idx]

    # risk_features = nn.Embedding(risk_feat.shape[0], risk_feat.shape[1])
    # risk_features.weight = nn.Parameter(risk_feat, requires_grad=False)
    risk_features = risk_feat.cpu()
    return risk_features,pseudo_labels


def get_page_rank_risk_feature(args, device='cpu',do_train=True,ckpt_file=''):
    graph_data = data_utils.load_graph(dataset_name=args['dataset'], raw_dir='./raw_data/',
                                       train_size=args['train_size'], val_size=args['val_size'],
                                       seed=args['seed'], norm=args['norm_feat'],
                                       force_reload=args['force_reload'])

    graph = graph_data
    num_classes = 2

    # retrieve labels of ground truth
    labels = graph.ndata["label"].to(device)

    # Extract node features
    feat = graph.ndata["feature"].to(device)

    # retrieve masks for train/validation/test
    train_mask = graph.ndata["train_mask"]
    train_idx = torch.nonzero(train_mask, as_tuple=False).squeeze(1).to(device)
    print(torch.sum(labels[train_idx]))
    graph = graph.to(device)

    # 标签传播
    g_h = dgl.to_homogeneous(graph)
    g_h = dgl.remove_self_loop(g_h)
    g_h = dgl.add_self_loop(g_h)

    labels_1 = F.one_hot(labels.view(-1)).to(torch.float32)
    y = torch.full_like(labels_1, 0.2)
    y[:, 0] = 0
    y[train_idx] = labels_1[train_idx]

    g_h.ndata['init_risk'] = y.to(device)
    trans = pseudo_label_methods.SIGNDiffusion(k=10, in_feat_name='init_risk', out_feat_name='risk_feat', alpha=0.15)
    trans(g_h)
    pseudo_labels = g_h.ndata['risk_feat_2'].argmax(dim=1)
    pseudo_labels[train_idx] = labels[train_idx]

    # 特征无关密集层
    features = nn.Embedding(feat.shape[0], feat.shape[1])
    features.weight = nn.Parameter(feat, requires_grad=False)
    # MLP = MLP_(features=features, input_dim=feat.shape[1], output_dim=args.embed_dim)

    gh_sampler = MultiLayerFullNeighborSampler(1)
    #  NodeDataLoader可以以小批次的形式对一个节点的集合进行迭代。
    gh_dataloader = NodeDataLoader(g_h, g_h.nodes(), gh_sampler, device=device,
                                   use_ddp=False, batch_size=512, shuffle=True, drop_last=False,
                                   num_workers=0)

    # 全局风险
    simple_gnn = graph_models.GNN(in_dim=feat.shape[1], out_dim=32,
                                 features=features, num_classes=num_classes,
                                 num_layers=1)
    simple_gnn = simple_gnn.to(device)
    if do_train:
        _, cnt = torch.unique(labels, return_counts=True)
        simple_gnn_loss = nn.CrossEntropyLoss(weight=1 / cnt)
        simple_gnn_opt = torch.optim.Adam(simple_gnn.parameters(), lr=0.005, weight_decay=1e-3)

        for epoch in range(100):
            simple_gnn.train()
            for i, (input_nodes, output_nodes, blocks) in enumerate(tqdm(gh_dataloader, desc=f"epoch: {epoch}")):
                batch_labels = pseudo_labels[output_nodes]
                blocks = [block.to(device) for block in blocks]
                _, logit = simple_gnn(blocks, input_nodes)
                loss = simple_gnn_loss(logit, batch_labels)
                simple_gnn_opt.zero_grad()
                loss.backward()
                simple_gnn_opt.step()

        state_dict = simple_gnn.state_dict()
        torch.save(state_dict, ckpt_file)
    else:
        simple_gnn.load_state_dict(torch.load(ckpt_file, map_location='cpu'))
    risk_feat = torch.zeros(len(train_mask), 32).float().to(device)

    simple_gnn.eval()
    for i, (input_nodes, output_nodes, blocks) in enumerate(tqdm(gh_dataloader)):
        blocks = [block.to(device) for block in blocks]
        feat, logits = simple_gnn(blocks, input_nodes)
        risk_feat[output_nodes] = feat.detach()
    risk_features = risk_feat.cpu()
    return risk_features,pseudo_labels