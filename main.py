import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import sequence
from modules import model_moe
from utils import utility, metrics, earlystopping, log_tools, plot_tools

from transformers import get_scheduler


def prepare_scheduler(optimizer, epochs, steps_per_epoch, warmup_rate, gradient_accumulation_steps=1):
    total_steps = (epochs * steps_per_epoch) / gradient_accumulation_steps
    warmup_steps = int(total_steps * warmup_rate)
    scheduler = get_scheduler(
        name='linear',
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps)
    return scheduler


# f_list = [open(f'./preprocessing/draw/manager_loss{i}.txt','w',encoding='utf-8') for i in range(4)]
def evaluation(args, model, eval_loader, threshold_moving=False, thres=0.5, device='cpu'):
    r"""Evaluate model.
    Parameters
    ----------
    args : dict
        Train configurations. (useless)
    model : nn.Module
        Trained model.
    eval_loader : DataLoader
        Dataloader for evaluation data.
    threshold_moving : bool
        Whether to use threshold-moving strategy. Default=True.
    thres : float
        The selected threshold to convert probabilities to crisp class labels.
    device : str
        Evalutaion on cpu or gpu. Default='cpu'.

    Return : tuple
        Returns (y_true, y_prob, y_pred), namely the ground truth labels,
        the probabilities and predicted labels.

    """
    model.to(device)
    model.eval()
    prob_list = []
    label_list = []
    with torch.no_grad():
        for (batch_seq, batch_labels, priori, batch_nodes) in tqdm(eval_loader):
            batch_seq = batch_seq.to(device)
            batch_labels = batch_labels.to(device)
            batch_nodes = batch_nodes.to(device)
            logits, loss, final_out_logits, origin_gates = model(batch_seq, batch_nodes, do_train=False)
            prob_list.append(final_out_logits.cpu())
            label_list.append(batch_labels.cpu())

    probs = torch.cat(prob_list, dim=0)

    eval_labels = torch.cat(label_list, dim=0)
    return metrics.convert_probs(eval_labels, probs, threshold_moving=threshold_moving, thres=thres)


def train(args, data, dataset, run_id):
    """Training process.
    Parameters
    ----------
    args : dict
        Train configurations.
    data : tuple
        Reference <data.sequence.load_sequence_data>.
    dataset : tuple
        It contains train/val/test dataset. Reference <data.sequence.SequenceDataset>
    run_id : int
        The execution id in multiple runs.

    Return : list
        Model performance. Listed as follows:
        F1-macro, 
        AUC, 
        GMean, 
        Precision of positive samples, 
        Recall of positive samples,
        Average Precision (aka Area Under Precision-Recall Curve), 
        F1-score of positive samples, 
        F1-score of negative samples, 
        Recall-macro, 
        best validation epoch.
    """
    # setup devices for training and evaluation
    if args['gpu'] < 0:
        device = 'cpu'
    else:
        device = f"cuda:{args['gpu']}"

    # evaluation device
    eval_device = 'cpu' if args['cpu_eval'] else device

    # dataset
    *useless, feat_dim, n_classes, n_relations = data
    train_set, val_set, test_set = dataset
    val_set = test_set

    # dataloaders
    train_loader = DataLoader(train_set, batch_size=args['batch_size'], shuffle=True,
                              drop_last=False, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=args['batch_size'], shuffle=False,
                            drop_last=False, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=args['batch_size'], shuffle=False,
                             drop_last=False, num_workers=0)

    # model
    model = model_moe.FDMoE(feat_dim=feat_dim, emb_dim=args['emb_dim'],
                            n_classes=n_classes, n_hops=args['n_hops'],
                            n_relations=n_relations, dim_feedforward=args['ff_dim'],
                            n_layers=args['n_layers'], n_heads=args['n_heads'],
                            dropout=args['dropout'], n_public_layers=args['public_layers'],
                            num_experts=args['num_experts'], num_gate_layer=args['gate_layer'],
                            agg_type='mean', mask_probability=args['mask_probability']
                            )

    model.to(device)

    # loss fuction and optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])

    # log tools
    summary = log_tools.SummaryBox(task_name=f"{args['dataset']}", flush_secs=args['flush_seconds'],
                                   log_dir=args['log_dir'])

    summary.save_config(args)

    # setup earlystopper to save best validation model
    if args['early_stop'] > 0:
        stopper = earlystopping.EarlyStopper(patience=args['early_stop'],
                                             dataset_name=args['dataset'],
                                             start_wall_time=summary.start_wall_time,
                                             log_dir=args['log_dir'])

    # record elapsed time per epoch
    timer = log_tools.Timer(task_name=f"Train on {args['dataset']}")

    is_mask_expert = False
    ratio = 0.4

    for epoch in range(args['max_epochs']):
        print(f"Train on epoch {epoch:>4d}:")
        model.train()
        timer.start()

        # average train loss
        total_loss = 0.0
        if (epoch + 1) / args['max_epochs'] > ratio:
            is_mask_expert = True
        for step, (batch_seq, batch_labels, priori, batch_nodes) in enumerate(tqdm(train_loader)):
            batch_seq = batch_seq.to(device)
            batch_labels = batch_labels.to(device)
            priori = priori.to(device)
            batch_nodes = batch_nodes.to(device)
            logits, loss, final_out_logits, origin_gates = model(batch_seq, batch_nodes, priori=priori,
                                                                 labels=batch_labels, is_mask_expert=is_mask_expert)

            total_loss += loss

            # log train loss per step
            summary.update_loss(loss, global_step=epoch *
                                                  len(train_loader) + step)

            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        timer.end()

        if epoch % args['eval_interval'] == 0:
            print(f"AVG. loss={total_loss / len(train_loader): 3.4f}, "
                  f"Elapsed time={timer.avg_time:.2f}(s)")
            val_true, val_prob, val_pred = evaluation(
                args, model, val_loader, device=eval_device)
            results = metrics.eval_model(val_true, val_prob, val_pred)

            # log evaluation results
            summary.update_metrics(results, global_step=epoch)
            if args['early_stop'] > 0:
                if stopper.step(results.f1_macro, epoch, model):
                    break

    summary.close()

    print("\nBest Epoch {}, Val {:.4f}".format(
        stopper.best_ep, stopper.best_score))

    if args['early_stop']:
        stopper.load_checkpoint(model)
        val_true, val_prob, val_pred = evaluation(
            args, model, val_loader, device=eval_device)
        val_results = metrics.eval_model(val_true, val_prob, val_pred)

        # log ROC and PRC
        summary.add_figure(figure=plot_tools.plot_roc_curve(val_true, val_prob),
                           fig_name=f"ROC-AUC Curve ({args['dataset']}_{run_id})")
        summary.add_figure(figure=plot_tools.plot_pr_curve(val_true, val_prob),
                           fig_name=f"PR Curve ({args['dataset']}_{run_id})")

        # report results on best threshold on Precision Recall Curve
        print(f"best_roc_thres: {val_results.best_roc_thres} \n"
              f"best_pr_thres: {val_results.best_pr_thres}")
        te_true, te_prob, te_pred = evaluation(args, model, test_loader,
                                               thres=val_results.best_pr_thres,
                                               device=eval_device)
        results = metrics.eval_model(te_true, te_prob, te_pred)

    return [results.f1_macro, results.auc_gnn, results.gmean_gnn,
            results.precision_1, results.recall_1, results.ap_gnn,
            results.f1_binary_1, results.f1_binary_0, results.recall_macro,
            stopper.best_ep]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GAGA")

    parser.add_argument('--config', type=str, default='configs/yelpchi_paper.json',
                        help='Path to training config.')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='<dir> to store train logs.')
    parser.add_argument('--early_stop', type=int, default=100,
                        help='The patience when using early stop.\n'
                             'Default: 30, 0 disables earlystopper.')
    parser.add_argument('--gpu', type=int, default=0,
                        help='Which gpu to use 0/1/..., -1 using only cpu')
    parser.add_argument('--n_workers', type=int, default=0,
                        help='Number of extra processes for dataloader.')
    parser.add_argument('--n_runs', type=int, default=1,
                        help='Repeat the training n times.')

    args = vars(parser.parse_args())
    train_config = utility.load_config(args['config'])
    args = utility.setup_args(args, train_config)

    utility.setup_seed(42)  # 0.8540
    data = sequence.load_sequence_data(args)
    dataset = sequence.split_dataset(data)

    # multiple runs
    result_list = []
    for i in range(args['n_runs']):
        res = train(args, data, dataset, i)
        result_list.append(res)

    # calculate average value and standard deviation
    print(result_list)
    metrics.calc_mean_sd(result_list)
