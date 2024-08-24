import random

import torch
import torch.nn as nn
from modules import embedding
import torch.nn.functional as F


class ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks without log operation."""

    def __init__(self, hidden_size, hidden_dropout_prob, num_labels):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.out_proj = nn.Linear(hidden_size, num_labels)

    def forward(self, x):
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


def compute_kl_loss(p, q, reduction="batchmean"):
    '''
    p/q: torch.Tensor类型，Size为：[batch_size, *], 例如：[batch_size, seq_length, hidden_size], [batch_size, hidden_size]
    reduction="batchmean" or "mean" or "sum"
    pad_mask: [batch_size, seq_length]
    attention_mask: [batch_size, seq_length]
    '''
    p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
    q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')

    if reduction == "mean":
        p_loss = p_loss.mean()
        q_loss = q_loss.mean()
    else:
        p_loss = p_loss.sum()
        q_loss = q_loss.sum()
    loss = (p_loss + q_loss) / 2

    if reduction == "batchmean":
        return loss / (p.size()[0])
    return loss


class FDMoE(nn.Module):
    def __init__(self, feat_dim, emb_dim, n_classes, n_hops, n_relations,
                 n_heads, dim_feedforward, n_layers, n_public_layers, num_experts,
                 num_gate_layer, dropout=0.1, mask_probability=0.1, agg_type='cat'):

        super(FDMoE, self).__init__()

        self.num_experts = num_experts
        self.encoder = FDMoEEncoder(feat_dim, emb_dim, n_classes, n_hops, n_relations,
                                    n_heads, dim_feedforward, n_layers,
                                    n_public_layers, num_experts, num_gate_layer,
                                    dropout, agg_type)

        proj_emb_dim = emb_dim * 2
        if agg_type == 'cat':
            global_proj_emb_dim = emb_dim * n_relations * 2
        elif agg_type == 'mean':
            global_proj_emb_dim = emb_dim * 2

        self.classifiers = nn.ModuleList([ClassificationHead(proj_emb_dim, dropout, n_classes) if _ != num_experts - 1
                                          else ClassificationHead(global_proj_emb_dim, dropout, n_classes) for _ in
                                          range(num_experts)])

        self.cross_entropy = nn.CrossEntropyLoss(reduction='none')
        self.loss_fct = torch.nn.KLDivLoss(reduction='batchmean')

        self.mask_probability = mask_probability
        # print("self.mask_probability::", self.mask_probability)
        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def get_expert_masked_prob2(self, origin_gates):
        probability_tmp = torch.full((1, self.num_experts), self.mask_probability)
        masked_tmp = torch.bernoulli(probability_tmp).bool().to(origin_gates.device)
        masked_idx = torch.where(masked_tmp == 1)[0]
        unmasked_indices = torch.ones((origin_gates.shape[0], self.num_experts)).to(origin_gates.device).bool()
        unmasked_indices[:, masked_idx] = 0
        idx1 = torch.where(unmasked_indices.sum(dim=1) == 0)[0]
        unmasked_indices[idx1, random.randint(0, self.num_experts - 1)] = True

        x = torch.ones(origin_gates.shape[0], self.num_experts).to(origin_gates.device)
        x = x * unmasked_indices

        gate_prob = torch.softmax(origin_gates, dim=-1)

        tmp = (gate_prob * (~unmasked_indices)).sum(dim=1)
        tmp = tmp / unmasked_indices.sum(dim=1)
        tmp = tmp.reshape(gate_prob.shape[0], 1)
        tmp = tmp.repeat(1, gate_prob.shape[1])
        masked_prob = (gate_prob + tmp) * x

        return masked_prob

    def forward(self, src_emb, nodes, src_mask=None, priori=None, labels=None, do_train=True, is_mask_expert=True):
        outputs, origin_gates = self.encoder(src_emb, nodes, src_mask)
        final_out = []
        for i in range(self.num_experts):
            final_out.append(self.classifiers[i](outputs[i]).unsqueeze(1))  # bsz * 1 * 2
        final_out_mat = torch.nn.functional.softmax(torch.cat(final_out, dim=1), dim=2)  # bsz * num of exp * 2
        gate_prob = torch.softmax(origin_gates, dim=-1)
        final_out_logits = torch.bmm(gate_prob.unsqueeze(1), final_out_mat).squeeze(1)
        if do_train:
            loss = []
            for logits in final_out:
                loss.append(self.cross_entropy(logits.squeeze(1), labels.view(-1)).view(-1, 1))
            if len(loss) == 1:
                loss_mat = loss[0].view(-1, 1)
            else:
                loss_mat = torch.cat(loss, dim=1)  # bsz * # of expert
            final_loss = torch.mean(torch.sum(gate_prob * loss_mat, dim=1))

            # guide loss
            if priori is not None:
                guide_loss = self.loss_fct(torch.nn.functional.log_softmax(origin_gates, dim=1), priori)
                final_loss += 0.1 * guide_loss
            if is_mask_expert:
                masked_prob_2 = self.get_expert_masked_prob2(origin_gates)

                logits2 = torch.bmm(masked_prob_2.unsqueeze(1), final_out_mat).squeeze(1)
                kl_div = compute_kl_loss(final_out_logits, logits2, reduction="batchmean")
                final_loss += 0.3 * kl_div
        else:
            final_loss = None
        for i in range(self.num_experts):
            final_out[i] = torch.nn.functional.softmax(final_out[i].squeeze(1), dim=1)
        return final_out, final_loss, final_out_logits, origin_gates


class FDMoEEncoder(nn.Module):
    def __init__(self, feat_dim, emb_dim, n_classes, n_hops, n_relations,
                 n_heads, dim_feedforward, n_layers,
                 n_public_layers, num_experts, num_gate_layer,
                 dropout=0.1, agg_type='cat', risk_feature=None):
        super(FDMoEEncoder, self).__init__()

        # encoder that provides hop, relation and group encodings
        self.feat_embedding = embedding.CustomEncoder(feat_dim=feat_dim,
                                                      emb_dim=emb_dim, n_relations=n_relations,
                                                      n_hops=n_hops, dropout=dropout,
                                                      n_classes=n_classes)

        # define transformer encoder
        self.num_experts = num_experts
        self.num_gate_layer = num_gate_layer

        encoder_layers = nn.TransformerEncoderLayer(emb_dim, n_heads, dim_feedforward, dropout)
        self.layer = nn.TransformerEncoder(encoder_layers, n_public_layers)
        self.expert_layer = nn.ModuleList(
            [nn.TransformerEncoder(encoder_layers, n_layers - n_public_layers) for _ in range(num_experts)])
        self.gate_layer = nn.TransformerEncoder(encoder_layers, self.num_gate_layer)
        if agg_type == 'cat':
            proj_emb_dim = emb_dim * n_relations * 2
        elif agg_type == 'mean':
            proj_emb_dim = emb_dim * 2
        self.gate_classifier = ClassificationHead(proj_emb_dim, dropout, num_experts)

        self.emb_dim = emb_dim
        self.n_hops = n_hops
        self.n_relations = n_relations
        self.n_classes = n_classes
        self.agg_type = agg_type

    def cross_relation_agg(self, out, return_type=0):
        device = out.device
        n_tokens = out.shape[0]

        block_len = 1 + 1 + self.n_hops * (2 * self.n_classes + 1)
        indices = torch.arange(0, n_tokens, block_len, dtype=torch.int64).to(device)
        risk_indices = indices + 1

        mr_feats = torch.index_select(out, dim=0, index=indices)
        risk_mr_feats = torch.index_select(out, dim=0, index=risk_indices)
        if return_type:
            return mr_feats, risk_mr_feats

        if self.agg_type == 'cat':
            mr_feats = torch.split(mr_feats, 1, dim=0)
            agg_feats = torch.cat(mr_feats, dim=2).squeeze()
            risk_mr_feats = torch.split(risk_mr_feats, 1, dim=0)
            risk_agg_feats = torch.cat(risk_mr_feats, dim=2).squeeze()

        elif self.agg_type == 'mean':
            agg_feats = torch.mean(mr_feats, dim=0)
            risk_agg_feats = torch.mean(risk_mr_feats, dim=0)

        return agg_feats, risk_agg_feats

    def forward(self, src_emb, nodes, src_mask=None):
        src_emb = torch.transpose(src_emb, 1, 0)

        out = self.feat_embedding(src_emb)

        out = self.layer(out, src_mask)
        out_inter = torch.clone(out)

        out = self.gate_layer(out, src_mask)
        agg_feats, risk_agg_feats = self.cross_relation_agg(out)
        out = torch.cat([agg_feats, risk_agg_feats], dim=1)
        gates = self.gate_classifier(out)

        out_list = []
        for i, expert_module in enumerate(self.expert_layer):
            now_out = expert_module(out_inter)
            if i + 1 < self.num_experts:
                mr_feats, risk_mr_feats = self.cross_relation_agg(now_out, return_type=1)
                now_out = torch.cat([mr_feats[i], risk_mr_feats[i]], dim=1)
            else:
                agg_feats, risk_agg_feats = self.cross_relation_agg(now_out)
                now_out = torch.cat([agg_feats, risk_agg_feats], dim=1)
            out_list.append(now_out)

        return out_list, gates
