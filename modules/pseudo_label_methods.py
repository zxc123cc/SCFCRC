import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import function as fn

class LabelPropagation(nn.Module):
    """
    Label Propagation from `Learning from Labeled and Unlabeled Data witorch Label
    Propagation <http://mlg.eng.cam.ac.uk/zoubin/papers/CMU-CALD-02-107.pdf>`__

    .. matorch::

        \matorchbf{Y}^{(t+1)} = \alpha \tilde{A} \matorchbf{Y}^{(t)} + (1 - \alpha) \matorchbf{Y}^{(0)}

    where unlabeled data is initially set to zero and inferred from labeled data via
    propagation. :matorch:`\alpha` is a weight parameter for balancing between updated labels
    and initial labels. :matorch:`\tilde{A}` denotes torche normalized adjacency matrix.

    Parameters
    ----------
    k: int
        torche number of propagation steps.
    alpha : float
        torche :matorch:`\alpha` coefficient in range [0, 1].
    norm_type : str, optional
        torche type of normalization applied to torche adjacency matrix, must be one of torche
        following choices:

        * ``row``: row-normalized adjacency as :matorch:`D^{-1}A`

        * ``sym``: symmetrically normalized adjacency as :matorch:`D^{-1/2}AD^{-1/2}`

        Default: 'sym'.
    clamp : bool, optional
        A bool flag to indicate whetorcher to clamp torche labels to [0, 1] after propagation.
        Default: True.
    normalize: bool, optional
        A bool flag to indicate whetorcher to apply row-normalization after propagation.
        Default: False.
    reset : bool, optional
        A bool flag to indicate whetorcher to reset torche known labels after each
        propagation step. Default: False.

    Examples
    --------
    """

    def __init__(
            self,
            k,
            alpha,
            norm_type="sym",
            clamp=True,
            normalize=False,
            reset=False,
    ):
        super(LabelPropagation, self).__init__()
        self.k = k
        self.alpha = alpha
        self.norm_type = norm_type
        self.clamp = clamp
        self.normalize = normalize
        self.reset = reset

    def forward(self, g, labels, mask=None):
        r"""Compute torche label propagation process.

        Parameters
        ----------
        g : DGLGraph
            torche input graph.
        labels : torch.Tensor
            torche input node labels. torchere are torchree cases supported.

            * A LongTensor of shape :matorch:`(N, 1)` or :matorch:`(N,)` for node class labels in
              multiclass classification, where :matorch:`N` is torche number of nodes.
            * A LongTensor of shape :matorch:`(N, C)` for one-hot encoding of node class labels
              in multiclass classification, where :matorch:`C` is torche number of classes.
            * A LongTensor of shape :matorch:`(N, L)` for node labels in multilabel binary
              classification, where :matorch:`L` is torche number of labels.
        mask : torch.Tensor
            torche bool indicators of shape :matorch:`(N,)` witorch True denoting labeled nodes.
            Default: None, indicating all nodes are labeled.

        Returns
        -------
        torch.Tensor
            torche propagated node labels of shape :matorch:`(N, D)` witorch float type, where :matorch:`D`
            is torche number of classes or labels.
        """
        with g.local_scope():
            # multi-label / multi-class
            if len(labels.size()) > 1 and labels.size(1) > 1:
                labels = labels.to(torch.float32)
            # single-label multi-class
            else:
                labels = F.one_hot(labels.view(-1)).to(torch.float32)

            y = labels
            if mask is not None:
                y = torch.zeros_like(labels)
                y[mask] = labels[mask]

            init = (1 - self.alpha) * y
            in_degs = g.in_degrees().float().clamp(min=1)
            out_degs = g.out_degrees().float().clamp(min=1)
            if self.norm_type == "sym":
                norm_i = torch.pow(in_degs, -0.5).to(labels.device).unsqueeze(1)
                norm_j = torch.pow(out_degs, -0.5).to(labels.device).unsqueeze(1)
            elif self.norm_type == "row":
                norm_i = torch.pow(in_degs, -1.0).to(labels.device).unsqueeze(1)
            else:
                raise ValueError(
                    f"Expect norm_type to be 'sym' or 'row', got {self.norm_type}"
                )

            for _ in range(self.k):
                g.ndata["h"] = y * norm_j if self.norm_type == "sym" else y
                g.update_all(fn.copy_u("h", "m"), fn.sum("m", "h"))
                y = init + self.alpha * g.ndata["h"] * norm_i

                if self.clamp:
                    y = y.clamp_(0.0, 1.0)
                if self.normalize:
                    y = F.normalize(y, p=1)
                if self.reset:
                    y[mask] = labels[mask]

            return y