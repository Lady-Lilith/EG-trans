
import torch
from sklearn.model_selection import KFold


from torch import nn
import torch.nn.functional as F

import load_data
from param import parameter_parser


def train(model, train_data, optimizer, opt, scheduler,):
    model.train()
    kf = KFold(n_splits=5, shuffle=True)
    args = parameter_parser()

    for epoch in range(0, opt.epoch):
        model.zero_grad()
        dataset, cd_pairs,cc_edge_index,dd_edge_index,one_index,cd_edge_index,dis,cis= load_data.load_dataset(args)
        crna_di_graph= load_data.dgl_heterograph(dis,cis,one_index, args)
        score, x, y = model(train_data,crna_di_graph)
        # loss = SigmoidLoss()
        loss = torch.nn.BCEWithLogitsLoss(reduction='mean')
        ## BCEWithLogitsLoss is a loss function suitable for binary classification,
        # which requires two-dimensional vectors (logits, labels).
        # It combines Sigmoid and binary cross-entropy loss calculation,
        # so the loss can be directly calculated (without going through the output of the activation function).
        loss = loss(score, train_data['c_d'])
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()
        print(loss.item())
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch + 1}/{opt.epoch}, Learning Rate: {current_lr:.6f}")
    score = score.detach().cpu().numpy()
    print('epoch:', epoch)
    return model



