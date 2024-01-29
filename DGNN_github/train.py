import time
import torch
import os
import scipy.sparse as sp
import numpy as np
import torch.nn.functional as F
import argparse
from torch_sparse import SparseTensor
from hps import get_hyper_param
from model.dgnn import DGNN
from util import load_dataset, root, get_mask, get_accuracy, set_seed
from knn_build import load_graph,generate_knn
from util import to_normalized_sparsetensor

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="citeseer")
parser.add_argument("--k", type=int, default=10, help="k of KNN graph.")
args = parser.parse_args()
set_seed(0xC0FFEE)
#set_seed(3407)
epochs = 1000
patience = 25
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# torch.cuda.set_device(2)
checkpoint_path = root + "/checkpoint"
feat, label, n, nfeat, nclass, adj = load_dataset(args.dataset, norm=True, device=device)
hp = get_hyper_param(args.dataset)

if not os.path.exists("../data/KNN/" + args.dataset + "/c" + str(args.k) + ".txt"):
    generate_knn(args.dataset, feat, args.k)
adj1 = load_graph(args.dataset,args.k,n)

row, col = np.argwhere(adj1 > 0)
adj1 = adj1 + adj1.T.multiply(adj1.T > adj1) - adj1.multiply(adj1.T > adj1)
adj1 = adj1 + torch.eye(adj1.shape[0], adj1.shape[0])
edge_index = np.argwhere(adj1 > 0)
adj1 = to_normalized_sparsetensor(edge_index, adj1.shape[0])


def train(model, optimizer, train_mask):
    model.train()
    optimizer.zero_grad()
    result = model(feat=feat, adj=adj,adj1=adj1)
    loss = F.nll_loss(result[train_mask], label[train_mask])
    loss.backward()
    optimizer.step()
    return get_accuracy(result[train_mask], label[train_mask]), loss.item()


def test(model, test_mask):
    model.eval()
    with torch.no_grad():
        result = model(feat=feat, adj=adj, adj1=adj1)
        return get_accuracy(result[test_mask], label[test_mask])


def validate(model, val_mask) -> float:
    model.eval()
    with torch.no_grad():
        result = model(feat=feat, adj=adj,adj1=adj1)
        loss = F.nll_loss(result[val_mask], label[val_mask].to(device))
        return get_accuracy(result[val_mask], label[val_mask]), loss.item()


def run():
    train_mask, val_mask, test_mask = get_mask(label, 0.6, 0.2, device=device)
    model = DGNN(
        n=n,
        nclass=nclass,
        nfeat=nfeat,
        nlayer=hp["layer"],
        lambda_1=hp["lambda_1"],
        lambda_2=hp["lambda_2"],
        lambda_3=hp["lambda_3"],
        dropout=hp["dropout"],
        epsilon=hp["epsilon"]
    ).to(device)
    optimizer = torch.optim.Adam(
        [
            {'params': model.params2, 'weight_decay': hp["wd2"]},
            {'params': model.params23, 'weight_decay': hp["wdy"]},
        ],
        lr=hp["lr"]
    )
    checkpoint_file = "{}/{}-{}.pt".format(checkpoint_path, model.__class__.__name__, args.dataset)
    tolerate = 0
    best_loss = 100
    for epoch in range(epochs):
        if tolerate >= patience:
            break
        train_acc, train_loss = train(model, optimizer, train_mask)
        val_acc, val_loss = validate(model, val_mask)
        if train_loss < best_loss:
            tolerate = 0
            best_loss = train_loss
        else:
            tolerate += 1
        message = "Epoch={:<4} | Tolerate={:<3} | Train_acc={:.4f} | Train_loss={:.4f} | Val_acc={:.4f} | Val_loss={:.4f}".format(
            epoch,
            tolerate,
            train_acc,
            train_loss,
            val_acc,
            val_loss,
        )
        print(message)
    test_acc = test(model, test_mask)
    torch.save(model.state_dict(), checkpoint_file)
    print("Validate accuracy {:.4f}.".format(test_acc))
    return test_acc

if __name__ == '__main__':
    run()
