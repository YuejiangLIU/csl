import fire
import numpy as np
import torch
import tqdm
import os
import json

from data import get_imagenet, get_domainnet
from models import alexnet, resnet50_dino, vitb8_dino, vits14_dino, vitb14_dino, probe, alexnet1
from torch import nn
from helper import save_embedding, load_embedding, save_result, load_weak_embedding, load_weak_domain_embedding, save_weak_embedding, load_strong_embedding, save_strong_embedding

import pdb


def get_model(name, path=None, num_outputs=1000):
    if name == "alexnet":
        model = alexnet(path, num_outputs)
    elif name == "alexnet1":
        model = alexnet1(path, num_outputs)
    elif name == "resnet50_dino":
        model = resnet50_dino()
    elif name == "vitb8_dino":
        model = vitb8_dino()
    elif name == "vits14_dino":
        model = vits14_dino()
    elif name == "vitb14_dino":
        model = vitb14_dino()
    else:
        raise ValueError(f"Unknown model {name}")
    model.cuda()
    model.eval()
    model = nn.DataParallel(model)
    return model


def get_embeddings(model, loader):
    all_embeddings, all_y, all_probs = [], [], []

    for x, y in tqdm.tqdm(loader):
        output = model(x.cuda())
        if len(output) == 2:
            embeddings, logits = output
            probs = torch.nn.functional.softmax(logits, dim=-1).detach().cpu()
            all_probs.append(probs)
        else:
            embeddings = output

        all_embeddings.append(embeddings.detach().cpu())
        all_y.append(y)

    all_embeddings = torch.cat(all_embeddings, axis=0)
    all_y = torch.cat(all_y, axis=0)
    if len(all_probs) > 0:
        all_probs = torch.cat(all_probs, axis=0)
        acc = (torch.argmax(all_probs, dim=1) == all_y).float().mean()
    else:
        all_probs = None
        acc = None
    return all_embeddings, all_y, all_probs, acc


def train_logreg(
    x_train,
    y_train,
    eval_datasets,
    n_epochs=10,
    weight_decay=0.0,
    lr=1.0e-3,
    batch_size=100,
    n_classes=1000,
    ckpt_path=None,
    save_every=0,
    lw=None,
    model=None,
    verbose=True,
):
    x_train = x_train.float()
    train_ds = torch.utils.data.TensorDataset(x_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_ds, shuffle=True, batch_size=batch_size, num_workers=min(batch_size//16, 8)) # <-- add num_workers=min(batch_size//16, 8)

    d = x_train.shape[1]
    if model is None:
        model = probe(d, n_classes)
        if verbose: print('Initialize model')
    else:
        if verbose: print('Warm-start model')

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=weight_decay, lr=lr)
    n_batches = len(train_loader)
    n_iter = n_batches * n_epochs
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=n_iter)

    results = {f"{key}_all": [] for key in eval_datasets.keys()}
    results["train_all"] = []

    # if lw is not None:
    #     nsample = 1000
    #     ntotal = y_train.shape[0]
    #     order = torch.argsort(lw)
    #     x_sorted = x_train[order]
    #     y_sorted = y_train[order]
    #     x_train_clean = x_sorted[:nsample]
    #     y_train_clean = y_sorted[:nsample].argmax(dim=1)
    #     x_train_noise = x_sorted[-nsample:]
    #     y_train_noise = y_sorted[-nsample:].argmax(dim=1)

    #     half = int(nsample/2)
    #     low = int(ntotal/10*4)
    #     x_train_low = x_sorted[low-half:low+half]
    #     y_train_low = y_sorted[low-half:low+half].argmax(dim=1)
    #     high = int(ntotal/10*6)
    #     x_train_high = x_sorted[high-half:high+half]
    #     y_train_high = y_sorted[high-half:high+half].argmax(dim=1)

    #     results["train_clean"] = []
    #     results["train_noise"] = []
    #     results["train_low"] = []
    #     results["train_high"] = []

    if verbose:
        pbar = tqdm.tqdm(range(n_epochs), desc="Epoch 0")
    else:
        pbar = range(n_epochs)
    for epoch in pbar:
        correct, total = 0, 0
        for x, y in train_loader:
            x, y = x.cuda(), y.cuda()
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            schedule.step()
            if len(y.shape) > 1:
                y = torch.argmax(y, dim=1)
            correct += (torch.argmax(pred, -1) == y).detach().float().sum().item()
            total += len(y)
        if verbose:
            pbar.set_description(f"Epoch {epoch}, Train Acc {correct / total:.3f}")
        results["train_all"].append(correct / total)

        for key, (x_test, y_test) in eval_datasets.items():
            x_test = x_test.float().cuda()
            pred = torch.argmax(model(x_test), axis=-1).detach().cpu()
            acc = (pred == y_test).float().mean()
            results[f"{key}_all"].append(acc)

        # if lw is not None:
        #     pred = torch.argmax(model(x_train_clean.float().cuda()), axis=-1).detach().cpu()
        #     acc_clean = (pred == y_train_clean).float().mean()
        #     results["train_clean"].append(acc_clean)

        #     pred = torch.argmax(model(x_train_noise.float().cuda()), axis=-1).detach().cpu()
        #     acc_noise = (pred == y_train_noise).float().mean()
        #     results["train_noise"].append(acc_noise)

        #     pred = torch.argmax(model(x_train_low.float().cuda()), axis=-1).detach().cpu()
        #     acc_noise = (pred == y_train_low).float().mean()
        #     results["train_low"].append(acc_noise)

        #     pred = torch.argmax(model(x_train_high.float().cuda()), axis=-1).detach().cpu()
        #     acc_noise = (pred == y_train_high).float().mean()
        #     results["train_high"].append(acc_noise)

    for key in eval_datasets.keys():
        results[key] = results[f"{key}_all"][-1]
    return results


def main(
    batch_size: int = 128,
    soft_teacher: bool = True,
    weak_model_name: str = "alexnet",
    strong_model_name: str = "resnet50_dino",
    n_train: int = 40000,
    seed: int = 0,
    data_name: str = "imagenet",   # [imagenet, domainnet]
    data_path: str = "/root/",
    embed_path: str = "embedding/",
    result_path: str = "result/",
    ckpt_path: str = "ckpt/",
    weak_path: str = "",
    save_every: int = 0,
    n_epochs: int = 10,
    lr: float = 1e-3,
):
    if data_name == "imagenet":
        _, loader = get_imagenet(data_path, split="val", batch_size=batch_size, shuffle=False)
        num_classes = 1000
    elif data_name == "domainnet":
        _, loader = get_domainnet(data_path, split="val", batch_size=batch_size, shuffle=False)
        num_classes = 345
    else:
        raise NotImplementedError

    weak_model = get_model(weak_model_name, weak_path, num_classes)
    category = weak_path.split('/')[-2]
    stage = 'epoch_' + '_'.join(weak_path.split('/')[-1].split('.')[0].split('-')[1:])
    fname = os.path.join(embed_path, f'data_{weak_model_name}_{category}_{stage}.pkl')
    if os.path.exists(fname):
        if data_name == "domainnet":
            gt_labels, weak_labels, domain_labels, weak_acc = load_weak_domain_embedding(fname)
        else:
            gt_labels, weak_labels, weak_acc = load_weak_embedding(fname)
    else:
        print(f"Cannot load weak embeddings from {fname}, generating them from weak model")
        _, gt_labels, weak_labels, weak_acc = get_embeddings(weak_model, loader)
        save_weak_embedding(gt_labels, weak_labels, weak_acc, fname)
    if not soft_teacher:
        weak_labels = nn.functional.one_hot(torch.argmax(weak_labels, dim=1), num_classes=num_classes).float()
        print('Convert teacher outputs to hard class labels')
    print(f"Weak label accuracy: {weak_acc:.3f}")

    strong_model = get_model(strong_model_name, num_outputs=num_classes)
    fname = os.path.join(embed_path, f'data_{strong_model_name}.pkl')
    if os.path.exists(fname):
        embeddings, strong_gt_labels = load_strong_embedding(fname)
    else:
        embeddings, strong_gt_labels, _, _ = get_embeddings(strong_model, loader)
        save_strong_embedding(embeddings, strong_gt_labels, fname)

    assert torch.all(gt_labels == strong_gt_labels)
    del strong_gt_labels

    order = np.arange(len(embeddings))
    rng = np.random.default_rng(seed)
    rng.shuffle(order)
    x = embeddings[order]
    y = gt_labels[order]
    yw = weak_labels[order]
    x_train, x_test = x[:n_train], x[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]
    yw_train, yw_test = yw[:n_train], yw[n_train:]
    yw_test = torch.argmax(yw_test, dim=1)
    eval_datasets = {"test": (x_test, y_test), "test_weak": (x_test, yw_test)}
    print("# examples: ", x_train.shape[0], x_test.shape[0])

    print("Training logreg on weak labels...")
    results_weak = train_logreg(x_train, yw_train, eval_datasets, n_epochs=n_epochs, lr=lr, n_classes=num_classes)
    print(f"Final accuracy: {results_weak['test']:.3f}")
    print(f"Final supervisor-student agreement: {results_weak['test_weak']:.3f}")
    print(f"Accuracy by epoch: {[acc.item() for acc in results_weak['test_all']]}")
    print(
        f"Supervisor-student agreement by epoch: {[acc.item() for acc in results_weak['test_weak_all']]}"
    )

    print("Training logreg on ground truth labels...")
    results_gt = train_logreg(x_train, y_train, eval_datasets, n_epochs=n_epochs, lr=lr, n_classes=num_classes)
    print(f"Final accuracy: {results_gt['test']:.3f}")
    print(f"Accuracy by epoch: {[acc.item() for acc in results_gt['test_all']]}")

    print("\n\n" + "=" * 100)
    print(f"Weak label accuracy: {weak_acc:.3f}")
    print(f"Weak→Strong accuracy: {results_weak['test']:.3f}")
    print(f"Strong accuracy: {results_gt['test']:.3f}")
    print(f"Accuracy recovery: {(results_weak['test'] - weak_acc) / (results_gt['test'] - weak_acc):.3f}")
    print("=" * 100)

    type_teacher = 'soft' if soft_teacher else 'hard'
    fname = os.path.join(result_path, f'result_{weak_model_name}_{strong_model_name}_{type_teacher}_{stage}_student_{lr:.6f}.json')
    summary = {
        'teacher': weak_acc.item(),
        'baseline': results_weak['test'].item(),
        'groundtruth': results_gt['test'].item(),
    }
    with open(fname, "w") as outfile:
        json.dump(summary, outfile)

if __name__ == "__main__":
    fire.Fire(main)
