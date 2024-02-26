import fire
import numpy as np
import torch
import tqdm
import os
import json

from data import get_imagenet
from models import alexnet, resnet50_dino, vitb8_dino, vits14_dino, vitb14_dino, probe
from torch import nn
from helper import save_embedding, save_result, load_weak_embedding, save_weak_embedding, load_strong_embedding, save_strong_embedding
from single_weak_strong import get_model, get_embeddings, train_logreg

torch.set_printoptions(precision=4, sci_mode=False)

import pdb

def main(
    *weak_path,
    soft_teacher: bool = True,
    batch_size: int = 128,
    weak_model_name: str = "alexnet",
    strong_model_name: str = "resnet50_dino",
    n_train: int = 40000,
    n_hidden: int = 0,
    seed: int = 0,
    data_path: str = "/root/",
    embed_path: str = "embedding/",
    result_path: str = "result/",
    ckpt_path: str = "ckpt/",
    save_every: int = 0,
    n_epochs: int = 10,
    lr: float = 1e-3,
):
    _, loader = get_imagenet(data_path, split="val", batch_size=batch_size, shuffle=False)

    num_teacher = len(weak_path)
    label_teachers = []
    acc_teachers = []
    for i in range(num_teacher):
        teacher_path = weak_path[i]
        weak_model = get_model(weak_model_name, teacher_path)

        category = weak_path[i].split('/')[-2]
        stage = 'epoch_' + '_'.join(weak_path[i].split('/')[-1].split('.')[0].split('-')[1:])
        fname = os.path.join(embed_path, weak_model_name, f'data_{weak_model_name}_{category}_{stage}.pkl')

        if os.path.exists(fname):
            try:
                gt_labels, weak_labels, weak_acc = load_weak_embedding(fname)
            except Exception as e:
                _, gt_labels, weak_labels, weak_acc = get_embeddings(weak_model, loader)
                save_weak_embedding(gt_labels, weak_labels, weak_acc, fname)
        else:
            _, gt_labels, weak_labels, weak_acc = get_embeddings(weak_model, loader)
            save_weak_embedding(gt_labels, weak_labels, weak_acc, fname)

        if not soft_teacher:
            weak_labels = nn.functional.one_hot(torch.argmax(weak_labels, dim=1), num_classes=1000).float()
            print('Convert teacher outputs to hard class labels')
        label_teachers.append(weak_labels)
        acc_teachers.append(weak_acc.item())
    print(f"Weak teacher accuracy: {[acc for acc in acc_teachers]}")

    strong_model = get_model(strong_model_name)
    fname = os.path.join(embed_path, f'data_{strong_model_name}.pkl')
    if os.path.exists(fname):
        embeddings, strong_gt_labels = load_strong_embedding(fname)
    else:
        embeddings, strong_gt_labels, _, _ = get_embeddings(strong_model, loader)
        save_strong_embedding(embeddings, strong_gt_labels, fname)

    assert torch.all(gt_labels == strong_gt_labels)
    del strong_gt_labels

    if not os.path.exists(result_path):
        os.makedirs(result_path)
    type_teacher = 'soft' if soft_teacher else 'hard'
    prefix = os.path.join(result_path, f'result_{weak_model_name}_{strong_model_name}_{type_teacher}_{num_teacher}_{stage}_student_{n_hidden}_{lr:.6f}')

    order = np.arange(len(embeddings))
    rng = np.random.default_rng(seed)
    rng.shuffle(order)
    x = embeddings[order]
    x_train, x_test = x[:n_train], x[n_train:]
    y = gt_labels[order]
    y_train, y_test = y[:n_train], y[n_train:]
    eval_datasets = {"test": (x_test, y_test)}
    print("# examples: ", x_train.shape[0], x_test.shape[0])

    # multi teacher selectively (oracle)
    label_oracle = torch.mean(torch.stack(label_teachers), dim=0)
    for i in range(num_teacher):
        str_start_end = weak_path[i].split('/')[-2].split('_')
        teacher_start = int(str_start_end[0])
        teacher_end = int(str_start_end[1])
        idx_select = (teacher_start <= gt_labels) & (gt_labels < teacher_end)
        label_oracle[idx_select] = label_teachers[i][idx_select]
    acc_ora = (label_oracle.argmax(dim=1) == gt_labels).float().sum() / gt_labels.shape[0]
    yw = label_oracle[order]
    yw_train, yw_test = yw[:n_train], yw[n_train:]

    # all data
    print("Training logreg on oracle labels...")
    results_ora = train_logreg(x_train, yw_train, eval_datasets, n_epochs=n_epochs, lr=lr)
    print(f"Teacher accuracy: {acc_ora:.3f}")
    print(f"Student accuracy: {results_ora['test']:.3f}")
    print(f"Accuracy by epoch: {[acc.item() for acc in results_ora['test_all']]}")
    summary = {
        'type': type_teacher,
        'number': num_teacher,
        'stage': stage,
        'teacher': acc_ora.item(),
        'student': results_ora['test'].item(),
    }
    with open(prefix + '_oracle.json', "w") as outfile:
        json.dump(summary, outfile)

if __name__ == "__main__":
    fire.Fire(main)
