import numpy as np
import torch
import pandas as pd
import pickle

import pdb

def load_embedding(fname):
    with open(fname, 'rb') as handle:
        gt_labels = pickle.load(handle)
        weak_labels = pickle.load(handle)
        weak_acc = pickle.load(handle)
        embeddings = pickle.load(handle)
        strong_gt_labels = pickle.load(handle)
    print('Load embeddings from', fname)
    return gt_labels, weak_labels, weak_acc, embeddings, strong_gt_labels


def load_weak_domain_embedding(fname):
    with open(fname, 'rb') as handle:
        gt_labels = pickle.load(handle)
        weak_labels = pickle.load(handle)
        domain_labels = pickle.load(handle)
        weak_acc = pickle.load(handle)
    print('Load weak domain embeddings from', fname)
    return gt_labels, weak_labels, domain_labels, weak_acc


def save_weak_domain_embedding(gt_labels, weak_labels, domain_labels, weak_acc, fname):
    with open(fname, 'wb') as handle:
        pickle.dump(gt_labels, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(weak_labels, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(domain_labels, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(weak_acc, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('Save weak domain embeddings to', fname)


def load_weak_embedding(fname):
    with open(fname, 'rb') as handle:
        gt_labels = pickle.load(handle)
        weak_labels = pickle.load(handle)
        weak_acc = pickle.load(handle)
    print('Load weak embeddings from', fname)
    return gt_labels, weak_labels, weak_acc


def save_weak_embedding(gt_labels, weak_labels, weak_acc, fname):
    with open(fname, 'wb') as handle:
        pickle.dump(gt_labels, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(weak_labels, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(weak_acc, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('Save weak embeddings to', fname)


def load_strong_embedding(fname):
    with open(fname, 'rb') as handle:
        gt_labels = pickle.load(handle)
        embeddings = pickle.load(handle)
    print('Load strong embeddings from', fname)
    return embeddings, gt_labels


def save_strong_embedding(embeddings, gt_labels, fname):
    with open(fname, 'wb') as handle:
        pickle.dump(gt_labels, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(embeddings, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('Save strong embeddings to', fname)


def load_result(fname):
    df = pd.read_csv(fname, index_col=0)
    return df


def print_param(model, key=None):
    for name, param in model.named_parameters():
        if key is None:
            print(name, param.data)
        else:
            print(name, param.data[key])
