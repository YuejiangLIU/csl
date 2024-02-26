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


def get_conservative_estimate(y_next, y_prev, maxk=2):
    _, pred = y_prev.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(y_next.argmax(dim=1).view(1, -1).expand_as(pred))
    correct = correct.t()
    consensus = correct.any(dim=1)
    return consensus


def get_oracle_rank(y_next, y_oracle):
    lw_train = nn.functional.cross_entropy(y_next.log(), y_oracle, reduction='none')
    lw_order = torch.argsort(lw_train)          # ascending order
    lw_ranks = torch.argsort(lw_order)
    return lw_ranks


def get_student_rank(logit_student, y_next):
    lw_train = nn.functional.cross_entropy(logit_student, y_next, reduction='none')
    lw_order = torch.argsort(lw_train)          # ascending order
    lw_ranks = torch.argsort(lw_order)
    return lw_ranks


def get_output(model, x_train, num_classes=1000, batch_size=1024):
    dataset = torch.utils.data.TensorDataset(x_train)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    num_sample = x_train.shape[0]
    output = torch.zeros((num_sample, num_classes))

    with torch.no_grad():
        for batch_idx, x_batch in enumerate(data_loader):
            x = x_batch[0].cuda()
            pred = model(x)
            output[batch_idx * batch_size: batch_idx * batch_size + x.size(0)] = pred.cpu()

    return output


def get_precision_recall(pred, target):
    # Calculate TP, FP, FN
    TP = torch.sum((pred == 1) & (target == 1))
    FP = torch.sum((pred == 1) & (target == 0))
    FN = torch.sum((pred == 0) & (target == 1))

    # Calculate Precision and Recall
    precision = TP.float() / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP.float() / (TP + FN) if (TP + FN) > 0 else 0

    return precision.item(), recall.item()


def normalize_topk_confidence(pred_prev, pred_curr, k=5):
    topk_values, topk_indices = torch.topk(pred_curr, k, dim=1)

    logit_prev = torch.log(pred_prev)
    logit_curr = torch.log(pred_curr)
    normalized_logit_prev = torch.gather(logit_prev, 1, topk_indices)
    normalized_logit_curr = torch.gather(logit_curr, 1, topk_indices)

    normalized_confidence_prev = torch.nn.functional.softmax(normalized_logit_prev, dim=1)
    normalized_confidence_curr = torch.nn.functional.softmax(normalized_logit_curr, dim=1)

    return normalized_confidence_prev.max(dim=1)[0], normalized_confidence_curr.max(dim=1)[0]


def get_consensus_rate(teacher_prev, teacher_curr, ground_truth=None):
    consensus_tt = (teacher_prev.argmax(dim=1) == teacher_curr.argmax(dim=1))
    rate_top1_consensus = (consensus_tt).sum() / consensus_tt.shape[0]
    # print(f'teacher-teacher top1 consensus rate: {rate_top1_consensus:.2f}')

    consensus_top2 = get_conservative_estimate(teacher_curr, teacher_prev, 2)
    rate_top2_consensus = (consensus_top2).sum() / consensus_top2.shape[0]
    # print(f'teacher-teacher top2 consensus rate: {rate_top2_consensus:.2f}')

    consensus_top3 = get_conservative_estimate(teacher_curr, teacher_prev, 3)
    rate_top3_consensus = (consensus_top3).sum() / consensus_top3.shape[0]
    # print(f'teacher-teacher top3 consensus rate: {rate_top3_consensus:.2f}')

    if ground_truth is not None:
        consensus_prev = (teacher_prev.argmax(dim=1) == ground_truth)
        rate_prev = (consensus_prev).sum() / consensus_prev.shape[0]
        print(f'previous teacher accuracy: {rate_prev:.2f}')

        consensus_curr = (teacher_curr.argmax(dim=1) == ground_truth)
        rate_curr = (consensus_curr).sum() / consensus_curr.shape[0]
        print(f'current teacher accuracy: {rate_curr:.2f}')

        precision_prev, recall_prev = get_precision_recall(consensus_tt, consensus_prev)
        print(f'consensus for previous teacher: precision = {precision_prev:.2f}, recall = {recall_prev:.2f}')
        precision_curr, recall_curr = get_precision_recall(consensus_tt, consensus_curr)
        print(f'consensus for current teacher: precision = {precision_curr:.2f}, recall = {recall_curr:.2f}')

        # confidence consistent
        p_prev, y_prev = teacher_prev.max(dim=1)
        p_curr, y_curr = teacher_curr.max(dim=1)
        consistent_tt = consensus_tt & (p_curr >= p_prev)
        rate_tt_consistent = (consistent_tt).sum() / consistent_tt.shape[0]
        print(f'teacher-teacher consistent rate: {rate_tt_consistent:.2f}')

        precision_prev, recall_prev = get_precision_recall(consistent_tt, consensus_prev)
        print(f'consistent for previous teacher: precision = {precision_prev:.2f}, recall = {recall_prev:.2f}')
        precision_curr, recall_curr = get_precision_recall(consistent_tt, consensus_curr)
        print(f'consistent for current teacher: precision = {precision_curr:.2f}, recall = {recall_curr:.2f}')

    return rate_top1_consensus.item(), rate_top2_consensus.item(), rate_top3_consensus.item()


def main(
    *weak_path,
    soft_teacher: bool = True,
    batch_size: int = 128,
    weak_model_name: str = "alexnet",
    strong_model_name: str = "resnet50_dino",
    denoise_criterion: str = "top3",
    n_train: int = 40000,
    seed: int = 0,
    data_path: str = "/root/",
    embed_path: str = "embedding/",
    result_path: str = "result/",
    ckpt_path: str = "ckpt/",
    save_every: int = 0,
    n_epochs: int = 10,
    lr: float = 1e-3,
    num_classes: int = 1000,
):
    label_layers = []
    label_teachers = []
    acc_teachers = []
    for teacher_path in weak_path:
        category = teacher_path.split('/')[-2]
        stage = 'epoch_' + '_'.join(teacher_path.split('/')[-1].split('.')[0].split('-')[1:])
        fname = os.path.join(embed_path, weak_model_name, f'data_{weak_model_name}_{category}_{stage}.pkl')

        str_start_end = category.split('_')
        teacher_start, teacher_end = int(str_start_end[0]), int(str_start_end[1])

        if os.path.exists(fname):
            try:
                gt_labels, weak_labels, weak_acc = load_weak_embedding(fname)
            except Exception as e:
                weak_model = get_model(weak_model_name, teacher_path)
                _, loader = get_imagenet(data_path, split="val", batch_size=batch_size, shuffle=False)
                _, gt_labels, weak_labels, weak_acc = get_embeddings(weak_model, loader)
                save_weak_embedding(gt_labels, weak_labels, weak_acc, fname)
        else:
            weak_model = get_model(weak_model_name, teacher_path)
            _, loader = get_imagenet(data_path, split="val", batch_size=batch_size, shuffle=False)
            _, gt_labels, weak_labels, weak_acc = get_embeddings(weak_model, loader)
            save_weak_embedding(gt_labels, weak_labels, weak_acc, fname)

        if not soft_teacher:
            weak_labels = nn.functional.one_hot(torch.argmax(weak_labels, dim=1), num_classes=num_classes).float()
            print('Convert teacher outputs to hard class labels')

        label_teachers.append(weak_labels)
        acc_teachers.append(weak_acc.item())

        if teacher_end == num_classes:
            label_layers.append(label_teachers)
            print(f"Weak teacher accuracy: {[acc for acc in acc_teachers]}")
            label_teachers = []
            acc_teachers = []

    num_teacher_all = [len(layer) for layer in label_layers]

    print('Teacher #:', num_teacher_all)

    strong_model = get_model(strong_model_name)
    fname = os.path.join(embed_path, f'data_{strong_model_name}.pkl')
    if os.path.exists(fname):
        embeddings, strong_gt_labels = load_strong_embedding(fname)
    else:
        _, loader = get_imagenet(data_path, split="val", batch_size=batch_size, shuffle=False)
        embeddings, strong_gt_labels, _, _ = get_embeddings(strong_model, loader)
        save_strong_embedding(embeddings, strong_gt_labels, fname)

    assert torch.all(gt_labels == strong_gt_labels)
    del strong_gt_labels

    if not os.path.exists(result_path):
        os.makedirs(result_path)
    type_teacher = 'soft' if soft_teacher else 'hard'
    num_teacher_str = '_'.join(map(str, num_teacher_all))
    prefix = os.path.join(result_path, f'result_{weak_model_name}_{strong_model_name}_{type_teacher}_{num_teacher_str}_{stage}_student_{lr:.6f}_{denoise_criterion}_{seed}')

    order = np.arange(len(embeddings))
    rng = np.random.default_rng(seed)
    rng.shuffle(order)
    x = embeddings[order]
    x_train, x_test = x[:n_train], x[n_train:]
    y = gt_labels[order]
    y_train, y_test = y[:n_train], y[n_train:]
    eval_datasets = {"test": (x_test, y_test)}
    print("# examples: ", x_train.shape[0], x_test.shape[0])

    # main loop
    result_teacher = list()
    result_student = list()
    result_rate = list()
    result_precision = list()
    result_recall = list()
    yw_prev = None
    logit_prev = None

    for label_teachers in label_layers:

        # teacher assignment
        num_teacher = len(label_teachers)
        if num_teacher > 1:
            weak_stack = torch.stack(label_teachers, dim=1)
            gt_assign = torch.zeros((weak_stack.shape[0], num_teacher))

            # class mapping
            mapping_class_teacher = torch.zeros((num_classes, num_teacher))
            num_classes_per_teacher = num_classes // num_teacher
            for i in range(num_teacher):
                teacher_start = i * num_classes_per_teacher
                teacher_end = (i+1) * num_classes_per_teacher
                mapping_class_teacher[teacher_start:teacher_end, i] = 1.0
                idx_select = (teacher_start <= gt_labels) & (gt_labels < teacher_end)
                gt_assign[idx_select, i] = 1.0
            gt_labels_full = nn.functional.one_hot(gt_labels, num_classes=1000).float()
            gt_assign_full = torch.matmul(gt_labels_full, mapping_class_teacher)

            yws = weak_stack[order]
            zw = gt_assign_full[order]

            yws_train, zw_train = yws[:n_train], zw[:n_train]

            pred_output = logit_prev # assignment by previous student
            # pred_output = yw_prev # assingment by previous teacher
            hard_assign = torch.matmul(nn.functional.one_hot(pred_output.argmax(dim=1), num_classes=1000).float(), mapping_class_teacher)
            acc_hard = (hard_assign.argmax(dim=1) == zw_train.argmax(dim=1)).sum() / hard_assign.shape[0]
            yw_train = (yws_train * hard_assign.unsqueeze(2)).sum(dim=1)

            # oracel assignment
            # yw_train = (yws_train * zw_train.unsqueeze(2)).sum(dim=1)

        else:
            yw = label_teachers[0][order]
            yw_train = yw[:n_train]

        acc_teacher = ((yw_train.argmax(dim=1) == y_train).sum() / n_train).item()
        result_teacher.append(acc_teacher)
        print(f"teacher x{num_teacher}: collective accuracy = {acc_teacher:.3f}")

        # Sample selection
        if num_teacher > 1 and denoise_criterion != 'all':
            rate_top1, rate_top2, rate_top3 = get_consensus_rate(torch.nn.functional.softmax(logit_prev, dim=1), yw_train)
            if denoise_criterion == 'top1':
                rate_keep = rate_top1
            elif denoise_criterion == 'top2':
                rate_keep = rate_top2
            elif denoise_criterion == 'top3':
                rate_keep = rate_top3
            elif denoise_criterion == 'oracle':
                raise NotImplementedError
            else:
                raise NotImplementedError
        else:
            rate_keep = 1.0

        if rate_keep < 1.0:
            rank_train = get_student_rank(logit_prev, yw_train.argmax(dim=1))
            keep_train = rank_train < (n_train * rate_keep)
        else:
            keep_train = torch.ones(n_train, dtype=torch.bool)

        correct_train = (yw_train.argmax(dim=1) == y_train)
        precision, recall = get_precision_recall(keep_train, correct_train)
        print(f"Select {rate_keep*100:.1f}% samples: precision = {precision:.3f}, recall = {recall:.3f}")
        result_rate.append(rate_keep)
        result_precision.append(precision)
        result_recall.append(recall)

        # Train
        print(f"Training logreg on the selected weak labels...")
        x_conserve, yw_conserve = x_train[keep_train], yw_train[keep_train]
        epochs_conserve = int(n_epochs / rate_keep)                 # keep the same number of iterations
        model = probe(x_train.shape[1], num_classes)
        results_weak = train_logreg(x_conserve, yw_conserve, eval_datasets, n_epochs=epochs_conserve, lr=lr, ckpt_path=ckpt_path, save_every=save_every, model=model)
        print(f"Final accuracy: {results_weak['test']:.3f}")
        result_student.append(max(results_weak['test_all']).item())

        # Update
        yw_prev = yw_train
        logit_prev = get_output(model, x_train)

    summary = {
        'type': type_teacher,
        'number': num_teacher_all,
        'stage': stage,
        'rate': result_rate,
        'precision': result_precision,
        'recall': result_recall,
        'teacher': result_teacher,
        'student': result_student,
    }
    with open(prefix + '_csl.json', "w") as outfile:
        json.dump(summary, outfile)

    print('Teacher accuracy: \n', result_teacher)
    print('Student accuracy: \n', result_student)


if __name__ == "__main__":
    fire.Fire(main)
