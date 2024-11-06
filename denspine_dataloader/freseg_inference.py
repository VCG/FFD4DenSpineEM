import sys

sys.path.append("/data/adhinart/dendrite/scripts/igneous")

from pathlib import Path
import os
import torch
from tqdm import tqdm
import math
import numpy as np
from collections import defaultdict

from frenet import get_closest, get_dataloader

from typing import Callable


def get_test_dataloaders(
    path_length: int,
    fold: int,
    num_workers: int,
    frenet: bool,
):
    num_points = 1000000
    batch_size = 1

    datasets = {}
    files = {}
    datasets["seg_den"], files["seg_den"] = get_dataloader(
        species="seg_den",
        path_length=path_length,
        num_points=num_points,
        fold=fold,
        is_train=False,
        batch_size=batch_size,
        num_workers=num_workers,
        frenet=frenet,
    )
    datasets["mouse"], files["mouse"] = get_dataloader(
        species="mouse",
        path_length=path_length,
        num_points=num_points,
        fold=-1,
        is_train=False,
        batch_size=batch_size,
        num_workers=num_workers,
        frenet=frenet,
    )
    datasets["human"], files["human"] = get_dataloader(
        species="human",
        path_length=path_length,
        num_points=num_points,
        fold=-1,
        is_train=False,
        batch_size=batch_size,
        num_workers=num_workers,
        frenet=frenet,
    )

    return datasets, files


def do_inference(
    dataloader, model, model_inference, npoint, batch_size, files, output_dir
):
    model.eval()
    with torch.no_grad():
        # NOTE: dataloader_idx assumes that shuffle is off (ie that is_train=False)
        for dataloader_idx, (trunk_id, points, target, unmodified_points) in tqdm(
            enumerate(dataloader), total=len(dataloader), smoothing=0.9
        ):
            # NOTE: assumes order already randomized
            # divide points into batches
            assert points.shape[0] == 1 and target.shape[0] == 1
            num_total_points = points.shape[1]

            # ceil
            num_batches = int(math.ceil(num_total_points / npoint))
            padding = num_batches * npoint - num_total_points

            if padding > 0:
                points = torch.cat([points, points[:, :padding, :]], dim=1)
            assert points.shape[1] % npoint == 0

            # now reshape points to (num_batches, npoint, 3) and target to (num_batches, npoint)
            points = points.view(num_batches, npoint, 3).float()
            preds = []

            for i in range(math.ceil(num_batches / batch_size)):
                points_batch = points[i * batch_size : (i + 1) * batch_size].cuda()
                preds.append(model_inference(model, points_batch).cpu().numpy())

            pred = np.concatenate(preds, axis=0)[:num_total_points]

            np.savez(
                os.path.join(output_dir, os.path.basename(files[dataloader_idx])),
                trunk_id=trunk_id,
                pc=unmodified_points.squeeze(0).numpy(),
                trunk_pc=np.load(files[dataloader_idx])["trunk_pc"],
                label=target.squeeze(0).numpy(),
                pred=pred,
            )


def metrics(output_path):
    datasets = os.listdir(output_path)
    datasets = [d for d in datasets if os.path.isdir(os.path.join(output_path, d))]

    results = {}
    for dataset in datasets:
        results[dataset] = defaultdict(int)
        files = os.listdir(os.path.join(output_path, dataset))
        for file in tqdm(files):
            data = np.load(os.path.join(output_path, dataset, file))
            trunk_id = data["trunk_id"].item()

            points = data["pc"]
            pred = data["pred"]
            label = data["label"]

            # create a structured array of pred, label
            structured = np.zeros(len(pred), dtype=[("pred", int), ("label", int)])
            structured["pred"] = pred
            structured["label"] = label

            unique, counts = np.unique(structured, return_counts=True)
            for u, c in zip(unique, counts):
                results[dataset][(trunk_id, u["label"], u["pred"])] += c
        results[dataset] = dict(results[dataset])

    binary_dice = {}
    binary_recall = {}
    for dataset in datasets:
        binary_dice[dataset] = defaultdict(lambda: {"tp": 0, "tn": 0, "fp": 0, "fn": 0})
        binary_recall[dataset] = defaultdict(lambda: {"tp": 0, "fn": 0})
        for (trunk_id, label, pred), c in results[dataset].items():
            if label > 0:
                if pred > 0:
                    binary_dice[dataset][trunk_id]["tp"] += c
                    binary_recall[dataset][(trunk_id, label)]["tp"] += c
                else:
                    binary_dice[dataset][trunk_id]["fn"] += c
                    binary_recall[dataset][(trunk_id, label)]["fn"] += c
            else:
                if pred > 0:
                    binary_dice[dataset][trunk_id]["fp"] += c
                else:
                    binary_dice[dataset][trunk_id]["tn"] += c

        binary_dice[dataset] = dict(binary_dice[dataset])
        binary_recall[dataset] = dict(binary_recall[dataset])
    np.savez(
        os.path.join(output_path, "metrics.npz"),
        binary_dice=binary_dice,
        binary_recall=binary_recall,
    )


def evaluation(
    output_path: str,
    fold: int,
    path_length: int,
    npoint: int,
    frenet: bool,
    batch_size: int,
    num_workers: int,
    load_model: Callable,
    model_inference: Callable,
):
    # output_path is of form {fold/pathlength/num_points/frenet specific path}
    model = load_model(output_path)
    print(
        f"Loaded model from {output_path}, fold {fold}, path_length {path_length}, npoint {npoint}, frenet {frenet}"
    )

    output_path = os.path.join(output_path, "output")

    dataloaders, files = get_test_dataloaders(
        path_length=path_length,
        fold=fold,
        num_workers=num_workers,
        frenet=frenet,
    )
    print(f"Loaded dataloaders")

    for dataset_name, dataloader in dataloaders.items():
        if not os.path.exists(os.path.join(output_path, dataset_name)):
            os.makedirs(os.path.join(output_path, dataset_name))
        do_inference(
            dataloader,
            model,
            model_inference,
            npoint,
            batch_size,
            files[dataset_name],
            os.path.join(output_path, dataset_name),
        )

    metrics(output_path)
