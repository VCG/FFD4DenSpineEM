import glob
import numpy as np
from collections import defaultdict, OrderedDict


def get_fold_freseg_model(file):
    assert sum([x in file for x in ["Pointnet", "RandLA", "point-transformer"]]) == 1
    if "Pointnet" in file:
        model = "pointnet"
    elif "RandLA" in file:
        model = "randla"
    elif "point-transformer" in file:
        model = "point-transformer"

    if model == "pointnet":
        file = file.replace("freseg_", "")
    parts = file.split("/")
    if model == "pointnet":
        description = parts[-4]
    else:
        description = parts[-3]
    parts = description.split("_")
    fold = parts[0]
    assert fold in ["0", "1", "2", "3", "4"]
    fold = int(fold)
    freseg = parts[-1]
    assert freseg in ["True", "False"]
    freseg = freseg == "True"

    return fold, freseg, model


def to_regular_dict(d):
    if isinstance(d, defaultdict):
        return {k: to_regular_dict(v) for k, v in d.items()}
    else:
        return d


def aggregate_metrics():
    recall_thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]

    BASE_PATH = "/data/adhinart/freseg/" # Andromeda1
    # BASE_PATH = "/home/adhinart/projects/freseg2" # Andromeda2

    files = (
        sorted(
            glob.glob(
                f"{BASE_PATH}/Pointnet_Pointnet2_pytorch/log/*/*/output/metrics.npz"
            )
        )
        + sorted(
            glob.glob(f"{BASE_PATH}/RandLA-Net-pytorch2/runs/*/output/metrics.npz")
        )
        + sorted(
            glob.glob(
                f"{BASE_PATH}/point-transformer/tool/exp/freseg/*/output/metrics.npz"
            )
        )
    )

    # across freseg settings
    # first layer is model, second is freseg setting, third layer is dataset, final is metric
    aggg = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    )

    for file in files:
        fold, freseg, model = get_fold_freseg_model(file)
        print(file)
        binary_metrics = np.load(file, allow_pickle=True)["binary_dice"].item()
        recall_metrics = np.load(file, allow_pickle=True)["binary_recall"].item()

        for dataset in binary_metrics.keys():
            new_recall_metrics = defaultdict(dict)
            for (trunk_id, label), values in recall_metrics[dataset].items():
                new_recall_metrics[trunk_id][label] = values

            agg = defaultdict(list)

            # ===
            scores = {}
            # ===
            for trunk_id in binary_metrics[dataset].keys():
                r = new_recall_metrics[trunk_id]
                bd = binary_metrics[dataset][trunk_id]

                binary_trunk_dice = 2 * bd["tn"] / (2 * bd["tn"] + bd["fn"] + bd["fp"])
                binary_trunk_iou = bd["tn"] / (bd["tn"] + bd["fn"] + bd["fp"])
                binary_spine_dice = 2 * bd["tp"] / (2 * bd["tp"] + bd["fn"] + bd["fp"])
                binary_spine_iou = bd["tp"] / (bd["tp"] + bd["fn"] + bd["fp"])

                spine_recall = [
                    r[label]["tp"] / (r[label]["tp"] + r[label]["fn"])
                    for label in r.keys()
                ]
                for recall_threshold in recall_thresholds:
                    thresh_spine_recall = [x >= recall_threshold for x in spine_recall]
                    agg[f"{recall_threshold}_spine_recall"].extend(thresh_spine_recall)
                agg["average_spine_recall"].extend(spine_recall)
                agg["binary_trunk_dice"].append(binary_trunk_dice)
                agg["binary_trunk_iou"].append(binary_trunk_iou)
                agg["binary_spine_dice"].append(binary_spine_dice)
                agg["binary_spine_iou"].append(binary_spine_iou)

                # ===
                scores[trunk_id] = binary_spine_iou
                # ===
            # ===
            # print trunk_ids in order of ascending binary_spine_iou
            print(dataset, sorted(scores.items(), key=lambda x: x[1]))
            # ===

            agg = {k: np.mean(v) for k, v in agg.items()}
            print(dataset, agg)
            for k, v in agg.items():
                aggg[model][freseg][dataset][k].append(v)
        print()
    aggg = to_regular_dict(aggg)
    new_aggg = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    )
    for model, freseg_datasets in aggg.items():
        for freseg, datasets in freseg_datasets.items():
            for dataset, metrics in datasets.items():
                for metric, values in metrics.items():
                    new_aggg[model][freseg][dataset][metric]["mean"] = np.mean(values)
                    new_aggg[model][freseg][dataset][metric]["std"] = np.std(values)
                    # 95 conf interval
                    plusminus = 1.96 * np.std(values) / np.sqrt(len(values))
                    new_aggg[model][freseg][dataset][metric]["conf"] = (
                        np.mean(values) - plusminus,
                        np.mean(values) + plusminus,
                    )
    new_aggg = to_regular_dict(new_aggg)
    print(new_aggg)
    aggg_to_md(aggg, new_aggg)


def aggg_to_md(aggg, new_aggg):
    # first layer is model, second is freseg setting, third layer is dataset, final is metric
    _models = sorted(new_aggg.keys())
    models_map = OrderedDict(
        [
            ("PointNet++", "pointnet"),
            ("RandLA-Net", "randla"),
            ("PointTransformer", "point-transformer"),
        ]
    )
    _datasets = sorted(new_aggg[_models[0]][True].keys())
    datasets_map = OrderedDict([("M50", "seg_den"), ("M10", "mouse"), ("H10", "human")])
    _metrics = sorted(new_aggg[_models[0]][True][_datasets[0]].keys())
    metrics_map = OrderedDict(
        [
            ("Spine IoU", "binary_spine_iou"),
            ("Trunk IoU", "binary_trunk_iou"),
            ("Spine Dice", "binary_spine_dice"),
            ("Trunk Dice", "binary_trunk_dice"),
            ("Spine Accuracy", "0.7_spine_recall"),
            ("Average Spine Recall", "average_spine_recall"),
        ]
    )
    md = ""
    pm = "±"

    md += (
        f"## Average Results:  Mean {pm} Standard Deviation (95% Confidence Interval)\n"
    )
    md += f"| Model | Dataset | " + " | ".join(list(metrics_map.keys())) + " |\n"
    md += "| --- | --- | " + " | ".join(["---"] * len(metrics_map)) + " |\n"
    for model in models_map:
        for freseg in [False, True]:
            for dataset in datasets_map:
                if freseg:
                    md += f"| {model} w. FFD | {dataset} | "
                else:
                    md += f"| {model} | {dataset} | "
                for metric in metrics_map:
                    mean = new_aggg[models_map[model]][freseg][datasets_map[dataset]][
                        metrics_map[metric]
                    ]["mean"]
                    std = new_aggg[models_map[model]][freseg][datasets_map[dataset]][
                        metrics_map[metric]
                    ]["std"]
                    conf = new_aggg[models_map[model]][freseg][datasets_map[dataset]][
                        metrics_map[metric]
                    ]["conf"]
                    md += (
                        f"{mean:.4f} {pm} {std:.4f} ({conf[0]:.4f} - {conf[1]:.4f}) | "
                    )
                md += "\n"
    md += "\n"

    # per fold
    for fold in range(5):
        md += f"## Fold {fold}\n"
        md += f"| Model | Dataset | " + " | ".join(list(metrics_map.keys())) + " |\n"
        md += "| --- | --- | " + " | ".join(["---"] * len(metrics_map)) + " |\n"
        for model in models_map:
            for freseg in [False, True]:
                for dataset in datasets_map:
                    if freseg:
                        md += f"| {model} w. FFD | {dataset} | "
                    else:
                        md += f"| {model} | {dataset} | "
                    for metric in metrics_map:
                        val = aggg[models_map[model]][freseg][datasets_map[dataset]][
                            metrics_map[metric]
                        ][fold]
                        md += f"{val:.4f} | "
                        # md += f"{mean*100:.1f} {pm} {std*100:.1f} | "
                    md += "\n"
        md += "\n"

    print(md)


if __name__ == "__main__":
    aggregate_metrics()
