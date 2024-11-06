import glob
import numpy as np
from collections import defaultdict


def get_fold_and_freseg(file):
    parts = file.split("/")
    description = parts[-3]
    parts = description.split("_")
    fold = parts[0]
    assert fold in ["0", "1", "2", "3", "4"]
    fold = int(fold)
    freseg = parts[-1]
    assert freseg in ["True", "False"]
    freseg = freseg == "True"

    return fold, freseg


def to_regular_dict(d):
    if isinstance(d, defaultdict):
        return {k: to_regular_dict(v) for k, v in d.items()}
    else:
        return d


def aggregate_metrics():
    recall_thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]

    files = (
        sorted(
            glob.glob(
                "/data/adhinart/freseg/Pointnet_Pointnet2_pytorch/log/*/*/output/metrics.npz"
            )
        )
        + sorted(
            glob.glob(
                "/mmfs1/data/adhinart/freseg/RandLA-Net-pytorch2/runs/*/output/metrics.npz"
            )
        )
        + glob.glob(
            "/mmfs1/data/adhinart/freseg/point-transformer/tool/exp/freseg/*/output/metrics.npz"
        )
    )

    # across freseg settings
    # first layer is freseg setting, second layer is dataset, final is metric
    aggg = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for file in files:
        fold, freseg = get_fold_and_freseg(file)
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

                for recall_threshold in recall_thresholds:
                    spine_recall = [
                        r[label]["tp"] / (r[label]["tp"] + r[label]["fn"])
                        for label in r.keys()
                    ]
                    spine_recall = [r >= recall_threshold for r in spine_recall]

                    agg[f"{recall_threshold}_spine_recall"].extend(spine_recall)
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
                aggg[freseg][dataset][k].append(v)
        print()
    aggg = to_regular_dict(aggg)
    new_aggg = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for freseg, datasets in aggg.items():
        for dataset, metrics in datasets.items():
            for metric, values in metrics.items():
                new_aggg[freseg][dataset][metric]["mean"] = np.mean(values)
                new_aggg[freseg][dataset][metric]["std"] = np.std(values)
                # 95 conf interval
                plusminus = 1.96 * np.std(values) / np.sqrt(len(values))
                new_aggg[freseg][dataset][metric]["conf"] = (
                    np.mean(values) - plusminus,
                    np.mean(values) + plusminus,
                )
    new_aggg = to_regular_dict(new_aggg)
    print("freseg true")
    print(new_aggg[True])
    print("freseg false")
    print(new_aggg[False])


if __name__ == "__main__":
    aggregate_metrics()
