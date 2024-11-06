import glob
from dataloader import visualize_batch
from cache_dataloader import CachedDataset
from magicpickle import MagicPickle

# def visualize_batch(pc, trunk_skel, label):
if __name__ == "__main__":
    with MagicPickle("think-jason") as mp:
        if mp.is_remote:
            dataset = CachedDataset(
                "/data/adhinart/dendrite/scripts/igneous/outputs/seg_den/dataset_1000000_10000",
                folds=[
                    [3, 5, 11, 12, 23, 28, 29, 32, 39, 42],
                    [8, 15, 19, 27, 30, 34, 35, 36, 46, 49],
                    [9, 14, 16, 17, 21, 26, 31, 33, 43, 44],
                    [2, 6, 7, 13, 18, 24, 25, 38, 41, 50],
                    [1, 4, 10, 20, 22, 37, 40, 45, 47, 48],
                ],
                num_points=10000,
                fold=2,
                is_train=False,
            )
            data = [dataset[i] for i in range(len(dataset))]
            mp.save(data)
        else:
            data = mp.load()
            for i in range(len(data)):
                print(f"Visualizing {i}/{len(data)}")
                trunk_id, pc, trunk_pc, label = data[i]
                visualize_batch(pc, trunk_pc, label)
# if __name__ == "__main__":
#     with MagicPickle("think-jason") as mp:
#         if mp.is_remote:
#             dataset = CachedDataset(
#                 # "/data/adhinart/dendrite/scripts/igneous/outputs/seg_den/dataset_0_10000_30000_old",
#                 "/data/adhinart/dendrite/scripts/igneous/outputs/human/dataset_1000000_10000",
#                 num_points=30000,
#                 folds=[
#                     [1, 2, 3, 4, 5, 6, 7, 8, 9],
#                 ],
#                 fold=-1,
#                 is_train=False,
#             )
#             data = [dataset[i] for i in range(len(dataset))]
#             mp.save(data)
#         else:
#             data = mp.load()
#             for i in range(len(data)):
#                 print(f"Visualizing {i}/{len(data)}")
#                 trunk_id, pc, trunk_pc, label = data[i]
#                 visualize_batch(pc, trunk_pc, label)
