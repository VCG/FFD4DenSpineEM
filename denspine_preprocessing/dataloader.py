import numpy as np
import zarr
from utils import get_conf, groupby
from visualize import read_mappings
from scipy.spatial import KDTree
import networkx as nx
from torch.utils.data import Dataset
import contextlib

from cloudvolume import Skeleton
from typing import Optional, List, Set, Iterator, Tuple, Dict, Callable
from collections import defaultdict
from tqdm import tqdm

from joblib import Parallel, delayed
import sys

sys.setrecursionlimit(10**6)


@contextlib.contextmanager
def temp_seed(seed):
    # https://stackoverflow.com/questions/49555991/can-i-create-a-local-numpy-random-seed
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def find_paths(
    G: nx.Graph,
    start: int,
    target_weight: float,
    path: Optional[List[int]] = None,
    seen: Optional[Set[int]] = None,
    current_weight: float = 0,
) -> Iterator[Tuple[Tuple[int], float]]:
    """
    Given a trunk skeleton graph, find all (non-intesecting) paths from a starting node to a node with total weight (slightly) greater than a target weight

    Parameters
    ----------
    G
    start: index of the starting node
    target_weight

    path: variable used for recursion
    seen: variable used for recursion
    current_weight: variable used for recursion

    Returns
    -------
    Iterator[np.ndarray]:
    and current_weight
    """
    if path is None:
        path = [start]
    if seen is None:
        seen = {start}

    # get direct descendants
    desc = nx.descendants_at_distance(G, start, 1)
    valid_desc = [n for n in desc if n not in seen]
    if not valid_desc:
        # yield stump
        yield tuple(path), current_weight
    else:
        for n in valid_desc:
            edge_weight = G[start][n]["weight"]
            new_weight = current_weight + edge_weight
            if new_weight >= target_weight:
                yield tuple(path + [n]), new_weight
            else:
                yield from find_paths(
                    G, n, target_weight, path + [n], seen.union([n]), new_weight
                )


def get_random_path(G: nx.Graph, target_weight: float) -> Tuple[Tuple[int], float]:
    """
    Given a trunk skeleton graph, find a random path greater than a target weight

    Parameters
    ----------
    G
    target_weight

    Returns
    -------
    Tuple[int]:
    """
    start = np.random.choice(list(G.nodes))
    paths = list(find_paths(G, start=start, target_weight=target_weight))
    return paths[np.random.choice(len(paths))]


def get_best_path(G, start, target_weight, nodes_remaining):
    candidate_paths = list(find_paths(G, start=start, target_weight=target_weight))
    assert len(candidate_paths) > 0, f"No paths found"
    candidate_paths.sort(key=lambda x: len(set(x[0]) & nodes_remaining), reverse=True)
    path, path_weight = candidate_paths[0]

    return path, path_weight


def get_spanning_paths(G: nx.Graph, target_weight: float) -> List[Tuple[int]]:
    """
    Given a trunk skeleton graph, find a set of paths that span all the vertices in the graph
    Parameters
    ----------
    G
    target_weight
    Returns
    -------
    List[Tuple[int]]:
    """
    nodes_remaining = set(G.nodes)
    paths = []
    while nodes_remaining:
        # Calculate degree for each node in nodes_remaining
        degree_dict = {
            node: sum(
                1 for neighbor in G.neighbors(node) if neighbor in nodes_remaining
            )
            for node in nodes_remaining
        }
        min_degree = min(degree_dict.values())
        min_degree_nodes = [
            node for node, degree in degree_dict.items() if degree == min_degree
        ]

        start = np.random.choice(min_degree_nodes)
        path, path_weight = get_best_path(G, start, target_weight, nodes_remaining)
        # if less than target_weight, redo search at end of path
        if path_weight < target_weight:
            path, path_weight = get_best_path(
                G,
                start=path[-1],
                target_weight=target_weight,
                nodes_remaining=nodes_remaining,
            )
            if path_weight < target_weight:
                print(f"Warning: path has weight {path_weight} out of {target_weight}")

        paths.append(path)
        nodes_remaining -= set(path)
    return paths


def nx_from_skel(skel: Skeleton) -> nx.Graph:
    """
    Given a cloudvolume Skeleton, convert it into an undirected graph, with edge weights as the euclidean distance between vertices

    Parameters
    ----------
    skel
    """
    edges = []
    for a, b in skel.edges:
        l2 = np.linalg.norm(skel.vertices[a] - skel.vertices[b])
        edges.append((a, b, l2))

    G = nx.Graph()
    G.add_weighted_edges_from(edges)
    return G


def link_seeds(
    seed_data, seg_to_trunk, trunk_to_segs
) -> Tuple[Dict[int, int], Dict[int, List[int]]]:
    """
    Returns a dictionary {seed_id : seed_id_of_trunk} and {seed_id_of_trunk: [seed_id of trunks and spines]}
    Entire spine segments are assigned to the id of the closest trunk vertex

    Parameters
    ----------
    seed_data
    seg_to_trunk
    """
    seg_ids = sorted(seg_to_trunk.keys())
    trunk_ids = sorted(trunk_to_segs.keys())
    spine_ids = sorted(list(set(seg_ids) - set(trunk_ids)))

    skeleton_ids, groups = groupby(seed_data, seed_data["skeleton_id"])
    skel_id_to_group = {skel_id: group for skel_id, group in zip(skeleton_ids, groups)}

    seed_id_to_trunk_seed_id = {}
    for skel_id, group in zip(skeleton_ids, groups):
        assert (skel_id in trunk_ids) ^ (skel_id in spine_ids)
        if skel_id in trunk_ids:
            for x in group["seed_id"]:
                seed_id_to_trunk_seed_id[x] = x
        else:
            trunk_group = skel_id_to_group[seg_to_trunk[skel_id]]
            # NOTE: everything here is isotropic
            trunk_pc = np.stack([trunk_group[f"seed_coord_{c}"] for c in "zyx"], axis=1)
            trunk_seed_id = trunk_group["seed_id"]
            spine_pc = np.stack([group[f"seed_coord_{c}"] for c in "zyx"], axis=1)
            spine_seed_id = group["seed_id"]

            dist, idx = get_closest(spine_pc, trunk_pc)
            closest_trunk_idx = trunk_seed_id[idx[np.argmin(dist)]]
            for x in spine_seed_id:
                seed_id_to_trunk_seed_id[x] = closest_trunk_idx

    trunk_seed_id_to_seed_ids = defaultdict(list)
    for k, v in seed_id_to_trunk_seed_id.items():
        trunk_seed_id_to_seed_ids[v].append(k)
    trunk_seed_id_to_seed_ids = dict(trunk_seed_id_to_seed_ids)

    return seed_id_to_trunk_seed_id, trunk_seed_id_to_seed_ids


def get_closest(pc_a, pc_b):
    """
    For each point in pc_a, find the closest point in pc_b
    Returns the distance and index of the closest point in pc_b for each point in pc_a
    Parameters
    ----------
    pc_a : [Mx3]
    pc_b : [Nx3]
    """
    tree = KDTree(pc_b)
    dist, idx = tree.query(pc_a, workers=-1)

    if np.max(idx) >= pc_b.shape[0]:
        raise ValueError("idx is out of range")

    return dist, idx


def skel_id_to_seed_id_mapping(seed_data):
    mapping = {}
    for row in seed_data:
        key = (row["skeleton_id"], row["vertex_id"])
        assert key not in mapping
        mapping[key] = row["seed_id"]
    return mapping


def seed_id_to_row_mapping(seed_data):
    mapping = {}
    for row in seed_data:
        key = row["seed_id"]
        assert key not in mapping
        mapping[key] = row
    return mapping


def weighted_random_sample(lengths: List[int], total_samples: int):
    """
    Determines the number of samples to take from each group based on the length of the group

    Parameters
    ----------
    lengths
    total_samples
    """
    weights = np.array(lengths) / sum(lengths)
    chosen_arrays = np.random.choice(
        len(lengths), size=total_samples, p=weights, replace=True
    )

    sample_counts = np.bincount(chosen_arrays, minlength=len(lengths))
    return sample_counts


from tqdm import tqdm


def sample_points(pc_group, seed_id, count, length):
    return pc_group[str(seed_id)][np.random.choice(length, count, replace=True)]


def sample_path(pc_lengths, pc_group, seed_path, num_points, num_threads):
    # NOTE: limitation
    # since everything is done via feature transform, if the point cloud of another dendrite is closer to trunk of another, it will be included here
    lengths = [pc_lengths[seed_id] for seed_id in seed_path]
    assert sum(lengths) > 0

    sample_counts = weighted_random_sample(lengths, num_points)
    points = []
    points = Parallel(n_jobs=num_threads, backend="threading")(
        delayed(sample_points)(pc_group, seed_id, count, length)
        for seed_id, count, length in zip(seed_path, sample_counts, lengths)
    )
    return np.concatenate(points)


def vertex_path_to_trunk_path(trunk_id, vertex_path, seed_id_to_trunk_seed_id):
    return [
        seed_id_to_trunk_seed_id[(trunk_id, vertex_id)] for vertex_id in vertex_path
    ]


def visualize_batch(pc, trunk_skel, label):
    # assumes anisotropic
    import open3d as o3d

    label = label > 0

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc.astype(float))
    colors = [[0, 1, 0] if lbl == 1 else [0, 0, 1] for lbl in label]
    pcd.colors = o3d.utility.Vector3dVector(colors)
    # pcd.paint_uniform_color([0.5, 0.5, 0.5])

    pcd_trunk = o3d.geometry.PointCloud()
    pcd_trunk.points = o3d.utility.Vector3dVector(trunk_skel.astype(float))
    pcd_trunk.paint_uniform_color([1, 0, 0])

    o3d.visualization.draw_geometries([pcd, pcd_trunk])


class FreSegDataset(Dataset):
    def __init__(
        self,
        mapping_path: str,
        seed_path: str,
        pc_zarr_path: str,
        pc_lengths_path: str,
        path_length: float,
        num_points: int,
        anisotropy: Tuple[float, float, float],
        folds: List[List[int]],
        fold: int,
        is_train: bool,
        num_threads: int,
        transform: Optional[Callable] = None,
    ):
        """
        Initialize the FreSegDataset object with given parameters and load necessary data.
        if fold is -1, use all data


        Parameters
        ----------
        mapping_path : str
        seed_path : str
        pc_zarr_path : str
        path_length : float
            geodesic distance to sample
        num_points : int
            num points to sample
        anisotropy : Tuple[float, float, float]
        folds : List[List[int]]
        fold : int
        is_train : bool
            used to determine which folds to use
        """
        assert -1 <= fold < len(folds)

        self.num_points = num_points
        self.pc_zarr_path = pc_zarr_path
        self.anisotropy = anisotropy
        self.transform = transform
        self.pc_lengths = np.load(pc_lengths_path, allow_pickle=True)["lengths"].item()
        self.num_threads = num_threads

        mapping = np.load(mapping_path)
        seg_to_trunk, trunk_to_segs = read_mappings(mapping)

        seed = np.load(seed_path, allow_pickle=True)
        seed_data, skeletons = seed["seed_data"], seed["skeletons"]

        seed_id_to_trunk_seed_id, trunk_seed_id_to_seed_ids = link_seeds(
            seed_data, seg_to_trunk, trunk_to_segs
        )

        skeletons = skeletons.item()
        self.skeletons = {k: nx_from_skel(v) for k, v in skeletons.items()}

        if fold == -1:
            trunk_ids = list(trunk_to_segs.keys())
        else:
            if is_train:
                trunk_ids = [
                    item
                    for idx, sublist in enumerate(folds)
                    if idx != fold
                    for item in sublist
                ]
            else:
                trunk_ids = folds[fold]

        trunk_ids = sorted(trunk_ids)
        for k in trunk_ids:
            components = list(nx.connected_components(self.skeletons[k]))
            if len(components) > 1:
                component_sizes = [len(component) for component in components]
                print(
                    f"Warning: {k} has {len(components)} connected components with sizes {component_sizes}, taking largest"
                )
                self.skeletons[k] = self.skeletons[k].subgraph(max(components, key=len))

        self.seed_id_to_row = seed_id_to_row_mapping(seed_data)

        self.spanning_paths = {}

        skeleton_id_to_seed_id = skel_id_to_seed_id_mapping(seed_data)
        for trunk_id in trunk_ids:
            proposed_paths = get_spanning_paths(self.skeletons[trunk_id], path_length)
            trunk_paths = [
                vertex_path_to_trunk_path(trunk_id, path, skeleton_id_to_seed_id)
                for path in proposed_paths
            ]
            lengths = []
            seed_paths = []
            for path in trunk_paths:
                seed_paths.append([])
                for trunk_seed_id in path:
                    seed_paths[-1].extend(trunk_seed_id_to_seed_ids[trunk_seed_id])
                lengths.append(
                    sum([self.pc_lengths[seed_id] for seed_id in seed_paths[-1]])
                )

            num_empty = sum([length == 0 for length in lengths])
            if num_empty > 0:
                print(
                    f"Warning: {trunk_id} has {num_empty} 0 length paths, skipping these"
                )

            self.spanning_paths[trunk_id] = [
                {"seed_path": seed_paths[i], "trunk_path": trunk_paths[i]}
                for i in range(len(trunk_paths))
                if lengths[i] > 0
            ]

    def __len__(self):
        return sum(len(v) for v in self.spanning_paths.values())

    def get_path_by_idx(self, idx):
        # given an index, return the trunk_id and path
        for k in sorted(self.spanning_paths.keys()):
            v = self.spanning_paths[k]
            if idx < len(v):
                return k, v[idx]
            idx -= len(v)
        raise IndexError(f"Index {idx} out of range")

    def __getitem__(self, idx):
        trunk_id, data = self.get_path_by_idx(idx)
        seed_path, trunk_path = data["seed_path"], data["trunk_path"]

        # contains ANISOTROPIC coords
        pc_group = zarr.open_group(self.pc_zarr_path, mode="r")

        # NOTE: zyx are in anisotropic coords
        points = sample_path(
            self.pc_lengths, pc_group, seed_path, self.num_points, self.num_threads
        )
        assert points.shape[0] == self.num_points

        trunk_points = [self.seed_id_to_row[seed_id] for seed_id in trunk_path]
        trunk_pc = np.stack(
            [
                np.array([point[f"seed_coord_{c}"] for c in "zyx"])
                for point in trunk_points
            ],
            axis=0,
        )

        pc = np.stack([points["z"], points["y"], points["x"]], axis=1) * np.array(
            self.anisotropy
        )
        label = points["seg"]

        if self.transform is None:
            return trunk_id, pc, trunk_pc, label
        else:
            return self.transform(trunk_id, pc, trunk_pc, label)


def main(conf):
    """
    seg_id: canonical labelling for each segment trunk/spines
    trunk_id: canonical labelling for each trunk (subset of seg_id)

    skeleton_ids: for each vertex, which seg_id
    vertex_ids: for each vertex, which index in the skeleton (multiple 0s)
    seed_coord_z/y/x: for each vertex, the ISOTROPIC coordinates
    seed_id: canonical vertex

    skeletons: cloudvolume skeletons {seg_id: skel}
    max_radius: max radius of all skeletons
    """
    if mp.is_remote:
        with temp_seed(0):
            dataset = FreSegDataset(
                conf.data.mapping,
                conf.data.seed,
                conf.data.pc_zarr,
                conf.data.pc_lengths,
                conf.dataloader.path_length,
                conf.dataloader.num_points,
                conf.anisotropy,
                conf.dataloader.folds,
                fold=-1,
                is_train=True,
                num_threads=conf.dataloader.num_threads,
            )
        # choose random idx
        # idx = 72
        idx = np.random.randint(len(dataset))
        trunk_id, pc, trunk_pc, label = dataset[idx]
        skeleton = np.load(conf.data.seed, allow_pickle=True)["skeletons"].item()[
            trunk_id
        ]
        mp.save((trunk_id, pc, trunk_pc, label, skeleton))
    else:
        trunk_id, pc, trunk_pc, label, skeleton = mp.load()
        trunk_pc = skeleton.vertices
        visualize_batch(pc, trunk_pc, label)


if __name__ == "__main__":
    from magicpickle import MagicPickle

    np.random.seed(42)
    with MagicPickle("think-jason") as mp:
        if mp.is_remote:
            conf = get_conf()
            mp.save(conf)
            main(conf)
        else:
            conf = mp.load()
            main(conf)
