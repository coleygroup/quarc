from __future__ import annotations

import warnings
from typing import Iterable, NamedTuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from chemprop.data import BatchMolGraph
from chemprop.data.collate import collate_multicomponent
from chemprop.data.datasets import MulticomponentDataset
from chemprop.data.samplers import ClassBalanceSampler, SeededSampler

from quarc.data.gnn_datasets import Datum_agent


class TrainingBatch_agent(NamedTuple):
    """Match Chemprop's TrainingBatch with added agent input."""

    a_input: Tensor
    bmg: BatchMolGraph
    V_d: Tensor | None
    X_d: Tensor | None
    Y: Tensor | None
    w: Tensor
    lt_mask: Tensor | None
    gt_mask: Tensor | None


def collate_batch_agent(
    batch: Iterable[Datum_agent], classification: bool = False
) -> TrainingBatch_agent:
    """Match Chemprop's collate_batch with added agent input."""
    a_input, mgs, V_ds, x_ds, ys, weights, lt_masks, gt_masks = zip(*batch)

    return TrainingBatch_agent(
        torch.from_numpy(np.vstack(a_input)).float(),
        BatchMolGraph(mgs),
        None if V_ds[0] is None else torch.from_numpy(np.concatenate(V_ds)).float(),
        None if x_ds[0] is None else torch.from_numpy(np.array(x_ds)).float(),
        (
            None
            if ys[0] is None
            else (
                torch.from_numpy(np.array(ys)).long()
                if classification
                else torch.from_numpy(np.array(ys)).float()
            )
        ),
        torch.tensor(weights, dtype=torch.float).unsqueeze(1),
        None if lt_masks[0] is None else torch.from_numpy(np.array(lt_masks)),
        None if gt_masks[0] is None else torch.from_numpy(np.array(gt_masks)),
    )


def build_dataloader_agent(
    dataset,
    batch_size: int = 64,
    num_workers: int = 0,
    class_balance: bool = False,
    seed: int | None = None,
    shuffle: bool = True,  # controls DistributedSampler shuffling
    distributed: bool = False,
    classification: bool = False,
    **kwargs,
):
    """Match chemprop's build_dataloader, uses collate_batch_agent to account for agent."""
    if distributed:
        sampler = DistributedSampler(
            dataset, shuffle=shuffle, seed=seed if seed is not None else 0
        )
        loader_shuffle = False
    elif class_balance:
        sampler = ClassBalanceSampler(dataset.Y, seed, shuffle)
        loader_shuffle = False
    elif shuffle and seed is not None:
        sampler = SeededSampler(len(dataset), seed)
        loader_shuffle = False
    else:
        sampler = None
        loader_shuffle = shuffle

    if isinstance(dataset, MulticomponentDataset):
        collate_fn = collate_multicomponent
    else:

        def collate_fn(x):
            return collate_batch_agent(x, classification=classification)

    if len(dataset) % batch_size == 1:
        warnings.warn(
            f"Dropping last batch of size 1 to avoid issues with batch normalization \
            (dataset size = {len(dataset)}, batch_size = {batch_size})",
            stacklevel=2,
        )
        drop_last = True
    else:
        drop_last = False

    return DataLoader(
        dataset,
        batch_size,
        shuffle=loader_shuffle,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=drop_last,
        **kwargs,
    )
