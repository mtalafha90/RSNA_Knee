"""A loader run under a deliberately small open-file limit.

Run as a script by `test_worker_file_descriptors.py`, in its own process,
because it lowers `RLIMIT_NOFILE` and that cannot be undone for the caller.

    python tests/_fd_probe.py fixed   256    # uses the package's worker setup
    python tests/_fd_probe.py unfixed 256    # seeds only, as before the fix

Prints one line and exits 0 on success, 1 on the failure being reproduced.
"""

from __future__ import annotations

import random
import resource
import sys

import torch
from torch.utils.data import DataLoader, Dataset

from rsna_knee.runtime import seed_worker

ARM = sys.argv[1] if len(sys.argv) > 1 else "fixed"


class ManyTensorsPerItem(Dataset):
    """One item is a list of small tensors, shaped like one study's series.

    The count is what matters, not the size: every separate tensor is its own
    shared allocation, and under the default strategy its own file descriptor.
    """

    def __len__(self) -> int:
        return 200

    def __getitem__(self, index: int) -> dict:
        return {
            "study_uid": f"study-{index}",
            "volumes": [torch.zeros(4, 3, 32, 32) for _ in range(8)],
            "target": torch.zeros(12),
            "weight": torch.zeros(12),
        }


def collate(items):
    return list(items)


def seed_only(worker_id: int) -> None:
    """What `seed_worker` did before the fix: seed, and share by descriptor."""
    seed = int(torch.initial_seed() % (2**32))
    random.seed(seed)
    torch.set_num_threads(1)


def main() -> int:
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 256
    _soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (limit, hard))

    loader = DataLoader(
        ManyTensorsPerItem(),
        batch_size=2,
        num_workers=2,
        prefetch_factor=4,
        persistent_workers=True,
        collate_fn=collate,
        worker_init_fn=seed_worker if ARM == "fixed" else seed_only,
        multiprocessing_context="spawn",
    )

    # Hold a few batches, as the training loop does while the GPU is busy: the
    # descriptors stay open for as long as anything references the tensors.
    held: list = []
    try:
        for batch in loader:
            held.append(batch)
            if len(held) > 20:
                held.pop(0)
    except Exception as error:  # noqa: BLE001
        print(f"{ARM}: FAIL {type(error).__name__}: {error}")
        return 1
    print(f"{ARM}: OK limit={limit} strategy={torch.multiprocessing.get_sharing_strategy()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
