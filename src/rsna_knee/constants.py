"""The twelve findings, and the vocabulary shared across the package."""
from __future__ import annotations

from pathlib import Path
from torch import nn
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader
from typing import Any, Sequence
from typing import Iterable
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
import re
import time
import torch
import unicodedata
import yaml


DEFAULT_SEED = 2026


TARGETS = [
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
]


CONSTRUCTION_SEED_OFFSET = 11


LOADER_SEED_OFFSET = 29


PLANES = ("Sagittal", "Coronal", "Axial")


PLANE_TO_ID = {"Sagittal": 1, "Coronal": 2, "Axial": 3}


# Capitalised to match the convention used by `load_series_csv`.
PLANES = ("Sagittal", "Coronal", "Axial")


DUAL_STREAMS = [
    "sagittal_fluid",
    "sagittal_structural",
    "coronal_fluid",
    "coronal_structural",
    "axial_fluid",
    "axial_structural",
]


N_TARGETS = len(TARGETS)

