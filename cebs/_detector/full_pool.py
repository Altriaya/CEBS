"""CEBS core numerical implementation."""
from __future__ import annotations

import ast
import argparse
import gc
import hashlib
import json
import shutil
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import yaml
from threadpoolctl import threadpool_limits
FORMAL = Path(__file__).resolve().parents[2]
from cebs.calibration import NumericalSources, adapt_recurrence, cell_ranks, choose_event, cross_rank, dump, event_scores, fit_mask, held_risk, normal_routing, physical_cut, physical_ranks, rank, safe_json, supervision_scores
from cebs.reference import REQUIRED_FULL_PIPELINE_STAGES, fit_readout, nested_normality, require_complete_pipeline, require_physical_disjoint
from cebs._detector.predictor import ProbePredictor
from cebs._detector.io import ahash, digest, read_npz
from cebs._detector.reference import ReferenceEvidence
from cebs._detector.pipeline import numerical_helpers

class FullPoolEvidence(ReferenceEvidence):

    def __init__(self, *args):
        super().__init__(*args)
        self.helpers = numerical_helpers(self.root, self.folder, -1)
        normal = self.helpers.normal
        self.extra_src = NumericalSources(self.root, self.folder / 'extra_numerical_sources')
        self.normal_helper = self.extra_src.load('normality.py', ['trusted_scores'], dict(features=normal.features, scaled=normal.scaled, knn_distance=normal.knn_distance))
