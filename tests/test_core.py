"""Lightweight checks for the core defense; no experiment sweeps or GPU jobs."""
import ast
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


class CoreTests(unittest.TestCase):
    def test_source_compiles(self):
        for path in ROOT.rglob('*.py'):
            compile(path.read_text(encoding='utf-8-sig'),str(path),'exec')

    def test_no_experimental_entrypoints(self):
        for name in ('experiments','attacks','results','compat','provenance','pilot_b','pilot_reference_ablation'):
            self.assertFalse((ROOT/name).exists(),name)
        for path in (ROOT/'cebs').rglob('*.py'):
            for node in ast.walk(ast.parse(path.read_text(encoding='utf-8-sig'))):
                if isinstance(node,ast.ImportFrom) and node.module:
                    self.assertFalse(node.module.startswith(('experiments.','pilot_')),node.module)

    def test_native_model_imports(self):
        for name in ('FEDformer','TimesNet'):
            self.assertTrue(hasattr(importlib.import_module('models.predictors.'+name),'Model'))

    def test_packaged_numerical_helpers(self):
        from cebs._detector.pipeline import numerical_helpers
        with tempfile.TemporaryDirectory(prefix='cebs_core_') as temp:
            root=Path(temp)
            h=numerical_helpers(root,root/'derived',-1)
            h.src.verify_originals()
            self.assertEqual(len(h.src.records),12)
            np.testing.assert_allclose(h.top.top_mean_rows(np.array([[1.,2.,3.],[4.,5.,6.]])),[3.,6.])

    def test_donor_is_neutral_if_unreliable(self):
        from cebs._numerics.input_selection import node_evidence
        score=node_evidence(np.array([.96,.96,.94]),np.array([0.,1.,1.]),np.array([False,True,True]),.95,.95,1.)
        np.testing.assert_allclose(score,[1.,2.,0.])

    def test_input_geometry_and_budget(self):
        from cebs.detection import validate_inputs
        observed=dict(series=np.zeros((6000,2),np.float32))
        ref=dict(x=np.zeros((12,12,2),np.float32),y=np.zeros((12,12,2),np.float32),
                 starts=np.arange(12,dtype=np.int64)*30+6000,block_ids=np.arange(12,dtype=np.int64),
                 mean=np.asarray(0.),std=np.asarray(1.))
        self.assertEqual(validate_inputs(observed,ref)['trusted_raw_points'],288)
        with self.assertRaises(ValueError):
            validate_inputs(dict(series=np.zeros((5000,2),np.float32)),ref)

    def test_mask_geometry(self):
        from cebs.training import validate_masks
        hard,local=np.zeros(10,bool),np.zeros((10,12,2),bool)
        validate_masks(hard,local,(10,12,2))
        hard[0]=True;local[0,0,0]=True
        with self.assertRaises(ValueError):validate_masks(hard,local,(10,12,2))

    def test_masked_window_projection(self):
        import torch
        from cebs.runtime import MaskedWindows
        series=np.arange(60,dtype=np.float32).reshape(30,2)
        hard,local=np.zeros(7,bool),np.zeros((7,12,2),bool)
        hard[0]=True;local[1,2,0]=True
        data=MaskedWindows(series,np.zeros((30,4),np.float32),hard,local,'cpu')
        x,y,marks,keep=data.batch(torch.tensor([0]))
        np.testing.assert_array_equal(x[0].numpy(),series[1:13])
        np.testing.assert_array_equal(y[0].numpy(),series[13:25])
        self.assertFalse(bool(keep[0,2,0]))

    def test_core_config(self):
        from cebs.training import load_config
        self.assertEqual(load_config(ROOT/'configs/pems03_backtime.yaml')['victim'],'FEDformer')

    def test_training_rejects_wrong_detection_identity(self):
        from cebs.training import masks_for_training
        with tempfile.TemporaryDirectory(prefix='cebs_masks_') as temp:
            out=Path(temp);folder=out/'detection';folder.mkdir()
            hard,local=np.zeros(7,bool),np.zeros((7,12,2),bool)
            np.savez_compressed(folder/'final_masks.npz',hard_mask=hard,local_mask=local)
            digest=hashlib.sha256((folder/'final_masks.npz').read_bytes()).hexdigest()
            (folder/'complete.json').write_text(json.dumps(dict(state='complete_fresh_cebs',mask_file_sha256=digest)))
            (folder/'input_identity.json').write_text(json.dumps(dict(reference_arrays=dict(x='incorrect'))))
            with self.assertRaises(RuntimeError):
                masks_for_training(dict(output=str(out)),{},dict(x=np.zeros(1)))

    def test_cli_help(self):
        result=subprocess.run([sys.executable,'-B','defense_cebs.py','--help'],cwd=ROOT,capture_output=True,text=True,timeout=30)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertNotIn('ablation',result.stdout.lower())


if __name__=='__main__':unittest.main()
