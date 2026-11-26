"""Train a victim on final CEBS masks using the native masked-loss trainer."""
from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from cebs import runtime as rt
from cebs.artifacts import RunLock, ahash, dump, read, sha
from cebs.detection import PERIODS, read_arrays, validate_inputs
from cebs.rng import seed_all

ROOT=Path(__file__).resolve().parents[1]


def resolve(value):
    path=Path(value)
    return path.resolve() if path.is_absolute() else (ROOT/path).resolve()


def load_config(path):
    cfg=yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    if cfg.get('dataset') not in PERIODS or cfg.get('victim') not in rt.ARCHITECTURES:
        raise ValueError('Choose a supported dataset and victim architecture')
    if (cfg.get('history',12),cfg.get('horizon',12))!=(12,12):
        raise ValueError('This implementation uses history=horizon=12')
    for key in ('observed','trusted','proxy','validation','output'):
        if not cfg.get(key):
            raise ValueError('Missing configuration field: '+key)
    output=resolve(cfg['output'])
    if not output.is_relative_to(ROOT/'outputs') or output==ROOT/'outputs':
        raise ValueError('Use a dedicated subdirectory under outputs/')
    return cfg


def preflight(cfg):
    observed=read_arrays(resolve(cfg['observed']))
    trusted=read_arrays(resolve(cfg['trusted']))
    geometry=validate_inputs(observed,trusted)
    validation=read_arrays(resolve(cfg['validation']))
    series=validation.get('series')
    if series is None or series.ndim!=2 or len(series)<24 or series.shape[1]!=geometry['channels'] or not np.isfinite(series).all():
        raise ValueError('Validation needs finite original-unit series with matching channels and length >=24')
    if 'marks' in validation and (validation['marks'].shape!=(len(series),4) or not np.isfinite(validation['marks']).all()):
        raise ValueError('Validation marks must have shape (validation_points,4)')
    offsets=trusted['starts']-len(observed['series'])
    if offsets.min()<0 or offsets.max()+24>len(series):
        raise ValueError('Trusted blocks are outside the declared validation segment')
    expected_x=(series[offsets[:,None]+np.arange(12)]-float(trusted['mean']))/float(trusted['std'])
    expected_y=series[offsets[:,None]+12+np.arange(12)]
    if not np.allclose(expected_x,trusted['x'],rtol=1e-5,atol=1e-5) or not np.array_equal(expected_y,trusted['y']):
        raise ValueError('Trusted records do not match their declared validation indices/normalization')
    for field in ('proxy','channel_proxy','initial_state','evaluation'):
        if cfg.get(field) and not resolve(cfg[field]).is_file():
            raise FileNotFoundError(resolve(cfg[field]))
    return observed,trusted,validation,geometry


def validate_masks(hard,local,shape):
    n,h,c=shape
    if hard.dtype!=bool or local.dtype!=bool or hard.shape!=(n,) or local.shape!=(n,h,c):
        raise ValueError('Final masks must be Boolean and match window/horizon/channel geometry')
    if hard.all() or local[hard].any() or local[~hard].all((1,2)).any():
        raise ValueError('Final hard/local masks overlap or leave no usable supervision')


def masks_for_training(cfg,observed,trusted):
    folder=resolve(cfg['output'])/'detection'
    complete=read(folder/'complete.json')
    identity=read(folder/'input_identity.json')
    path=folder/'final_masks.npz'
    if complete.get('state')!='complete_fresh_cebs' or sha(path)!=complete['mask_file_sha256']:
        raise RuntimeError('Final masks are missing, incomplete, or changed')
    for key in ('x','y','starts','block_ids','mean','std'):
        if ahash(trusted[key])!=identity['reference_arrays'][key]:
            raise RuntimeError('Trusted reference differs from detection')
    reference_marks=trusted.get('marks',np.zeros((len(trusted['starts']),12,4),np.float32))
    if ahash(reference_marks)!=identity['reference_marks_sha256']:
        raise RuntimeError('Trusted time features differ from detection')
    normalized=dict(observed)
    if 'marks' not in normalized:
        normalized['marks']=np.zeros((len(observed['series']),4),np.float32)
    for key in ('series','marks'):
        if ahash(normalized[key])!=identity['observed_arrays'][key]:
            raise RuntimeError('Observed training records differ from detection')
    if sha(resolve(cfg['proxy']))!=identity['checkpoint_sha256']:
        raise RuntimeError('Infected proxy differs from detection')
    channel=resolve(cfg['channel_proxy']) if cfg.get('channel_proxy') else resolve(cfg['proxy'])
    if sha(channel)!=identity['channel_checkpoint_sha256']:
        raise RuntimeError('Channel-evidence proxy differs from detection')
    if identity['dataset']!=cfg['dataset'] or identity['period']!=cfg.get('period',PERIODS[cfg['dataset']]):
        raise RuntimeError('Dataset/period differs from detection')
    arrays=read_arrays(path)
    hard,local=arrays['hard_mask'],arrays['local_mask']
    validate_masks(hard,local,(len(observed['series'])-23,12,observed['series'].shape[1]))
    return hard,local


def train(cfg,config_path,device='cuda'):
    observed,trusted,validation,geometry=preflight(cfg)
    hard,local=masks_for_training(cfg,observed,trusted)
    models=yaml.safe_load((ROOT/'configs/models.yaml').read_text(encoding='utf-8'))
    victim=cfg['victim']
    channels=geometry['channels']
    model_cfg=dict(models[victim],seq_len=12,pred_len=12,enc_in=channels,dec_in=channels,c_out=channels)
    points=len(observed['series'])
    spec=dict(case=cfg['dataset'].lower()+'_cebs',mean=float(trusted['mean']),std=float(trusted['std']),
              channels=channels,train_end=points,validation_end=points+len(validation['series']),
              poison=resolve(cfg['observed']))
    if cfg.get('initial_state'):
        if victim!='FEDformer':
            raise ValueError('The native initial-state adapter is FEDformer-specific')
        spec['fedformer_initial_state']=resolve(cfg['initial_state'])
    full=np.concatenate([observed['series'],validation['series']])
    marks=np.concatenate([observed.get('marks',np.zeros((points,4),np.float32)),
                          validation.get('marks',np.zeros((len(validation['series']),4),np.float32))])
    initial=rt._initial_model(victim,model_cfg,spec,device)
    initial_hash=rt.model_hash(initial)
    del initial
    gc.collect()
    sources=sorted((ROOT/'cebs').rglob('*.py'))+sorted((ROOT/'models').rglob('*.py'))+[ROOT/'configs/models.yaml']
    implementation=hashlib.sha256(json.dumps({p.relative_to(ROOT).as_posix():sha(p) for p in sources},sort_keys=True).encode()).hexdigest()
    validation_sha=sha(resolve(cfg['validation']))
    configuration_sha=hashlib.sha256((sha(config_path)+validation_sha+
        (sha(resolve(cfg['initial_state'])) if cfg.get('initial_state') else 'native-seed-1')).encode()).hexdigest()
    protocol=dict(history=12,horizon=12,seed=1,epochs=int(cfg.get('epochs',100)),
                  batch_size=int(cfg.get('batch_size',64)),learning_rate=float(cfg.get('learning_rate',.0001)),
                  model_config=model_cfg,initial_hash=initial_hash,hard_mask_sha256=ahash(hard),local_mask_sha256=ahash(local),
                  config_sha256=configuration_sha,implementation_sha256=implementation,runtime_environment=rt.runtime_environment(),
                  validation_sha256=validation_sha,checkpoint_selection='minimum clean validation MAE; earliest tie',
                  selection_uses_test_set=False,detection_recalculated_during_training=False)
    if protocol['epochs']<1 or protocol['batch_size']<1 or protocol['learning_rate']<=0:
        raise ValueError('Training epochs, batch size and learning rate must be positive')
    folder=resolve(cfg['output'])/'training'/victim.lower()
    with RunLock(folder/'run.lock'):
        contract=folder/'protocol.json'
        if contract.exists() and read(contract)!=protocol:
            raise RuntimeError('Training output belongs to another protocol; use a new output')
        dump(contract,protocol)
        checkpoint=rt._train(folder,victim,model_cfg,spec,protocol,hard,local,full,marks,device)
        result=dict(state='complete',checkpoint_sha256=sha(checkpoint),victim=victim,
                    hard_mask_sha256=ahash(hard),local_mask_sha256=ahash(local),protocol=protocol)
        if cfg.get('evaluation'):
            spec['evaluation']=resolve(cfg['evaluation'])
            seed_all(1)
            result['metrics']=rt._evaluate(checkpoint,victim,model_cfg,spec,protocol['batch_size'],device)
        dump(folder/'complete.json',result)
    return result
