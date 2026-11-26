"""CEBS: compute final masks, then train the defended forecasting model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from cebs.detection import detect
from cebs.training import ROOT,load_config,preflight,resolve,train


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/pems03_backtime.yaml')
    parser.add_argument('--stage',choices=('check','detect','train','all'),default='all')
    parser.add_argument('--device',default='cuda',help='victim-training device; detection requires CUDA')
    args=parser.parse_args()
    cfg=load_config(args.config)
    if args.stage=='check':
        report=preflight(cfg)[-1]
        report.update(state='inputs_checked',training_started=False,detector_started=False)
    else:
        report={}
        if args.stage in ('detect','all'):
            report['detection']=detect(resolve(cfg['observed']),resolve(cfg['trusted']),resolve(cfg['proxy']),
                resolve(cfg['output'])/'detection',cfg['dataset'],period=cfg.get('period'),batch=int(cfg.get('detection_batch',64)),
                channel_checkpoint=resolve(cfg['channel_proxy']) if cfg.get('channel_proxy') else None)
        if args.stage in ('train','all'):
            report['training']=train(cfg,args.config.resolve(),args.device)
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
