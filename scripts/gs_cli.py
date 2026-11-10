#!/usr/bin/env python3
"""Portable CLI. Run --help and each subcommand's --help for interfaces."""
from pathlib import Path
import argparse,json,sys
import numpy as np
from gs_core import *
def emit(d):print(json.dumps(d,ensure_ascii=False,indent=2))
def new_path(p):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists():raise FileExistsError(p)
 return p
def matrix(path):
 d=load_json(path);return np.asarray(d.get('matrix',d)if isinstance(d,dict)else d,'f8')
def reader_args(p):
 p.add_argument('--opacity',choices=['logit','alpha'],default='logit');p.add_argument('--scale',choices=['log','linear'],default='log');p.add_argument('--quaternion',choices=['wxyz','xyzw'],default='wxyz');p.add_argument('--allow-cast',action='store_true');p.add_argument('--allow-extras',action='store_true')
def opts(a):return dict(opacity=a.opacity,scale=a.scale,quaternion=a.quaternion,allow_cast=a.allow_cast,allow_extras=a.allow_extras)
def main(argv=None):
 parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='cmd',required=True)
 p=sub.add_parser('inspect');p.add_argument('input');reader_args(p)
 p=sub.add_parser('validate');p.add_argument('input');p.add_argument('--report');reader_args(p)
 p=sub.add_parser('normalize');p.add_argument('input');p.add_argument('output');reader_args(p)
 p=sub.add_parser('fit-pairs');p.add_argument('--pairs',required=True);p.add_argument('--threshold',required=True,type=float);p.add_argument('--output',required=True);p.add_argument('--iterations',type=int,default=2000)
 p=sub.add_parser('icp');p.add_argument('--source',required=True);p.add_argument('--target',required=True);p.add_argument('--initial',required=True);p.add_argument('--voxel',required=True,type=float);p.add_argument('--max-distance',required=True,type=float);p.add_argument('--output',required=True);p.add_argument('--source-mask');p.add_argument('--target-mask')
 p=sub.add_parser('transform');p.add_argument('input');p.add_argument('output');p.add_argument('--matrix',required=True);reader_args(p)
 p=sub.add_parser('fuse');p.add_argument('--manifest',required=True)
 p=sub.add_parser('shell');p.add_argument('--config',required=True);p.add_argument('--output',required=True)
 p=sub.add_parser('material');p.add_argument('input');p.add_argument('output');p.add_argument('--indices',required=True);p.add_argument('--anchor-camera');p.add_argument('--gain',nargs=3,type=float,default=[1,1,1]);p.add_argument('--alpha-exponent',type=float,default=1);p.add_argument('--expected-sha')
 p=sub.add_parser('render');p.add_argument('input');p.add_argument('--camera',required=True);p.add_argument('--output',required=True)
 p=sub.add_parser('trace');p.add_argument('input');p.add_argument('--camera',required=True);p.add_argument('--pixel-mask',required=True);p.add_argument('--point-mask');p.add_argument('--out-prefix',required=True);p.add_argument('--min-score',type=float,default=0)
 p=sub.add_parser('audit');p.add_argument('--config',required=True);p.add_argument('--output-dir',required=True)
 p=sub.add_parser('preserve');p.add_argument('--baseline',required=True);p.add_argument('--model',required=True);p.add_argument('--provenance',required=True);p.add_argument('--source-id',type=int,default=0);p.add_argument('--changed-rows');p.add_argument('--allowed-columns',nargs='*',type=int,default=[]);p.add_argument('--required-rows');p.add_argument('--report',required=True)
 p=sub.add_parser('colmap-camera');p.add_argument('--sparse',required=True);p.add_argument('--image',required=True);p.add_argument('--transform');p.add_argument('--output',required=True)
 p=sub.add_parser('match-colmap');p.add_argument('--source-sparse',required=True);p.add_argument('--source-db',required=True);p.add_argument('--target-sparse',required=True);p.add_argument('--target-db',required=True);p.add_argument('--every',type=int,default=3);p.add_argument('--ratio',type=float,default=.75);p.add_argument('--output',required=True)
 a=parser.parse_args(argv)
 if a.cmd in ['inspect','validate']:
  with SplatFile(a.input,**opts(a))as f:
   report=f.info()
   if a.cmd=='validate':
    count=0
    for _,d in f.chunks():validate_rows(d);count+=len(d)
    report.update(valid=True,validated_rows=count,sha256=sha256(f.path))
  if getattr(a,'report',None):save_json(new_path(a.report),report)
  emit(report);return 0
 if a.cmd in ['normalize','transform']:
  from gs_pipeline import transform_file
  M=np.eye(4)if a.cmd=='normalize'else matrix(a.matrix);emit(transform_file(a.input,a.output,M,opts(a)));return 0
 if a.cmd=='fit-pairs':
  d=load_json(a.pairs);M,report=fit_pairs(d['source'],d['target'],a.threshold,iterations=a.iterations,holdout_indices=d.get('trusted_holdout_indices'))
  trusted=d.get('trusted_holdout_indices')is not None;accepted=(report['holdout']is not None and report['holdout']['p95']<=a.threshold)if trusted else None
  report.update(matrix=M.tolist(),accepted=accepted,source_controls=str(Path(a.pairs).resolve()),note='A tentative-match holdout is not independently verified ground truth. Inspect structural alignment in held-out views.')
  save_json(new_path(a.output),report);emit({k:v for k,v in report.items()if k not in ['errors','holdout_indices']});return 2 if accepted is False else 0
 if a.cmd=='icp':
  from gs_registration import refine_icp
  M,report=refine_icp(a.source,a.target,matrix(a.initial),a.voxel,a.max_distance,source_mask=a.source_mask,target_mask=a.target_mask);report['matrix']=M.tolist();save_json(new_path(a.output),report);emit(report);return 0
 if a.cmd=='fuse':
  from gs_pipeline import fuse_manifest
  emit(fuse_manifest(a.manifest));return 0
 if a.cmd=='shell':
  from gs_pipeline import shell_file
  emit(shell_file(a.config,a.output));return 0
 if a.cmd=='material':
  from gs_pipeline import material_file
  if a.expected_sha and sha256(a.input)!=a.expected_sha:raise ValueError('Selection belongs to a different model version')
  cam=load_json(a.anchor_camera)if a.anchor_camera else None
  emit(material_file(a.input,a.output,np.load(a.indices),cam['position']if cam else None,a.gain,a.alpha_exponent));return 0
 if a.cmd in ['render','trace']:
  from PIL import Image
  from gs_render import load_model,render_image,trace_visibility
  model=load_model(a.input);cam=load_json(a.camera)
  if a.cmd=='render':
   path=new_path(a.output);render_image(model,cam).save(path);save_json(str(path)+'.json',dict(model_sha256=sha256(a.input),camera=cam,renderer='CPU SH/EWA center-depth reference approximation'));emit(dict(image=str(path)));return 0
  pix=np.asarray(Image.open(a.pixel_mask).convert('L'))>0;pm=None
  if a.point_mask:
   v=np.load(a.point_mask)
   if v.dtype==bool:pm=v
   else:pm=np.zeros(len(model),bool);pm[np.asarray(v,'i8')]=True
  score=trace_visibility(model,cam,pix,pm);rank=np.flatnonzero(score>a.min_score);rank=rank[np.argsort(score[rank])[::-1]];prefix=Path(a.out_prefix);np.save(new_path(str(prefix)+'.scores.npy'),score);np.save(new_path(str(prefix)+'.indices.npy'),rank);report=dict(model=str(Path(a.input).resolve()),model_sha256=sha256(a.input),camera=cam,selected_pixels=int(pix.sum()),positive_contributors=len(rank),min_score=a.min_score,warning='Visible contribution is not semantic object identity. Inspect the selected source points before editing.')
  save_json(new_path(str(prefix)+'.trace.json'),report);emit(report);return 0
 if a.cmd=='audit':
  from gs_audit import run_audit
  report=run_audit(a.config,a.output_dir);emit(dict(all_passed=report['all_passed'],cases=len(report['results']),report=str(Path(a.output_dir)/'audit.json')));return 0 if report['all_passed']else 2
 if a.cmd=='preserve':
  from gs_verify import verify_preservation
  report=verify_preservation(a.baseline,a.model,a.provenance,a.source_id,np.load(a.changed_rows)if a.changed_rows else None,a.allowed_columns,np.load(a.required_rows)if a.required_rows else None);save_json(new_path(a.report),report);emit(report);return 0 if report['all_preserved']else 2
 if a.cmd=='colmap-camera':
  from colmap_io import camera_json
  report=camera_json(a.sparse,a.image,matrix(a.transform)if a.transform else None);save_json(new_path(a.output),report);emit(report);return 0
 if a.cmd=='match-colmap':
  from colmap_io import point_features,match_features
  one=point_features(a.source_sparse,a.source_db,a.every);two=point_features(a.target_sparse,a.target_db,a.every);matches=match_features(one,two,a.ratio);save_json(new_path(a.output),matches);emit(dict(tentative_pairs=len(matches['source']),output=a.output));return 0
if __name__=='__main__':
 try:sys.exit(main())
 except (ValueError,OSError,KeyError)as e:print(f'ERROR: {e}',file=sys.stderr);sys.exit(2)

