"""Verify canonical base records using output source/row provenance."""
import numpy as np
from gs_core import *
def verify_preservation(baseline,model,provenance,source_id=0,changed_rows=None,allowed_columns=(),required_rows=None):
 with SplatFile(baseline)as a,SplatFile(model)as b,np.load(provenance)as p:
  sid=p['source_id'];rows=p['source_row']
  if sid.shape!=(b.n,)or rows.shape!=(b.n,):raise ValueError('Provenance length mismatch')
  degree=b.degree
  if degree<a.degree:raise ValueError('Output truncates baseline SH')
  idx=np.flatnonzero(sid==source_id)
  if not len(idx):raise ValueError('No baseline records retained')
  src=rows[idx]
  if np.any(src<0)or np.any(src>=a.n):raise ValueError('Invalid source row')
  if len(np.unique(src))!=len(src):raise ValueError('Baseline source row duplicated')
  changed=np.asarray([]if changed_rows is None else changed_rows,'i8');allowed=np.asarray(list(allowed_columns),'i8');bad=[];unchanged=0
  base=a.rows();out=b.rows()
  for j in range(0,len(idx),32768):
   oi=idx[j:j+32768];ri=src[j:j+32768];expected=promote(base[ri],degree);got=out[oi];is_changed=np.isin(ri,changed);mask=np.ones(expected.shape,bool)
   if len(allowed):mask[np.ix_(is_changed,np.isin(np.arange(expected.shape[1]),allowed))]=False
   mismatch=np.any((got!=expected)&mask,axis=1);bad.extend(ri[mismatch].tolist());unchanged+=int(sum(~is_changed&~mismatch))
  missing=[]
  if required_rows is not None:missing=np.setdiff1d(np.asarray(required_rows,'i8'),src).tolist()
  return dict(all_preserved=not bad and not missing,canonical_scope='positions, log-scales, opacity, wxyz quaternion, DC and real SH; not dropped auxiliary PLY fields',retained_baseline_rows=len(idx),exact_unchanged_rows=unchanged,changed_scope_rows=int(sum(np.isin(src,changed))),bad_rows=bad[:100],bad_count=len(bad),missing_required_rows=missing[:100],missing_required_count=len(missing),baseline_sha256=sha256(baseline),model_sha256=sha256(model))

