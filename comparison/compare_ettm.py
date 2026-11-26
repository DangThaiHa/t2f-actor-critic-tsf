"""Run from the prepared t2f-forecasting project with the project on PYTHONPATH."""
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize

from src.data import build_training_set
from src.model import core, logger
from final.experts import NAMES, fit_experts, predict_experts

root = Path.cwd()
out = Path(__file__).resolve().parent
folder = root / 'results/final/full/ETTm_h30'
saved = np.load(folder / 'forecasts.npz')
groups = {str(i): pd.read_csv(root / f'data/final/ETTm{i}.csv', parse_dates=['date']).set_index('date') for i in (1, 2)}
features = ['HUFL','HULL','MUFL','MULL','LUFL','LULL']
times = groups['1'].index
assert times.equals(groups['2'].index)
v, t = int(.7*len(times)), int(.8*len(times))
cut = (v+t)//2
origins = np.arange(t,len(times)-29,30)
train,mu,sd,_ = build_training_set(groups,'OT',v,512,'level')
rows = []
for station,g in groups.items():
    idx = saved['series'].astype(str)==station
    assert np.array_equal(saved['origin'][idx],times[origins-1].values)
    stamps = times.values[origins[:,None]+np.arange(30)]
    actual = g.OT.to_numpy()[origins[:,None]+np.arange(30)]
    np.testing.assert_allclose(actual,saved['actual'][idx],rtol=1e-6)
    rows.append(pd.DataFrame(dict(station=station,origin=np.repeat(times[origins-1],30),timestamp=stamps.ravel(),step=np.tile(np.arange(1,31),len(origins)),actual=actual.ravel(),stack_full=saved['stacked'][idx].ravel())))
data=pd.concat(rows,ignore_index=True)
assert len(data)==27840 and not data.duplicated(['station','origin','step']).any()
torch.set_num_threads(1)
for seed in (42,43,44):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model=core.T2f(6,1,16,1,30,35,list(groups),freq=None,logger=logger,batch_size=16,timestamp_col='date',series_id_col='station',epochs=8)
    model.fit(train[features],train[['OT']],train[['HULL']])
    forecasts=[]
    for i in origins:
        history = pd.concat({s:g.iloc[i-32:i][features] for s,g in groups.items()},names=['station','date'])
        prediction=model.predict(history).prediction
        for s in groups:
            p=prediction.loc[s]
            assert p.index.equals(times[i:i+30])
            forecasts.append(pd.DataFrame(dict(station=s,origin=times[i-1],step=np.arange(1,31),pred=p.values*sd[s]+mu[s])))
    col=f't2f_{seed}'
    forecasts=pd.concat(forecasts).rename(columns={'pred':col})
    data=data.merge(forecasts,on=['station','origin','step'],validate='1:1',how='left')
    assert np.isfinite(data[col]).all()
    print(seed,(data[col]-data.actual).abs().mean(),flush=True)
    data.to_csv(out/'predictions.csv.gz',index=False)
errors=[]
test=[]
for s,g in groups.items():
    values=g[['OT']+features].to_numpy(float)
    models,_=fit_experts(values,times,v,30,96,budget=512)
    idx=np.arange(v,cut-29,15)
    scale=values[:v,0].std()
    pred,_=predict_experts(models,values,times,idx,30,96,scale)
    actual=values[idx[:,None]+np.arange(30),0]
    errors.append(((pred-actual[:,:,None])/scale).reshape(-1,len(NAMES)))
    pred,_=predict_experts(models,values,times,origins,30,96,scale)
    test.append(pred.reshape(-1,len(NAMES)))
e=np.concatenate(errors)
def loss(w):
    r=e@w
    return np.abs(r).mean(),np.sign(r)@e/len(r)
opt=minimize(loss,np.full(6,1/6),jac=True,method='SLSQP',bounds=[(0,1)]*6,constraints=[dict(type='eq',fun=lambda w:w.sum()-1)],options=dict(ftol=1e-9,maxiter=150))
assert opt.success
data['stack_512']=np.concatenate(test)@opt.x
data.to_csv(out/'predictions.csv.gz',index=False)
scores=[]
for station,frame in [('all',data),*list(data.groupby('station'))]:
    for col in ['stack_full','stack_512','t2f_42','t2f_43','t2f_44']:
        error=frame[col]-frame.actual
        scores.append(dict(station=station,model=col,n=len(frame),MAE=error.abs().mean(),RMSE=np.sqrt((error**2).mean())))
scores=pd.DataFrame(scores)
scores.to_csv(out/'scores.csv',index=False)
print(scores.to_string(index=False))

summary = []
for model in ['stack_full', 'stack_512', 't2f_42', 't2f_43', 't2f_44']:
    absolute = (data[model] - data.actual).abs()
    summary.append(dict(model=model, MAE=absolute.mean(), RMSE=np.sqrt(absolute.pow(2).mean()),
                        median_absolute_error=absolute.median(),
                        top_30_error_share=absolute.nlargest(30).sum()/absolute.sum()))
pd.DataFrame(summary).to_csv(out/'summary.csv', index=False)
