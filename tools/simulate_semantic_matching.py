#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Synthetic feasibility test for semantic-map-to-satellite semantic matching.
Classes:
  0 open/unknown ground, 1 road, 2 building, 3 vegetation, 255 unobserved.
This script creates a satellite semantic map, simulates a noisy local geometric-semantic BEV,
then searches translation/yaw candidates around a noisy GPS prior using foreground-only semantic agreement.
"""
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path("./semantic_matching_sim_out")
OUT.mkdir(exist_ok=True, parents=True)

COLORS = {
    0: np.array([238,235,220],np.uint8),
    1: np.array([105,105,105],np.uint8),
    2: np.array([210,75,55],np.uint8),
    3: np.array([60,150,70],np.uint8),
    255: np.array([0,0,0],np.uint8),
}
WEIGHTS = np.array([0.2, 2.0, 3.0, 1.4], float)

def colorize(seg):
    img=np.zeros((*seg.shape,3),np.uint8)
    for k,c in COLORS.items(): img[seg==k]=c
    return img

def make_global(seed=42,H=620,W=620):
    rng=np.random.default_rng(seed); s=np.zeros((H,W),np.uint8)
    for y in [95,280,485]:
        cv2.line(s,(0,y),(W-1,y+int(rng.integers(-25,25))),1,int(rng.integers(20,32)))
    for x in [125,330,520]:
        cv2.line(s,(x,0),(x+int(rng.integers(-20,20)),H-1),1,int(rng.integers(20,30)))
    pts=np.array([[30,555],[150,495],[290,420],[450,350],[600,285]],np.int32)
    cv2.polylines(s,[pts],False,1,24)
    rects=[(55,45,80,52),(220,50,80,65),(390,60,100,55),(55,185,75,90),
           (210,195,100,80),(410,190,95,80),(75,360,100,80),(230,365,80,90),
           (405,405,120,80),(525,80,60,90),(535,410,65,120)]
    for x,y,w,h in rects:
        cv2.rectangle(s,(x,y),(x+w,y+h),2,-1)
        if w>90 and h>70:
            cv2.rectangle(s,(x+24,y+18),(x+w-24,y+h-18),0,-1)
    for _ in range(65):
        x=int(rng.integers(15,W-15)); y=int(rng.integers(15,H-15)); r=int(rng.integers(7,19))
        if s[y,x]!=2: cv2.circle(s,(x,y),r,3,-1)
    cv2.rectangle(s,(455,500),(600,600),0,-1); cv2.rectangle(s,(475,520),(580,580),3,2)
    return s

def extract_local(g,center,yaw,patch=180):
    H,W=g.shape
    M=cv2.getRotationMatrix2D(center,yaw,1.0)
    rot=cv2.warpAffine(g,M,(W,H),flags=cv2.INTER_NEAREST,borderValue=255)
    x,y=center; x0=int(x-patch/2); y0=int(y-patch/2)
    crop=np.full((patch,patch),255,np.uint8)
    sx0=max(0,x0); sy0=max(0,y0); sx1=min(W,x0+patch); sy1=min(H,y0+patch)
    crop[sy0-y0:sy1-y0, sx0-x0:sx1-x0]=rot[sy0:sy1,sx0:sx1]
    return crop

def degrade(local,seed,unknown=0.28,noise=0.06):
    rng=np.random.default_rng(seed); loc=local.copy(); h,w=loc.shape
    loc[rng.random((h,w))<unknown]=255
    for _ in range(6):
        x=int(rng.integers(0,w-25)); y=int(rng.integers(0,h-25))
        ww=int(rng.integers(18,55)); hh=int(rng.integers(18,55))
        loc[y:min(h,y+hh),x:min(w,x+ww)]=255
    known=loc!=255; m=(rng.random((h,w))<noise)&known
    loc[m]=rng.choice([0,1,2,3],size=int(m.sum()))
    b=(loc==2).astype(np.uint8)*255
    if b.sum()>0:
        edges=cv2.Canny(b,50,100); edges=cv2.dilate(edges,np.ones((3,3),np.uint8),iterations=1)
        loc[(loc==2)&(edges==0)]=255
    road=(loc==1).astype(np.uint8)*255
    if road.sum()>0:
        rd=cv2.erode(road,np.ones((3,3),np.uint8),iterations=1)
        loc[(loc==1)&(rd==0)]=255
    return loc

def rotate_local(local,yaw):
    h,w=local.shape; M=cv2.getRotationMatrix2D((w/2,h/2),-yaw,1.0)
    return cv2.warpAffine(local,M,(w,h),flags=cv2.INTER_NEAREST,borderValue=255)

def score_patch(lrot,g,cand):
    patch=lrot.shape[0]; H,W=g.shape; x,y=cand
    x0=int(round(x-patch/2)); y0=int(round(y-patch/2))
    if x0<0 or y0<0 or x0+patch>W or y0+patch>H: return -1
    gp=g[y0:y0+patch,x0:x0+patch]
    mask=(lrot!=255)&(lrot!=0)
    if mask.sum()<180 or mask.mean()<0.006: return -1
    l=lrot[mask]; gg=gp[mask]
    w=np.take(WEIGHTS,l)
    compat=(l==gg).astype(float)
    compat[((l==1)&(gg==0))]=0.15
    compat[((l==3)&(gg==0))]=0.1
    return float((compat*w).sum()/(w.sum()+1e-6))

def search(g,local,gps,search_radius=50,step=4,yaws=range(-15,16,5)):
    best=(-1,None,None); heat=[]
    for yaw in yaws:
        lr=rotate_local(local,yaw)
        for dy in range(-search_radius,search_radius+1,step):
            for dx in range(-search_radius,search_radius+1,step):
                sc=score_patch(lr,g,(gps[0]+dx,gps[1]+dy))
                heat.append((dx,dy,yaw,sc))
                if sc>best[0]: best=(sc,(gps[0]+dx,gps[1]+dy),yaw)
    return best,pd.DataFrame(heat,columns=['dx','dy','yaw','score'])

def run_trial(seed):
    g=make_global(42); rng=np.random.default_rng(seed)
    center=(int(rng.integers(210,430)),int(rng.integers(210,430)))
    yaw_true=float(rng.choice([-8,-4,0,4,8]))
    loc=degrade(extract_local(g,center,yaw_true),seed+100)
    gps=(center[0]+rng.normal(0,22),center[1]+rng.normal(0,22))
    best,heat=search(g,loc,gps)
    score,cand,yaw=best
    return g,loc,heat,dict(seed=seed,true_x=center[0],true_y=center[1],true_yaw=yaw_true,
                           gps_x=gps[0],gps_y=gps[1],
                           gps_err_m=float(np.linalg.norm(np.array(gps)-np.array(center))*0.5),
                           est_x=cand[0],est_y=cand[1],est_yaw=yaw,score=score,
                           match_err_m=float(np.linalg.norm(np.array(cand)-np.array(center))*0.5))

if __name__ == "__main__":
    rows=[]; sample=None
    for seed in range(40):
        g,loc,heat,rec=run_trial(seed)
        rows.append(rec)
        if seed==7: sample=(g,loc,heat,rec)
    df=pd.DataFrame(rows)
    df.to_csv(OUT/"semantic_matching_trials.csv", index=False)
    print(df[["gps_err_m","match_err_m","score"]].describe())
    print("success <5m:", (df.match_err_m<5).mean(), "success <10m:", (df.match_err_m<10).mean())

    g,loc,heat,rec=sample
    yaw_est=rec["est_yaw"]
    hm=heat[heat.yaw==yaw_est].pivot(index="dy",columns="dx",values="score").sort_index().values
    hm=(hm-np.nanmin(hm))/(np.nanmax(hm)-np.nanmin(hm)+1e-6)
    overlay=colorize(g).astype(np.float32)
    lr=rotate_local(loc,yaw_est); patch=lr.shape[0]
    x=int(round(rec["est_x"])); y=int(round(rec["est_y"]))
    x0=x-patch//2; y0=y-patch//2
    mask=(lr!=255)&(lr!=0)
    sub=overlay[y0:y0+patch,x0:x0+patch]
    if sub.shape[:2]==lr.shape:
        sub[mask]=0.35*sub[mask]+0.65*np.array([0,255,80],np.float32)
        overlay[y0:y0+patch,x0:x0+patch]=sub
    fig=plt.figure(figsize=(14,9))
    ax=fig.add_subplot(2,3,1); ax.imshow(colorize(g)); ax.set_title("Satellite semantic map"); ax.axis("off")
    ax.scatter([rec["true_x"]],[rec["true_y"]],c="cyan",s=40,label="true")
    ax.scatter([rec["gps_x"]],[rec["gps_y"]],c="magenta",s=40,label="GPS prior")
    ax.scatter([rec["est_x"]],[rec["est_y"]],c="yellow",s=40,label="matched")
    ax.legend(fontsize=8)
    ax=fig.add_subplot(2,3,2); ax.imshow(colorize(loc)); ax.set_title("Noisy local geometric-semantic BEV"); ax.axis("off")
    ax=fig.add_subplot(2,3,3); ax.imshow(hm,cmap="hot",origin="lower"); ax.set_title("Search heatmap"); ax.axis("off")
    ax=fig.add_subplot(2,3,4); ax.imshow(np.clip(overlay,0,255).astype(np.uint8)); ax.set_title("Overlay: green local BEV on satellite semantic map"); ax.axis("off")
    ax=fig.add_subplot(2,3,5); ax.hist(df.gps_err_m,alpha=0.6,label="GPS prior"); ax.hist(df.match_err_m,alpha=0.6,label="semantic match"); ax.legend(); ax.set_xlabel("error / m")
    ax=fig.add_subplot(2,3,6); ax.scatter(df.gps_err_m,df.match_err_m,c=df.score,cmap="viridis"); ax.plot([0,40],[0,40],"--",color="gray"); ax.set_xlabel("GPS error / m"); ax.set_ylabel("semantic match error / m")
    plt.tight_layout(); fig.savefig(OUT/"semantic_matching_simulation_result.png", dpi=220)
