import numpy as np, cv2, math, pandas as pd
from pathlib import Path
outdir=Path('/mnt/data/chamfer_sim'); outdir.mkdir(exist_ok=True)

def make_satellite_structure(size=512, seed=0):
    rng=np.random.default_rng(seed); img=np.zeros((size,size), np.uint8)
    for p1,p2,w in [((20,140),(500,160),10),((40,370),(480,350),12),((220,20),(240,500),9),((30,260),(490,275),8),((360,50),(380,480),8)]:
        cv2.line(img,p1,p2,170,w); cv2.line(img,p1,p2,255,2)
    for i in range(12):
        x=int(rng.integers(30,size-100)); y=int(rng.integers(30,size-100)); w=int(rng.integers(35,85)); h=int(rng.integers(30,80)); angle=float(rng.choice([0,0,10,-10])); rect=((x+w/2,y+h/2),(w,h),angle); box=cv2.boxPoints(rect).astype(np.int32); cv2.drawContours(img,[box],0,115,-1); cv2.drawContours(img,[box],0,255,2)
    for i in range(18):
        c=(int(rng.integers(20,size-20)), int(rng.integers(20,size-20))); r=int(rng.integers(6,18)); cv2.circle(img,c,r,90,-1); cv2.circle(img,c,r,170,1)
    return cv2.GaussianBlur(img,(3,3),0)

def edges(gray):
    e=cv2.Canny(gray,50,120); return cv2.dilate(e,np.ones((3,3),np.uint8),1)

def make_bev(sat_edges, center, crop=128, rng=None, dropout=.55):
    rng=rng or np.random.default_rng(); h=crop//2; x,y=center; p=sat_edges[y-h:y+h,x-h:x+h].copy(); bev=(p>0).astype(np.uint8)*255; bev[(rng.random(bev.shape)<dropout)]=0; bev=cv2.dilate(bev,np.ones((2,2),np.uint8),1); mask=np.zeros_like(bev); cv2.ellipse(mask,(h,h),(58,35),float(rng.integers(-30,30)),0,360,255,-1); bev=cv2.bitwise_and(bev,mask); ys=rng.integers(0,crop,80); xs=rng.integers(0,crop,80); bev[ys,xs]=255; return cv2.morphologyEx(bev,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8),1)

def score(bev, crop):
    sat=(crop>0).astype(np.uint8); pts=bev>0
    if sat.sum()<10 or pts.sum()<10: return -1e9,999
    dt=cv2.distanceTransform(1-sat, cv2.DIST_L2, 3); md=float(dt[pts].mean()); return math.exp(-md/5.0), md

def search(bev, sat_edges, pred, radius=40, step=8):
    crop=bev.shape[0]; h=crop//2; H,W=sat_edges.shape; res=[]
    for dy in range(-radius,radius+1,step):
      for dx in range(-radius,radius+1,step):
        cx,cy=pred[0]+dx,pred[1]+dy
        if cx-h<0 or cy-h<0 or cx+h>W or cy+h>H: continue
        sc,md=score(bev,sat_edges[cy-h:cy+h,cx-h:cx+h]); res.append((sc,cx,cy,md))
    res.sort(reverse=True,key=lambda x:x[0]); return res

sat=make_satellite_structure(seed=4); sat_e=edges(sat); rng=np.random.default_rng(42); true=(260,260); pred=(287,244); bev=make_bev(sat_e,true,rng=rng,dropout=.45); top=search(bev,sat_e,pred); print('top3',top[:3])
c=bev.shape[0]; h=c//2; canvas=np.ones((c,c*4+30,3),np.uint8)*255
imgs=[bev,sat_e[pred[1]-h:pred[1]+h,pred[0]-h:pred[0]+h],sat_e[top[0][2]-h:top[0][2]+h,top[0][1]-h:top[0][1]+h],sat_e[true[1]-h:true[1]+h,true[0]-h:true[0]+h]]
titles=['BEV partial','GPS crop','Chamfer best','True crop']
for i,(im,t) in enumerate(zip(imgs,titles)):
    rgb=cv2.cvtColor(im,cv2.COLOR_GRAY2BGR); x=i*(c+10); canvas[:,x:x+c]=rgb; cv2.putText(canvas,t,(x+3,18),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,0,255),1,cv2.LINE_AA)
cv2.imwrite(str(outdir/'simulation_example.png'),canvas)
over=np.zeros((c,c,3),np.uint8); over[:,:,1]=bev; over[:,:,2]=imgs[2]; cv2.imwrite(str(outdir/'best_overlay_green_bev_red_sat.png'),over)
rows=[]
for seed in range(12):
    rng=np.random.default_rng(100+seed); sat=make_satellite_structure(seed=seed); sat_e=edges(sat); h=64; radius=40
    true=(int(rng.integers(h+radius+5,512-h-radius-5)),int(rng.integers(h+radius+5,512-h-radius-5))); ge=(int(rng.normal(0,18)),int(rng.normal(0,18))); pred=(true[0]+ge[0],true[1]+ge[1])
    bev=make_bev(sat_e,true,rng=rng,dropout=.55); top=search(bev,sat_e,pred,radius=radius,step=8); best=top[0]; rows.append({'gps_err_pix':math.hypot(pred[0]-true[0],pred[1]-true[1]),'match_err_pix':math.hypot(best[1]-true[0],best[2]-true[1]),'score':best[0]})
df=pd.DataFrame(rows); df.to_csv(outdir/'simulation_trials.csv',index=False); print(df.describe()); print('success <=8pix', (df.match_err_pix<=8).mean())
