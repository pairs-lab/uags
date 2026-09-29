import sys, numpy as np
rows=[l.split() for l in open(sys.argv[1]) if l.startswith('seq')]
sc={}
for r in rows:
    s=r[0].split('/')[0]
    if not s.endswith('_02'): continue
    sc.setdefault(s,[]).append((float(r[3]),float(r[4])))
from scipy.stats import spearmanr
for s in sorted(sc):
    a=np.array(sc[s]); sem,geo=a[:,0],a[:,1]
    hi_s=sem>=np.quantile(sem,0.75); lo_g=geo<=np.quantile(geo,0.25)
    print(f"{s} n={len(a)} SemChange p50={np.median(sem):.3f} spearman(Sem,Geo)={spearmanr(sem,geo).correlation:.2f} frac(highSem & lowGeo)={np.mean(hi_s&lo_g):.3f}")
