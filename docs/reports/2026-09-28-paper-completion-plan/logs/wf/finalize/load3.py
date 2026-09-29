weeks = {
 "W1 Sep29-Oct4": (4, [("admin day1",0.25),("K1",0.5),("K2",1.0),("K3",1.25)]),
 "W2 Oct5-11":    (5, [("K14",0.75),("K8",1.5),("K9",1.25),("port",0.5)]),
 "W3 Oct12-18":   (4, [("K4",1.5),("K6",1.0),("RIO10 dl+disk",0.25),("K5*",0.5)]),
 "W4 Oct19-25":   (5, [("K12",3.0),("K10a",0.5),("method text 1",0.5)]),
 "W5 Oct26-Nov1": (5, [("K13",1.75),("val II runs",1.25),("K16 sweeps",0.5),("K15",0.5)]),
 "W6 Nov2-8":     (4, [("K7",1.5),("K10b",1.5)]),
 "W7 Nov9-15":    (5, [("K11",2.5),("val I runs",1.25)]),
 "W8 Nov16-22":   (5, [("test runs",1.25),("K16 figs",0.25),("method text 2",1.25),("figures+redraw",1.0)]),
 "W9 Nov23-27":   (4, [("exp text",1.75),("abstract/intro",1.0),("G5",0.25),("revise",0.5)]),
}
tot=0; wd=0
for w,(d,ts) in weeks.items():
    s=sum(p for _,p in ts); tot+=s; wd+=d
    print(f"{w:15s} workdays={d} cap80={0.8*d:.2f} load={s:.2f} {'OVER' if s>0.8*d+1e-9 else 'ok'}")
print(f"TOTAL workdays={wd} load={tot:.2f} (incl K5 0.5) buffer={(wd-tot)/wd*100:.1f}%; without K5 load={tot-0.5:.2f} buffer={(wd-tot+0.5)/wd*100:.1f}%")
d=4.25; print(f"all defer items kept: load={tot-0.5+d:.2f} buffer={(wd-(tot-0.5+d))/wd*100:.1f}%")
print(f"supervisor takes figures (1.0): load={tot-0.5-1.0:.2f} buffer={(wd-(tot-1.5))/wd*100:.1f}%")
