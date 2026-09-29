# Weekly load of the plan's own pd numbers (sec 4, 5) against workdays (sec 6).
# Japanese holidays (author at JAIST per usgloc_ral_draft.tex:61): Oct 12 Sports Day, Nov 3 Culture Day.
import datetime as d
hol={d.date(2026,10,12),d.date(2026,11,3)}
weeks=[("W1",d.date(2026,9,29),d.date(2026,10,4),{"K1":.5,"K2":1,"K3":1,"port":.5,"bib":1.5}),
       ("W2",d.date(2026,10,5),d.date(2026,10,11),{"K4":2,"K5":.75,"K6":1.25,"K7":1.75,"K8":1}),
       ("W3",d.date(2026,10,12),d.date(2026,10,18),{"K9":1.5,"K10":2,"K11":3.5}),
       ("W4",d.date(2026,10,19),d.date(2026,10,25),{}),
       ("W5",d.date(2026,10,26),d.date(2026,11,1),{"K12":2.5,"K13":.75}),
       ("W6",d.date(2026,11,2),d.date(2026,11,8),{"K14":1.5,"K15":.5}),
       ("W7",d.date(2026,11,9),d.date(2026,11,15),{})]
tot_code=0
for n,a,b,t in weeks:
    wd=sum(1 for i in range((b-a).days+1) if (a+d.timedelta(i)).weekday()<5)
    wdh=sum(1 for i in range((b-a).days+1) if (a+d.timedelta(i)).weekday()<5 and (a+d.timedelta(i)) not in hol)
    s=sum(t.values()); tot_code+=s
    print(f"{n} {a}..{b} workdays={wd} (JP holidays removed: {wdh}) coded_pd={s:.2f} tasks={list(t)}")
print("code pd placed:", tot_code, "; writing pd not placed in a week except port+bib (2.0 of 7.5)")
