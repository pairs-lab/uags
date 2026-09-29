import sys,json,subprocess,time
def get(u):
    for i in range(6):
        r=subprocess.run(['curl','-sS','-m','60',u],capture_output=True,text=True).stdout
        if 'Too Many' not in r and r.strip(): return json.loads(r)
        time.sleep(15)
    return {}
for pid in sys.argv[1:]:
    d=get(f'https://api.semanticscholar.org/graph/v1/paper/{pid}/citations?fields=title,year,venue,externalIds,publicationDate&limit=200')
    print('##',pid,'n=',len(d.get('data',[])))
    rows=[]
    for x in d.get('data',[]):
        p=x['citingPaper']; rows.append((str(p.get('publicationDate') or p.get('year')),p.get('title'),p.get('venue'),(p.get('externalIds') or {}).get('ArXiv')))
    for r in sorted(rows,reverse=True): print(' ',' | '.join(str(v) for v in r))
    time.sleep(3)
