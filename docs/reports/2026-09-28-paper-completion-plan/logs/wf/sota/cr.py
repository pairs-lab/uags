import sys,json,subprocess,urllib.parse
def show(it):
    print('  ',it.get('title'),'|',', '.join(f"{a.get('given','')} {a.get('family','')}" for a in it.get('author',[])),'|',it.get('container-title'),'| vol',it.get('volume'),'no',it.get('issue'),'pp',it.get('page'),'|',it.get('DOI'),'|',(it.get('published-print') or it.get('published') or {}).get('date-parts'),'| type',it.get('type'))
for q in sys.argv[1:]:
    if q.startswith('10.'):
        u='https://api.crossref.org/works/'+q
        r=subprocess.run(['curl','-sS','-m','60',u],capture_output=True,text=True).stdout
        print('## DOI',q)
        try: show(json.loads(r)['message'])
        except Exception as e: print(' ERR',r[:200])
        continue
    u='https://api.crossref.org/works?rows=4&query.bibliographic='+urllib.parse.quote(q)
    r=subprocess.run(['curl','-sS','-m','60',u],capture_output=True,text=True).stdout
    print('##',q)
    try:
        for it in json.loads(r)['message']['items']: show(it)
    except Exception as e: print(' ERR',r[:200])
