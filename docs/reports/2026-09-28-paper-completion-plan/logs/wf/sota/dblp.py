import sys,json,subprocess,urllib.parse
for q in sys.argv[1:]:
    u='https://dblp.org/search/publ/api?format=json&h=8&q='+urllib.parse.quote(q)
    r=subprocess.run(['curl','-sS','-m','60',u],capture_output=True,text=True).stdout
    print('##',q)
    try: hits=json.loads(r)['result']['hits'].get('hit',[])
    except Exception as e: print(' ERR',r[:200]); continue
    for h in hits:
        i=h['info']; a=i.get('authors',{}).get('author',[])
        if isinstance(a,dict): a=[a]
        print(' -',i.get('title'),'|',', '.join(x['text'] for x in a),'|',i.get('venue'),i.get('year'),'vol',i.get('volume'),'no',i.get('number'),'pp',i.get('pages'),'|',i.get('doi'),'|',i.get('type'))
