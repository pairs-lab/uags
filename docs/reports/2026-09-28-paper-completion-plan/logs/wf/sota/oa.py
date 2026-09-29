import sys,json,subprocess,urllib.parse
def get(u):
    r=subprocess.run(['curl','-sS','-m','90',u],capture_output=True,text=True).stdout
    return json.loads(r)
def cites(q):
    # q: DOI or search title
    if q.startswith('10.'):
        w=get('https://api.openalex.org/works/doi:'+q)
    else:
        w=get('https://api.openalex.org/works?per-page=1&search='+urllib.parse.quote(q))['results'][0]
    print('##',w['display_name'],w['id'],'cited_by',w.get('cited_by_count'))
    wid=w['id'].split('/')[-1]
    page=1
    while True:
        d=get(f'https://api.openalex.org/works?filter=cites:{wid}&per-page=200&page={page}&select=display_name,publication_year,primary_location,ids')
        for x in d['results']:
            src=((x.get('primary_location') or {}).get('source') or {}).get('display_name')
            print(' ',x['publication_year'],'|',x['display_name'],'|',src)
        if len(d['results'])<200: break
        page+=1
for q in sys.argv[1:]: cites(q)
