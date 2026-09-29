import sys,re,html,subprocess,os
for aid in sys.argv[1:]:
    fn=f'abs_{aid.replace("/","_")}.html'
    if not os.path.exists(fn) or os.path.getsize(fn)<1000:
        subprocess.run(['curl','-sS','-m','60','-o',fn,f'https://arxiv.org/abs/{aid}'])
    s=open(fn,encoding='utf-8',errors='ignore').read()
    def meta(n): return [html.unescape(x) for x in re.findall(r'<meta name="%s" content="([^"]*)"'%n,s)]
    print('==',aid)
    print(' title:',meta('citation_title'))
    print(' authors:','; '.join(meta('citation_author')))
    print(' date:',meta('citation_date'))
    for cls in ['comments','jref']:
        m=re.search(r'<td class="tablecell %s[^"]*">(.*?)</td>'%cls,s,re.S)
        if m: print(' %s:'%cls,' '.join(re.sub('<[^>]+>','',html.unescape(m.group(1))).split()))
    m=re.search(r'data-doi="([^"]+)"',s)
    if m: print(' doi:',m.group(1))
    v=re.findall(r'<strong>\[(v\d+)\]</strong>\s*([^<]*)',s)
    print(' versions:',[(a,' '.join(b.split())[:30]) for a,b in v])
    ab=meta('citation_abstract')
    if ab: print(' abstract:',' '.join(ab[0].split()))
