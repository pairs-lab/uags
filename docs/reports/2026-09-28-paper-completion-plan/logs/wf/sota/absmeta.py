import sys,re,html,subprocess
for aid in sys.argv[1:]:
    fn=f'abs_{aid.replace("/","_")}.html'
    subprocess.run(['curl','-sS','-m','60','-o',fn,f'https://arxiv.org/abs/{aid}'])
    s=open(fn,encoding='utf-8',errors='ignore').read()
    def meta(n): return [html.unescape(x) for x in re.findall(r'<meta name="%s" content="([^"]*)"'%n,s)]
    print('==',aid)
    print(' title:',meta('citation_title'))
    print(' authors:','; '.join(meta('citation_author')))
    print(' date:',meta('citation_date'))
    m=re.search(r'<td class="tablecell comments[^"]*">(.*?)</td>',s,re.S)
    if m: print(' comments:',' '.join(re.sub('<[^>]+>','',html.unescape(m.group(1))).split()))
    m=re.search(r'<td class="tablecell jref">(.*?)</td>',s,re.S)
    if m: print(' jref:',' '.join(re.sub('<[^>]+>','',html.unescape(m.group(1))).split()))
    m=re.search(r'<td class="tablecell doi">(.*?)</td>',s,re.S) or re.search(r'data-doi="([^"]+)"',s)
    if m: print(' doi:',' '.join(re.sub('<[^>]+>','',html.unescape(m.group(1))).split()))
    v=re.findall(r'<strong>\[(v\d+)\]</strong>\s*([^<]*)',s)
    print(' versions:',[(a,' '.join(b.split())) for a,b in v])
