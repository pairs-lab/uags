import sys,re,html,subprocess,urllib.parse
for q in sys.argv[1:]:
    u='https://arxiv.org/search/?searchtype=all&size=50&order=-announced_date_first&query='+urllib.parse.quote(q)
    s=subprocess.run(['curl','-sS','-m','60','-A','Mozilla/5.0',u],capture_output=True,text=True).stdout
    items=re.findall(r'<p class="list-title[^>]*>\s*<a href="https://arxiv.org/abs/([^"]+)".*?<p class="title is-5 mathjax">(.*?)</p>.*?Submitted</span>\s*([^;<]*)',s,re.S)
    print('##',q,'hits',len(items))
    for aid,t,d in items: print(' ',aid,'|',' '.join(re.sub('<[^>]+>','',html.unescape(t)).split()),'|',d.strip())
