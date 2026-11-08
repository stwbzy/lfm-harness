import json,sys
for f in sys.argv[1:]:
    rows=[json.loads(l) for l in open(f)]
    p1=pr=tot=0; toks=0; outs={}
    for r in rows:
        a=r['attempts']; tot+=1
        toks+=sum(x['gen_tokens'] for x in a)
        if a[0]['outcome']=='pass': p1+=1
        if any(x['outcome']=='pass' for x in a): pr+=1
        outs[r['id']]=[x['outcome'] for x in a]
    print(f.split('/')[-1], 'tasks',tot,'pass@1',p1,'after repair',pr,'gen_tokens',toks, 'mean_tok_per_task', round(toks/max(tot,1)))
