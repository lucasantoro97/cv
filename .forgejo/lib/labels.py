#!/usr/bin/env python3
"""Trusted terminal label/comment update; replaces labels atomically via Forgejo API."""
from __future__ import annotations
import argparse, json, os, urllib.request

def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument('--status', choices=('done','failed'), required=True); args=parser.parse_args()
    root=os.environ['API_URL']; repo=os.environ['REPOSITORY']; issue=os.environ['ISSUE']; token=os.environ['GIT_TOKEN']
    headers={'Authorization':'token '+token,'Content-Type':'application/json'}
    get=urllib.request.Request(f'{root}/repos/{repo}/issues/{issue}',headers=headers)
    with urllib.request.urlopen(get,timeout=10) as response: current=json.load(response)
    labels=[x['name'] for x in current.get('labels',[]) if x.get('name') not in {'ai:run','ai:running','ai:done','ai:failed'}]
    labels.append('ai:'+args.status)
    put=urllib.request.Request(f'{root}/repos/{repo}/issues/{issue}/labels',data=json.dumps({'labels':labels}).encode(),headers=headers,method='PUT')
    with urllib.request.urlopen(put,timeout=10): pass
    comment='Agent completed.' if args.status=='done' else 'Agent failed safely; no branch was published.'
    post=urllib.request.Request(f'{root}/repos/{repo}/issues/{issue}/comments',data=json.dumps({'body':comment}).encode(),headers=headers,method='POST')
    with urllib.request.urlopen(post,timeout=10): pass
if __name__=='__main__': main()
