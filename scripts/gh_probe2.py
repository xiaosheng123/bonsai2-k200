#!/usr/bin/env python3
import urllib.request, json, time

def get(url, timeout=60):
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.status, r.read()
        except Exception as e:
            print('retry', attempt, repr(e)[:80], flush=True)
            time.sleep(5)
    return None, b''

def save(path, body):
    open('/home/caden/bonsai2/fork-src/' + path, 'wb').write(body)
    print('SAVED', path, len(body))

# llama.cpp fork README
code, body = get('https://raw.githubusercontent.com/PrismML-Eng/llama.cpp/master/README.md')
if code == 200:
    save('fork_README.md', body)
else:
    print('README fail', code)

# Bonsai-demo 仓库整树
def walk_repo(repo, path='', depth=0):
    if depth > 2:
        return
    api = 'https://api.github.com/repos/PrismML-Eng/%s/contents/%s' % (repo, path)
    code, body = get(api)
    if code != 200:
        return
    try:
        d = json.loads(body)
    except Exception as e:
        print('parse fail', e)
        return
    for item in d:
        nm = item['name']
        if item['type'] == 'dir':
            print('DIR', nm)
            walk_repo(repo, path + '/' + nm if path else nm, depth + 1)
        elif item['type'] == 'file':
            print('FILE', nm)
            if nm.endswith('.py') or nm.endswith('.md') or nm.endswith('.sh'):
                c2, b2 = get(item['download_url'])
                if c2 == 200:
                    save('bonsaidemo_' + nm, b2)

walk_repo('Bonsai-demo')
print('DONE')