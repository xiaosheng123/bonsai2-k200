import sys
P='/home/caden/ornc/serve2.py'
s=open(P,encoding='utf-8').read()
if 'VISINFO_FIX' in s: print('已修'); sys.exit(0)
old='''                if line.startswith("__VIS__"):
                    p = line.split()'''
new='''                if line.startswith("[vis] 完成"):        # ★ VISINFO_FIX: 视觉塔耗时在此被消费
                    self.last_vis_info = line
                if line.startswith("__VIS__"):
                    p = line.split()'''
assert s.count(old)==1
s=s.replace(old,new)
open(P,'w',encoding='utf-8').write(s)
import ast; ast.parse(s); print('VISINFO_FIX OK')
