import sys
P='/home/caden/orn_engine/orn3.cpp'
s=open(P,encoding='utf-8').read()
if 'VIS_APPEND_FIX' in s:
    print('已打'); sys.exit(0)
old='    g_imgemb.swap(out); g_img_n = ntok; g_img_used = 0;'
new='''    g_imgemb.insert(g_imgemb.end(), out.begin(), out.end());   // ★ VIS_APPEND_FIX: 多图累加
    g_img_n += ntok;'''
assert s.count(old)==1
s=s.replace(old,new)
old2='''                } else if (s.compare(0, 7, "!VISIMG") == 0) {'''
new2='''                } else if (s.compare(0, 5, "!VISR") == 0) {
                    g_imgemb.clear(); g_img_n = 0; g_img_used = 0;
                    printf("__VISR__ 0\\n");
                } else if (s.compare(0, 7, "!VISIMG") == 0) {'''
assert s.count(old2)==1
s=s.replace(old2,new2)
open(P,'w',encoding='utf-8').write(s)
print('ok')
