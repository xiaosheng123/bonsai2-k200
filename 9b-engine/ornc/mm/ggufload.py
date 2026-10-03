"""Minimal GGUF reader (mmproj: F32/BF16/BF16) - numpy only."""
import numpy as np, struct, os

GGML_F32  = 0
GGML_F16  = 1
GGML_BF16 = 30
GGML_I32  = 26

NBYTES = {GGML_F32: 4, GGML_F16: 2, GGML_BF16: 2, GGML_I32: 4}

class GGUF:
    def __init__(self, path):
        self.path = path
        self.f = open(path, 'rb')
        f = self.f
        magic = f.read(4)
        assert magic == b'GGUF', magic
        self.version, self.n_tensors, self.n_kv = struct.unpack('<IQQ', f.read(20))
        self.kv = {}
        for _ in range(self.n_kv):
            k = self._str()
            t = struct.unpack('<I', f.read(4))[0]
            self.kv[k] = self._val(t)
        self.tensors = {}
        for _ in range(self.n_tensors):
            name = self._str()
            nd = struct.unpack('<I', f.read(4))[0]
            ne = list(struct.unpack('<%dQ' % nd, f.read(8 * nd)))
            t = struct.unpack('<I', f.read(4))[0]
            off = struct.unpack('<Q', f.read(8))[0]
            self.tensors[name] = dict(ne=ne, type=t, off=off)
        align = self.kv.get('general.alignment', 32)
        self.data_off = (f.tell() + align - 1) // align * align

    def _str(self):
        n = struct.unpack('<Q', self.f.read(8))[0]
        return self.f.read(n).decode('utf-8')

    def _val(self, t):
        f = self.f
        if t == 0:  return struct.unpack('<B', f.read(1))[0]
        if t == 1:  return struct.unpack('<b', f.read(1))[0]
        if t == 2:  return struct.unpack('<H', f.read(2))[0]
        if t == 3:  return struct.unpack('<h', f.read(2))[0]
        if t == 4:  return struct.unpack('<I', f.read(4))[0]
        if t == 5:  return struct.unpack('<i', f.read(4))[0]
        if t == 6:  return struct.unpack('<f', f.read(4))[0]
        if t == 7:  return struct.unpack('<B', f.read(1))[0]
        if t == 8:  return self._str()
        if t == 9:
            et = struct.unpack('<I', f.read(4))[0]
            n = struct.unpack('<Q', f.read(8))[0]
            return [self._val(et) for _ in range(n)]
        if t == 10: return struct.unpack('<Q', f.read(8))[0]
        if t == 11: return struct.unpack('<q', f.read(8))[0]
        if t == 12: return struct.unpack('<d', f.read(8))[0]
        raise ValueError('kv type %d' % t)

    def raw(self, name):
        t = self.tensors[name]
        nb = int(np.prod(t['ne'])) * NBYTES[t['type']]
        self.f.seek(self.data_off + t['off'])
        return self.f.read(nb)

    def np(self, name):
        """返回逻辑数组 arr[i0,i1,..] (ggml ne 序, i0 最快)"""
        t = self.tensors[name]
        ne = t['ne']
        raw = self.raw(name)
        if t['type'] == GGML_F32:
            a = np.frombuffer(raw, dtype='<f4')
        elif t['type'] == GGML_F16:
            a = np.frombuffer(raw, dtype='<f2').astype(np.float32)
        elif t['type'] == GGML_BF16:
            u = np.frombuffer(raw, dtype='<u2').astype(np.uint32) << 16
            a = u.view(np.float32)
        elif t['type'] == GGML_I32:
            a = np.frombuffer(raw, dtype='<i4')
        else:
            raise ValueError('type')
        if len(ne) <= 1:
            return np.ascontiguousarray(a)
        r = tuple(range(len(ne)-1, -1, -1))
        return np.ascontiguousarray(a.reshape(tuple(ne[::-1])).transpose(r))

    def flat(self, name):
        """原始一维缓冲 (ggml 序)"""
        return np.ascontiguousarray(self.np(name).transpose(
            tuple(range(len(self.tensors[name]['ne'])-1, -1, -1)))).ravel() \
            if len(self.tensors[name]['ne']) > 1 else np.ascontiguousarray(self.np(name))

    def f32(self, name):
        """[out,in] 行优先的线性权重 (经典约定)"""
        a = self.np(name)
        if a.ndim == 1:
            return a
        return np.ascontiguousarray(np.moveaxis(a, 0, -1))

def dequantize(pil_img):
    return pil_img
