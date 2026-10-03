struct GT { std::string name; std::vector<uint64_t> dims; uint32_t type = 0; uint64_t off = 0; };
struct GF {
    FILE *f = nullptr;
    uint64_t data_start = 0, align = 32;
    std::vector<GT> ts;  std::map<std::string, int> idx;
    std::vector<std::string> tokens, merges;
    std::vector<int32_t> ttypes;
    std::map<std::string, std::string> strs;
    std::map<std::string, uint64_t> u64s;

    uint64_t nbytes(const GT &t) const {
        uint64_t n = 1; for (size_t i = 0; i < t.dims.size(); i++) n *= t.dims[i];
        switch (t.type) {
            case 0:  return n * 4;
            case 1:  return n * 2;
            case 28: return n * 2;
            case 8:  return (n / 32) * 34;   // Q8_0
            case 2:  return (n / 32) * 18;   // Q4_0
            case 14: return (n / 256) * 210; // Q6_K
            case 13: return (n / 256) * 176; // Q5_K
            case 12: return (n / 256) * 144; // Q4_K
            default: return 0;
        }
    }
    bool read(const GT &t, void *buf) const {
        if (fseek(f, (long)(data_start + t.off), SEEK_SET)) return false;
        return fread(buf, 1, nbytes(t), f) == nbytes(t);
    }
    const GT *get(const char *n) const { auto it = idx.find(n); return it == idx.end() ? nullptr : &ts[it->second]; }

    bool open(const char *path) {
        f = fopen(path, "rb");
        if (!f) { perror("open"); return false; }
        char mg[4]; if (fread(mg, 1, 4, f) != 4 || memcmp(mg, "GGUF", 4)) { printf("not gguf\n"); return false; }
        auto r32 = [&]() { uint32_t v = 0; if (fread(&v, 4, 1, f) != 1) exit(1); return v; };
        auto r64 = [&]() { uint64_t v = 0; if (fread(&v, 8, 1, f) != 1) exit(1); return v; };
        auto rstr = [&]() { uint64_t n = r64(); std::string s; s.resize(n);
                             if (n && fread(&s[0], 1, n, f) != n) exit(1); return s; };
        r32(); uint64_t nten = r64(), nkv = r64();
        // 通用值读取; 只保留我们需要的
        std::function<void(uint32_t, const std::string &)> rd = [&](uint32_t t, const std::string &key) {
            bool wtok = (key == "tokenizer.ggml.tokens"), wmrg = (key == "tokenizer.ggml.merges");
            bool wtt  = (key == "tokenizer.ggml.token_type");
            switch (t) {
                case 0: { uint8_t v; if (fread(&v,1,1,f)!=1) exit(1); u64s[key]=v; break; }
                case 1: { int8_t v;  if (fread(&v,1,1,f)!=1) exit(1); u64s[key]=(uint64_t)(int64_t)v; break; }
                case 2: { uint16_t v; if (fread(&v,2,1,f)!=1) exit(1); u64s[key]=v; break; }
                case 3: { int16_t v; if (fread(&v,2,1,f)!=1) exit(1); u64s[key]=(uint64_t)(int64_t)v; break; }
                case 4: { u64s[key]=r32(); break; }
                case 5: { int32_t v; if (fread(&v,4,1,f)!=1) exit(1); u64s[key]=(uint64_t)(int64_t)v; break; }
                case 6: { float v; if (fread(&v,4,1,f)!=1) exit(1); u64s[key]=(uint64_t)v; break; }
                case 7: { uint8_t v; if (fread(&v,1,1,f)!=1) exit(1); u64s[key]=v; break; }
                case 8: { strs[key]=rstr(); break; }
                case 10: { u64s[key]=r64(); break; }
                case 11: { int64_t v; if (fread(&v,8,1,f)!=1) exit(1); u64s[key]=(uint64_t)v; break; }
                case 12: { double v; if (fread(&v,8,1,f)!=1) exit(1); u64s[key]=(uint64_t)v; break; }
                case 9: {
                    uint32_t et = r32(); uint64_t n = r64();
                    if (et == 8) { for (uint64_t i = 0; i < n; i++) { std::string s = rstr();
                                     if (wtok) tokens.push_back(s); else if (wmrg) merges.push_back(s); } }
                    else if (et == 4) { for (uint64_t i = 0; i < n; i++) { uint32_t v = r32(); if (wtt) ttypes.push_back((int32_t)v); } }
                    else if (et == 5) { for (uint64_t i = 0; i < n; i++) { int32_t v; if (fread(&v,4,1,f)!=1) exit(1); if (wtt) ttypes.push_back(v); } }
                    else if (et == 10) { for (uint64_t i = 0; i < n; i++) r64(); }
                    else if (et == 0 || et == 1 || et == 7) { for (uint64_t i = 0; i < n; i++) { uint8_t v; if (fread(&v,1,1,f)!=1) exit(1); } }
                    else if (et == 6) { for (uint64_t i = 0; i < n; i++) { float v; if (fread(&v,4,1,f)!=1) exit(1); } }
                    else { printf("array et=%u unsupported (%s)\n", et, key.c_str()); exit(1); }
                    u64s[key] = n; break; }
                default: printf("kv type %u key=%s\n", t, key.c_str()); exit(1);
            }
        };
        for (uint64_t i = 0; i < nkv; i++) {
            std::string k = rstr(); uint32_t t = r32();
            rd(t, k);
        }
        for (uint64_t i = 0; i < nten; i++) {
            GT t; t.name = rstr(); uint32_t nd = r32(); t.dims.resize(nd);
            for (uint32_t d = 0; d < nd; d++) t.dims[d] = r64();
            t.type = r32(); t.off = r64();
            idx[t.name] = (int)ts.size(); ts.push_back(t);
        }
        long pos = ftell(f);
        align = u64s.count("general.alignment") ? u64s["general.alignment"] : 32;
        data_start = (uint64_t)((pos + align - 1) / align * align);
        return true;
    }
};

