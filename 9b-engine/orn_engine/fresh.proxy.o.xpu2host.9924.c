/* THIS FILE WAS GENERATED AUTOMATICALLY BY XPU-ELFCONV TOOL */
typedef struct { int type; unsigned long long binary; unsigned int size; unsigned int entry_offset; unsigned int param_dword_size; unsigned long long crc32; char* name; void* private; } XPU_LAUNCH_STRUCT;
extern int xpu_launch_async(XPU_LAUNCH_STRUCT* kernel);
