// src/kernels/qsa_attn_g8.cpp - dense QSA decode attention for a NO-indexer model (Ornith 1.5 / qwen35moe):
//   16 query heads / 2 KV = G=8, head_dim 256, INT8 KV (kv_mode=1), the WHOLE populated context attended.
//
// This is the controlled experiment for report 05: the fork's INT8 v3 attention kernel (attn_chunk_q8_v3_kernel)
// is specialised for Flash-Next's G=12 only. This model is G=8, so it never runs - and the default G=8 kernel runs
// dense attention over 131K/262K cells at ~6x the memory floor (report 05). Here we (a) numerically verify the new
// specialised G=8 INT8 kernel (attn_chunk_q8_v3_g8_kernel, opt-in via STRATA_ATTN_V3_G8) is bit-for-bit the same as
// the default attn_chunk_kernel<1,false,8>, and (b) benchmark the two against each other with CUDA events at 131K and
// 260K.  The kernel is selected by toggling the env var around the call (both go through the real
// qsa_decode_attn_batch dispatch), so parity exercises production's exact launch path - not a reimplementation.
//
//   qsa_attn_g8 [--device N] [--iters 200] [--trace]
//   exit 0 (parity + perf ok), 1 (a parity mismatch), 77 (no GPU)
#include "strata/kernels/kv_q8.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace k = strata::kernels;

namespace {
cudaStream_t st = nullptr;
void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) { std::fprintf(stderr, "qsa_attn_g8: %s: %s\n", what, cudaGetErrorString(e)); std::exit(2); }
}
template <typename T> T* dalloc(size_t n) { T* p = nullptr;
    ck(cudaMalloc((void**) &p, n * sizeof(T) + 256), "malloc"); ck(cudaMemset(p, 0, n * sizeof(T) + 256), "memset");
    return p; }
template <typename T> T* upload(const std::vector<T>& h) { T* d = dalloc<T>(h.size());
    ck(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice), "upload"); return d; }
template <typename T> std::vector<T> download(const T* d, size_t n) { std::vector<T> h(n);
    ck(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost), "download"); return h; }
uint16_t f16_scale_q8(uint32_t r) { return (uint16_t) ((9u << 10) | (r & 0x3FFu)); }
struct Timer {
    cudaEvent_t e0 = nullptr, e1 = nullptr;
    Timer() { ck(cudaEventCreate(&e0), "ev"); ck(cudaEventCreate(&e1), "ev"); }
    ~Timer() { cudaEventDestroy(e0); cudaEventDestroy(e1); }
    // microseconds per call, median of 3 back-to-back runs (warmed; CUDA events don't pay the first-launch fault here)
    template <typename F> double us(F&& f, int iters) {
        double r[3];
        for (int a = 0; a < 3; ++a) { for (int i = 0; i < 4; ++i) f(); ck(cudaDeviceSynchronize(), "warm");
            ck(cudaEventRecord(e0, st), "r"); for (int i = 0; i < iters; ++i) f();
            ck(cudaEventRecord(e1, st), "r"); ck(cudaEventSynchronize(e1), "s"); float ms = 0.0f;
            ck(cudaEventElapsedTime(&ms, e0, e1), "el"); r[a] = 1000.0 * ms / iters; }
        std::sort(r, r + 3); return r[1]; }
};
constexpr int NH = 16, NHV = 2, HD = 256, CHUNK = 64;
}  // namespace

// A scenario's inputs, freshly built (so the two kernels see byte-identical pools).  cap = whole context (dense).
struct In {
    int64_t ctx, cells, pages, prows, cap, n_chunks;
    k::QsaAttnPools pools = {};
    int8_t* k8 = nullptr, * v8 = nullptr; uint16_t* k8s = nullptr, * v8s = nullptr; int32_t* table = nullptr;
    float* q = nullptr; int32_t* ids = nullptr; int32_t* steps = nullptr;
    float* scrA = nullptr, * scrB = nullptr; float* attnA = nullptr, * attnB = nullptr;
};

static In build(int64_t ctx, bool masked) {
    In in{};
    const int64_t PS = k::qsa_real_shapes().page_size;   // 4: the engine's page granule; prows = cells * NHV either way
    in.ctx = ctx; in.cells = (ctx + PS - 1) / PS * PS; in.pages = in.cells / PS; in.prows = in.pages * NHV * PS;
    in.cap = in.cells; in.n_chunks = (in.cap + CHUNK - 1) / CHUNK;
    std::mt19937 rng(20261012u); std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<int8_t> c8((size_t) (in.prows * HD));
    for (auto& x : c8) x = (int8_t) ((int) (rng() % 255u) - 127);
    in.k8 = upload(c8);
    for (auto& x : c8) x = (int8_t) ((int) (rng() % 255u) - 127);
    in.v8 = upload(c8);
    std::vector<uint16_t> sc((size_t) (in.prows * (HD / k::KV_Q8_GROUP)));
    for (auto& x : sc) x = f16_scale_q8(rng());
    in.k8s = upload(sc);
    for (auto& x : sc) x = f16_scale_q8(rng());
    in.v8s = upload(sc);
    std::vector<int32_t> tab((size_t) in.pages);
    for (int64_t i = 0; i < in.pages; ++i) { int32_t t = (int32_t) i; if (masked && (i % 7) == 3) t = -1; tab[i] = t; }
    in.table = upload(tab);
    in.pools.page_table = in.table; in.pools.k_q = in.k8; in.pools.v_q = in.v8; in.pools.k_scale = in.k8s; in.pools.v_scale = in.v8s;
    std::vector<float> hq((size_t) (NH * HD)); for (auto& x : hq) x = nd(rng); in.q = upload(hq);
    std::vector<int32_t> hid((size_t) in.cap); for (int64_t j = 0; j < in.cap; ++j) hid[j] = (int32_t) j;
    in.ids = upload(hid);
    std::vector<int32_t> hst(k::kStepCount);
    hst[k::kStepPos] = (int32_t) in.ctx; hst[k::kStepNKv] = (int32_t) (in.ctx + 1); hst[k::kStepNBid] = 0;
    hst[k::kStepWidth] = (int32_t) in.cap;
    in.steps = upload(hst);
    in.scrA = dalloc<float>((size_t) k::qsa_decode_attn_scratch_floats(in.cap, k::qsa_real_shapes()));
    in.scrB = dalloc<float>((size_t) k::qsa_decode_attn_scratch_floats(in.cap, k::qsa_real_shapes()));
    in.attnA = dalloc<float>((size_t) (NH * HD)); in.attnB = dalloc<float>((size_t) (NH * HD));
    return in;
}
static void freeIn(In& in) {
    void* mems[] = { (void*) in.k8, (void*) in.v8, (void*) in.k8s, (void*) in.v8s, (void*) in.table, (void*) in.q,
                     (void*) in.ids, (void*) in.steps, (void*) in.scrA, (void*) in.scrB, (void*) in.attnA, (void*) in.attnB };
    for (void* p : mems) if (p) cudaFree(p);
}
// set the env var the dispatch reads, then run qsa_decode_attn_batch into `out` (separate scratch so the two are
// independent). on=1 selects the INT8 v3 G=8 kernel, on=0 the default G=8 kernel.
static void run_one(const In& in, float* out, int on) {
    setenv("STRATA_ATTN_V3_G8", on ? "1" : "0", 1);
    k::qsa_decode_attn_batch(in.q, in.pools, in.ids, in.steps, in.cap, k::qsa_real_shapes(),
                             on ? in.scrA : in.scrB, out, 1, st);
}
static int parity(const char* label, const In& in) {
    run_one(in, in.attnA, 0);   // default G=8 kernel
    run_one(in, in.attnB, 1);   // INT8 v3 G=8 kernel
    cudaDeviceSynchronize();
    auto a = download(in.attnA, (size_t) (NH * HD)), b = download(in.attnB, (size_t) (NH * HD));
    int bad = 0; double maxdiff = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i] != b[i]) { if (bad < 8) std::printf("    bit diff idx %zu: %.9g vs %.9g\n", i, a[i], b[i]); ++bad; }
        maxdiff = std::max(maxdiff, fabsf(a[i] - b[i]));
    }
    std::printf("  %-22s qsa_decode_attn_batch  cap=%lld  n_chunks=%lld | default vs v3-g8  "
                "%s (max abs diff %.3g)\n", label, (long long) in.cap, (long long) in.n_chunks,
                bad == 0 ? "PASS (bit-for-bit)" : "FAIL", maxdiff);
    return bad == 0 ? 0 : 1;
}

int main(int argc, char** argv) {
    int device = 0, iters = 200, trace = 0;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--device") == 0 && i + 1 < argc) device = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--iters") == 0 && i + 1 < argc) iters = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--trace") == 0) trace = 1;
        else { std::fprintf(stderr, "usage: qsa_attn_g8 [--device N] [--iters 200] [--trace]\n"); return 2; }
    }
    int devices = 0; if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) { std::puts("SKIP: no GPU"); return 77; }
    ck(cudaSetDevice(device), "SetDevice"); ck(cudaStreamCreate(&st), "Stream");
    cudaDeviceProp prop{}; ck(cudaGetDeviceProperties(&prop, device), "Props");
    // production geometry: 16 query heads / 2 KV (G=8), head_dim 256, rotated 64, NO indexer (dense), INT8 KV.
    k::QsaShapes s = k::qsa_real_shapes();
    s.n_head = 16; s.n_head_kv = 2; s.head_dim = 256; s.n_rot = 64; s.idx_n_head = 0; s.idx_dim = 0; s.idx_block = 0;
    s.idx_top_k = k::kDenseTopK;

    std::printf("qsa_attn_g8 on device %d (%s %d.%d), %d iters, kv_mode=1 (INT8 KV), 16 query heads/2 KV (G=8), dense\n\n",
                device, prop.name, prop.major, prop.minor, iters);

    const char* trace_env = std::getenv("STRATA_ATTN_V3_G8_TRACE");
    if (!trace_env) std::putenv(const_cast<char*>("STRATA_ATTN_V3_G8_TRACE=1"));

    int fails = 0;
    std::puts("== parity: default attn_chunk_kernel<1,false,8> vs INT8 v3 G=8 (bit-for-bit, production dispatch) ==");
    { In in = build(131072, false); fails |= parity("full 131K  nq1", in); freeIn(in); }
    { In in = build(262144, false); fails |= parity("full 260K  nq1", in); freeIn(in); }
    { In in = build(131052, false); fails |= parity("partial131K", in); freeIn(in); }   // 131052 not a multiple of 64
    { In in = build(131072, true);  fails |= parity("masked 131K", in);  freeIn(in); }   // every 7th page masked
    for (int64_t ctx = 65536; ctx <= 262144; ctx *= 2) { In in = build(ctx, false); fails |= parity((std::to_string(ctx) + " nq1").c_str(), in); freeIn(in); }

    // multi-query (MTP) parity: the scratch/attn are sized for one query here; MTP uses the batched path.  Re-run a
    // 2-query and 4-query window through the batched call on its own scratch to cover n_q > 1 (report 05 req 7).
    {
        const int64_t ctx = 131072, PS = k::qsa_real_shapes().page_size;   // 4
        int64_t cells = (ctx + PS - 1) / PS * PS, cap = cells, chunks = (cap + 63) / 64;
        std::mt19937 rng(20261012u); std::normal_distribution<float> nd(0.0f, 1.0f);
        int8_t* k8 = dalloc<int8_t>((size_t) (cells / PS * 2 * PS * HD));
        int8_t* v8 = dalloc<int8_t>((size_t) (cells / PS * 2 * PS * HD));
        uint16_t* ks = dalloc<uint16_t>((size_t) (cells / PS * 2 * (HD / k::KV_Q8_GROUP)));
        uint16_t* vs = dalloc<uint16_t>((size_t) (cells / PS * 2 * (HD / k::KV_Q8_GROUP)));
        for (int64_t i = 0; i < cells / PS * 2 * PS * HD; ++i) { ((int8_t*)k8)[i] = (int8_t)((rng() % 255) - 127); ((int8_t*)v8)[i] = ((int8_t*)k8)[i]; }
        for (int64_t i = 0; i < cells / PS * 2 * (HD / k::KV_Q8_GROUP); ++i) { ks[i] = vs[i] = f16_scale_q8(rng()); }
        std::vector<int32_t> tab((size_t) (cells / PS)); for (int64_t i = 0; i < cells / PS; ++i) tab[i] = (int32_t) i;
        int32_t* table = upload(tab);
        k::QsaAttnPools pools; pools.page_table = table; pools.k_q = k8; pools.v_q = v8; pools.k_scale = ks; pools.v_scale = vs;
        std::vector<float> hq((size_t) (4 * NH * HD)); for (auto& x : hq) x = nd(rng);
        float* q = upload(hq);
        std::vector<int32_t> hid((size_t) cap); for (int64_t j = 0; j < cap; ++j) hid[j] = (int32_t) j; int32_t* ids = upload(hid);
        std::vector<int32_t> hst(k::kStepCount); hst[k::kStepPos] = (int32_t) ctx; hst[k::kStepNKv] = (int32_t) (ctx + 4);
        hst[k::kStepNBid] = 0; hst[k::kStepWidth] = (int32_t) cap; int32_t* steps = upload(hst);
        float* scrA = dalloc<float>((size_t) k::qsa_decode_attn_scratch_floats(cap, s));
        float* scrB = dalloc<float>((size_t) k::qsa_decode_attn_scratch_floats(cap, s));
        float* a16 = dalloc<float>((size_t) (4 * NH * HD)), * b16 = dalloc<float>((size_t) (4 * NH * HD));
        setenv("STRATA_ATTN_V3_G8", "0", 1); k::qsa_decode_attn_batch(q, pools, ids, steps, cap, s, scrA, a16, 4, st);
        setenv("STRATA_ATTN_V3_G8", "1", 1); k::qsa_decode_attn_batch(q, pools, ids, steps, cap, s, scrB, b16, 4, st);
        cudaDeviceSynchronize();
        auto A = download(a16, (size_t)(4 * NH * HD)), B = download(b16, (size_t)(4 * NH * HD));
        int bad = 0; for (size_t i = 0; i < A.size(); ++i) if (A[i] != B[i]) ++bad;
        std::printf("  %-22s 4-query MTP window  cap=%lld | default vs v3-g8  %s\n", "multi-query nq4", (long long) cap,
                    bad == 0 ? "PASS (bit-for-bit)" : "FAIL");
        fails |= (bad == 0 ? 0 : 1);
        cudaFree(k8); cudaFree(v8); cudaFree(ks); cudaFree(vs); cudaFree(table); cudaFree(q); cudaFree(ids); cudaFree(steps);
        cudaFree(scrA); cudaFree(scrB); cudaFree(a16); cudaFree(b16);
    }

    // == performance: CUDA-event timing of the two kernels at 131K and 260K, and the effective memory bandwidth. ==
    std::puts("\n== performance (CUDA events, median of 3 x back-to-back) ==");
    for (int64_t ctx : {131072LL, 262144LL}) {
        In in = build(ctx, false);
        // warm both kernels so first-launch faults are paid once, then time each
        for (int w = 0; w < 8; ++w) run_one(in, in.attnA, 0);
        for (int w = 0; w < 8; ++w) run_one(in, in.attnB, 1);
        Timer t;
        double us_def = t.us([&]{ run_one(in, in.attnA, 0); }, iters);
        double us_v3 = t.us([&]{ run_one(in, in.attnB, 1); }, iters);
        const int64_t kv_bytes = in.cap * 1024;             // K+V int8, one KV head pair: cap*256*2*2 ... x1 query
        const int64_t scr_bytes = (size_t) in.n_chunks * NH * HD * 4;   // part_acc partials written
        const double eff_gbs = (double) (kv_bytes + scr_bytes) / (us_def / 1000.0) / 1e9;
        std::printf("  ctx=%lld cells  (%.0f MB INT8 K+V, %.0f MB partials) :\n", (long long) ctx,
                    (double) in.cap * 1024 / 1e6, (double) scr_bytes / 1e6);
        std::printf("      default attn_chunk_kernel<1,false,8> : %8.2f us/call   (%.3f ms one layer)\n", us_def, us_def / 1000.0);
        std::printf("      INT8 v3 G=8 (STRATA_ATTN_V3_G8)     : %8.2f us/call   (%.3f ms one layer)\n", us_v3, us_v3 / 1000.0);
        double speedup = us_v3 > 0 ? us_def / us_v3 : 0;
        std::printf("      speedup v3/default : %5.2fx   |   effective BW (KV+partials) %.0f GB/s of %.0f GB/s peak\n",
                    speedup, eff_gbs, 912.0);
        std::printf("      x10 QSA layers: default %.2f ms/window, v3-g8 %.2f ms/window\n", 10 * us_def / 1000.0, 10 * us_v3 / 1000.0);
        freeIn(in);
    }
    std::printf("\n%s\n", fails == 0 ? "qsa_attn_g8: ALL PARITY PASS + perf measured" : "qsa_attn_g8: PARITY MISMATCH");
    return fails == 0 ? 0 : 1;
}
