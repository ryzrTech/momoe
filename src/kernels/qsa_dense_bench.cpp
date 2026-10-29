// src/kernels/qsa_dense_bench.cpp - timing of the QSA decode attention exactly as qwen35moe (Ornith 1.5 35B-A3B)
// runs it: the model has NO indexer, so every cached cell is attended (qsa_select_all) and the attention works over
// the WHOLE populated context, not a 2051-cell selection. This times qsa_decode_attn_batch at the model's real
// geometry (16 query heads / 2 KV = G=8, head_dim 256, INT8 KV, kv_mode=1) at several context lengths, CUDA-event
// timing (median of 3). It tells us whether dense attention is the context-scaling cost the plain 2051-cell bench
// (qsa_rows_bench) could not see.
//
//   qsa_dense_bench [--device N] [--iters 200] [--ctx 131072] [--ctx 262144]
#include "strata/kernels/kv_q8.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"

#include <cuda_runtime.h>
#include <chrono>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

namespace k = strata::kernels;

namespace {
cudaStream_t st = nullptr;
void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) { std::fprintf(stderr, "qsa_dense_bench: %s: %s\n", what, cudaGetErrorString(e)); std::exit(2); }
}
template <typename T> T* dalloc(size_t n) { T* p = nullptr; ck(cudaMalloc((void**) &p, n * sizeof(T) + 256), "malloc");
    ck(cudaMemset(p, 0, n * sizeof(T) + 256), "memset"); return p; }
template <typename T> T* upload(const std::vector<T>& h) { T* d = dalloc<T>(h.size());
    ck(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice), "upload"); return d; }
template <typename T> std::vector<T> download(const T* d, size_t n) { std::vector<T> h(n);
    ck(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost), "download"); return h; }
uint16_t f16_scale_q8(uint32_t r) { return (uint16_t) ((9u << 10) | (r & 0x3FFu)); }
struct Timer {
    cudaEvent_t e0 = nullptr, e1 = nullptr;
    Timer() { ck(cudaEventCreate(&e0), "ev"); ck(cudaEventCreate(&e1), "ev"); }
    ~Timer() { cudaEventDestroy(e0); cudaEventDestroy(e1); }
    template <typename F> double us(F&& f, int iters) {
        double r[3];
        for (int a = 0; a < 3; ++a) { for (int i = 0; i < 4; ++i) f(); ck(cudaDeviceSynchronize(), "warm");
            ck(cudaEventRecord(e0, st), "r"); for (int i = 0; i < iters; ++i) f(); ck(cudaEventRecord(e1, st), "r");
            ck(cudaEventSynchronize(e1), "s"); float ms = 0.0f; ck(cudaEventElapsedTime(&ms, e0, e1), "el"); r[a] = 1000.0 * ms / iters; }
        std::sort(r, r + 3); return r[1];
    }
};
}  // namespace

int main(int argc, char** argv) {
    int device = 0, iters = 200;
    std::vector<int64_t> ctxs = {131072, 262144};
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--device") == 0 && i + 1 < argc) device = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--iters") == 0 && i + 1 < argc) iters = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--ctx") == 0 && i + 1 < argc) ctxs.clear(), ctxs.push_back(std::atoll(argv[++i]));
        else { std::fprintf(stderr, "usage: qsa_dense_bench [--device N] [--iters 200] [--ctx 131072] [--ctx 262144]\n"); return 2; }
    }
    int devices = 0; if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) { std::puts("SKIP: no GPU"); return 77; }
    ck(cudaSetDevice(device), "SetDevice"); ck(cudaStreamCreate(&st), "Stream");
    cudaDeviceProp prop{}; ck(cudaGetDeviceProperties(&prop, device), "Props");

    // ---- the model's real QSA geometry: 16 query heads / 2 KV (G=8), head_dim 256, rotated 64, NO indexer (dense),
    // ---- INT8 KV (kv_mode=1) exactly as production passes to qsa_decode_attn_step.
    k::QsaShapes s = k::qsa_real_shapes();
    s.n_head = 16; s.n_head_kv = 2; s.head_dim = 256; s.n_rot = 64;
    s.idx_n_head = 0; s.idx_dim = 0; s.idx_block = 0; s.idx_top_k = k::kDenseTopK;
    static_assert(8 * 2 == 16, "G=8 for this model");

    std::mt19937 rng(20261012u); std::normal_distribution<float> nd(0.0f, 1.0f);

    std::printf("qsa_dense_bench on device %d (%s %d.%d), %d iters, kv_mode=1 (INT8 KV), 16 query heads/2 KV (G=8)\n\n",
                device, prop.name, prop.major, prop.minor, iters);

    for (int64_t ctx : ctxs) {
        const int64_t PS = s.page_size;
        const int64_t cells = (ctx + PS - 1) / PS * PS;
        const int64_t pages = cells / PS, prows = pages * 2 * PS;
        const int64_t cap = cells;

        // INT8 KV pools: k_q/v_q int8 codes, k_scale/v_scale uint16 (one per KV_Q8_GROUP=64 values), identity table
        std::vector<int8_t> c8((size_t) (prows * 256));
        for (auto& x : c8) x = (int8_t) ((int) (rng() % 255u) - 127);
        int8_t* k8 = upload(c8);
        for (auto& x : c8) x = (int8_t) ((int) (rng() % 255u) - 127);
        int8_t* v8 = upload(c8);
        std::vector<uint16_t> sc((size_t) (prows * (256 / k::KV_Q8_GROUP)));
        for (auto& x : sc) x = f16_scale_q8(rng());
        uint16_t* k8s = upload(sc);
        for (auto& x : sc) x = f16_scale_q8(rng());
        uint16_t* v8s = upload(sc);
        std::vector<int32_t> tab((size_t) pages); for (int64_t i = 0; i < pages; ++i) tab[i] = (int32_t) i;
        int32_t* table = upload(tab);
        k::QsaAttnPools pools; pools.page_table = table; pools.k_q = k8; pools.v_q = v8; pools.k_scale = k8s; pools.v_scale = v8s;

        std::vector<float> hq((size_t) (16 * 256)); for (auto& x : hq) x = nd(rng); float* q = upload(hq);
        std::vector<int32_t> hid((size_t) cap); for (int64_t j = 0; j < cap; ++j) hid[j] = (int32_t) j;
        int32_t* ids = upload(hid);
        std::vector<int32_t> hst(k::kStepCount);
        hst[k::kStepPos] = (int32_t) ctx; hst[k::kStepNKv] = (int32_t) (ctx + 1);
        hst[k::kStepNBid] = 0;   // unused by dense cell_row (it pages by page_size, not idx_block)
        hst[k::kStepWidth] = (int32_t) cap;   // dense: the whole context is the selection width
        int32_t* steps = upload(hst);

        int64_t n_q = 1;
        size_t scr = (size_t) k::qsa_decode_attn_scratch_floats(cap, s);
        float* scratch = dalloc<float>(scr);
        float* attn = dalloc<float>((size_t) (16 * 256));

        const int64_t n_chunks = (cap + 63) / 64;

        if (cudaGetLastError() != cudaSuccess)
            std::printf("  [%lld] LAUNCH ERROR %s\n", (long long) ctx, cudaGetErrorString(cudaGetLastError()));

        // authoritative: 12 individually-synced calls, min wins. Pure host-side chrono (immune to the cuda-event
        // first-launch-fault accounting bug the back-to-back timer hit). Warm first so faults are paid once.
        for (int w = 0; w < 8; ++w) k::qsa_decode_attn_batch(q, pools, ids, steps, cap, s, scratch, attn, n_q, st);
        cudaDeviceSynchronize();
        double best = 1e9;
        for (int i = 0; i < 12; ++i) {
            auto t0 = std::chrono::steady_clock::now();
            k::qsa_decode_attn_batch(q, pools, ids, steps, cap, s, scratch, attn, n_q, st);
            cudaDeviceSynchronize();
            auto t1 = std::chrono::steady_clock::now();
            best = std::min(best, std::chrono::duration<double>(t1 - t0).count() * 1000.0);
        }
        double attn_ms = best;
        // host-side wall time of a single call (launch + implicit drain), to separate launch overhead from GPU work
        cudaDeviceSynchronize();
        auto h0 = std::chrono::steady_clock::now();
        k::qsa_decode_attn_batch(q, pools, ids, steps, cap, s, scratch, attn, n_q, st);
        cudaDeviceSynchronize();
        auto h1 = std::chrono::steady_clock::now();
        double host_ms = std::chrono::duration<double>(h1 - h0).count() * 1000.0;

        // active partial count + finite-output check.
        // Layout: part_acc[chunks*16*256] | part_m[chunks*16] | part_l[chunks*16]
        const size_t off_m = (size_t) n_chunks * 16 * 256;
        auto pm = download((const float*) (scratch + off_m), (size_t) n_chunks * 16);
        auto pl = download((const float*) (scratch + off_m + (size_t) n_chunks * 16), (size_t) n_chunks * 16);
        int activec = 0; float psum = 0;
        for (int i = 0; i < n_chunks * 16; ++i) {
            if (std::isfinite(pm[i]) && pm[i] != -FLT_MAX) activec++;
            if (pl[i] > 0) psum += pl[i];
        }
        auto hv = download((const float*) attn, 16 * 256);
        float sum = 0, rms = 0; for (auto x : hv) { sum += fabsf(x); rms += x * x; }
        const int64_t total_p = n_chunks * 16;
        std::printf("  [%lld cells] cap=%lld n_chunks=%lld grid=%lld blocks | active partials %lld/%lld (dense=%s) | "
                    "part_l sum=%.1f | attn rms=%.3g\n",
                    (long long) ctx, (long long) cap, (long long) n_chunks, (long long) n_chunks * 2,
                    (long long) activec, (long long) total_p, activec == total_p ? "full" : "PARTIAL",
                    psum, sqrtf(rms / (16 * 256)));
        std::printf("      host-side single-call wall: %.1f ms  (gpu-event: %.1f ms) -> launch/host overhead is %s\n",
                    host_ms, attn_ms, (host_ms - attn_ms) > 5.0 ? "LARGE (not GPU compute)" : "normal");
        std::printf("      attention: %.3f ms/call (= one QSA layer)  ->  x4 layers %.3f ms/window, x8 layers %.3f ms/window\n",
                    attn_ms, 4 * attn_ms / 1000.0, 8 * attn_ms / 1000.0);
        std::printf("      INT8 KV pools %.1f MB, scratch %.1f MB\n\n",
                    (size_t) prows * 256 / 1e6, scr * 4 / 1e6);

        void* mems[] = { (void*) k8, (void*) v8, (void*) k8s, (void*) v8s, (void*) table, (void*) q,
                         (void*) ids, (void*) steps, (void*) scratch, (void*) attn };
        for (void* p : mems) ck(cudaFree(p), "free");
    }
    std::puts("qsa_dense_bench: done");
    return 0;
}
