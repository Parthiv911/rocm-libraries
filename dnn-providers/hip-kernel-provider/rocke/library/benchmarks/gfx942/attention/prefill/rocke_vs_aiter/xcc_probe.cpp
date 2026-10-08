#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>

#define CHECK_HIP(x) do {                                      \
    hipError_t err = (x);                                      \
    if (err != hipSuccess) {                                   \
        fprintf(stderr, "%s:%d: %s\n",                         \
                __FILE__, __LINE__, hipGetErrorString(err));    \
        std::exit(1);                                          \
    }                                                          \
} while (0)

constexpr int NUM_BLOCKS = 304;
constexpr int THREADS    = 512;

// Match the attention kernel closely enough to force ~1 WG/CU.
// Current attention kernel is ~51.2 KB LDS/WG.
constexpr int LDS_BYTES = 51200;

// gfx942 XCC_ID:
//   register = 20
//   offset   = 0
//   width    = 4 bits
//
// ROCm GETREG_IMMED encoding:
//   ((SIZE_MINUS_1) << 11) | (OFFSET << 6) | REG
//
// SIZE_MINUS_1 = 3 for a 4-bit field:
//   (3 << 11) | (0 << 6) | 20 = 0x1814
//
// HIP compiles this TU twice: once for the host and once for the GPU.
// The AMDGCN builtin is only valid in the device compilation, so guard
// it with __HIP_DEVICE_COMPILE__ rather than #error'ing in the host pass.
__device__ __forceinline__
unsigned read_xcc_id()
{
#if defined(__HIP_DEVICE_COMPILE__)
    return __builtin_amdgcn_s_getreg(0x1814);
#else
    return 0;
#endif
}

__global__
void xcc_probe(unsigned *block_xcc, unsigned *hist)
{
    extern __shared__ unsigned char lds[];

    // Touch the requested dynamic LDS so this WG genuinely carries the
    // same approximate LDS residency constraint as the attention kernel.
    if (threadIdx.x == 0) {
        lds[0] = static_cast<unsigned char>(blockIdx.x);
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        const unsigned xcc = read_xcc_id();

        block_xcc[blockIdx.x] = xcc;

        if (xcc < 16) {
            atomicAdd(&hist[xcc], 1u);
        }

        // Keep the WG resident long enough that the scheduler has a chance
        // to populate the complete persistent grid instead of immediately
        // recycling finished blocks onto the same CUs.
        const unsigned long long start = clock64();
        constexpr unsigned long long HOLD_CYCLES = 10000000ULL;

        while ((clock64() - start) < HOLD_CYCLES) {
            asm volatile("s_nop 0");
        }
    }

    __syncthreads();
}

int main()
{
    unsigned *d_block_xcc = nullptr;
    unsigned *d_hist = nullptr;

    CHECK_HIP(hipMalloc(&d_block_xcc, NUM_BLOCKS * sizeof(unsigned)));
    CHECK_HIP(hipMalloc(&d_hist, 16 * sizeof(unsigned)));

    CHECK_HIP(
        hipMemset(
            d_block_xcc,
            0xff,
            NUM_BLOCKS * sizeof(unsigned)
        )
    );

    CHECK_HIP(
        hipMemset(
            d_hist,
            0,
            16 * sizeof(unsigned)
        )
    );

    CHECK_HIP(
        hipFuncSetAttribute(
            reinterpret_cast<const void*>(xcc_probe),
            hipFuncAttributeMaxDynamicSharedMemorySize,
            LDS_BYTES
        )
    );

    hipLaunchKernelGGL(
        xcc_probe,
        dim3(NUM_BLOCKS),
        dim3(THREADS),
        LDS_BYTES,
        0,
        d_block_xcc,
        d_hist
    );

    CHECK_HIP(hipGetLastError());
    CHECK_HIP(hipDeviceSynchronize());

    unsigned block_xcc[NUM_BLOCKS];
    unsigned hist[16] = {};

    CHECK_HIP(
        hipMemcpy(
            block_xcc,
            d_block_xcc,
            sizeof(block_xcc),
            hipMemcpyDeviceToHost
        )
    );

    CHECK_HIP(
        hipMemcpy(
            hist,
            d_hist,
            sizeof(hist),
            hipMemcpyDeviceToHost
        )
    );

    std::printf("\nXCC histogram\n");
    std::printf("=============\n");

    unsigned total = 0;

    for (int i = 0; i < 16; ++i) {
        if (hist[i] != 0) {
            std::printf(
                "XCC %2d : %3u workgroups\n",
                i,
                hist[i]
            );
            total += hist[i];
        }
    }

    std::printf("total  : %3u workgroups\n\n", total);

    std::printf("Block -> XCC mapping\n");
    std::printf("====================\n");

    for (int i = 0; i < NUM_BLOCKS; ++i) {
        std::printf(
            "%3d -> %u%s",
            i,
            block_xcc[i],
            ((i + 1) % 8 == 0) ? "\n" : "    "
        );
    }

    if (NUM_BLOCKS % 8 != 0) {
        std::printf("\n");
    }

    CHECK_HIP(hipFree(d_block_xcc));
    CHECK_HIP(hipFree(d_hist));

    return 0;
}