// opencl/probe/ocl_row_repro.cpp - run the peer's own generated row kernel on the bytes the peer itself
// uploaded, with no host code in between.
//
// The parity run says: the blob the peer holds is byte-identical to the reference's, the q8_1 it reads is
// byte-identical, and the geometry it derived (down_off 1049600, d_row 360, d_chunk 40) is the reference's
// - yet its row 0 is 447.458 where both engine kernels say 322.  A re-implementation of the kernel from
// the same source text reproduces 322, so either the driver is building something other than the text, or
// the text is not what it looks like.  This runs that text directly on those bytes to tell the two apart.
//
//     ocl_row_repro [kernel.cl] [blob.bin] [hq.bin] [row_off] [row_count]

#define CL_TARGET_OPENCL_VERSION 200
#include <CL/cl.h>

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

bool read_file(const char* path, std::vector<uint8_t>& out) {
    FILE* f = std::fopen(path, "rb");
    if (!f) return false;
    std::fseek(f, 0, SEEK_END);
    const long n = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    out.resize((size_t) n);
    const size_t got = std::fread(out.data(), 1, (size_t) n, f);
    std::fclose(f);
    return got == (size_t) n;
}

void dump_log(cl_program p, cl_device_id d, const char* what) {
    size_t n = 0;
    clGetProgramBuildInfo(p, d, CL_PROGRAM_BUILD_LOG, 0, nullptr, &n);
    std::vector<char> log(n + 1, 0);
    if (n > 0) clGetProgramBuildInfo(p, d, CL_PROGRAM_BUILD_LOG, n, log.data(), nullptr);
    std::printf("%s build log:\n%s\n", what, log.data());
}

}  // namespace

int main(int argc, char** argv) {
    const char* kpath = argc > 1 ? argv[1] : "ocl_src_IQ4_NL_at_40_calls_row.cl";
    const char* bpath = argc > 2 ? argv[2] : "peer_blob_l0.bin";
    const char* hpath = argc > 3 ? argv[3] : "peer_hq_l0.bin";
    const uint32_t blk_head = argc > 4 ? (uint32_t) std::atoi(argv[4]) : 64;   // peer_blob_*.bin hdr
    const int n_show = argc > 5 ? std::atoi(argv[5]) : 6;

    std::vector<uint8_t> src;
    if (!read_file(kpath, src)) { std::printf("cannot read %s\n", kpath); return 2; }
    std::vector<uint8_t> blobfile;
    if (!read_file(bpath, blobfile)) { std::printf("cannot read %s\n", bpath); return 2; }
    std::vector<uint8_t> hq;
    if (!read_file(hpath, hq)) { std::printf("cannot read %s\n", hpath); return 2; }

    // ---- platform / device
    cl_platform_id plat = nullptr;
    cl_device_id dev = nullptr;
    cl_uint np = 0;
    clGetPlatformIDs(0, nullptr, &np);
    std::vector<cl_platform_id> plats(np);
    clGetPlatformIDs(np, plats.data(), nullptr);
    for (cl_platform_id p : plats) {
        char name[256] = {0};
        clGetPlatformInfo(p, CL_PLATFORM_NAME, sizeof(name), name, nullptr);
        cl_uint nd = 0;
        clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, 0, nullptr, &nd);
        if (nd == 0) continue;
        std::vector<cl_device_id> ds(nd);
        clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, nd, ds.data(), nullptr);
        for (cl_device_id d : ds) {
            char dn[256] = {0};
            clGetDeviceInfo(d, CL_DEVICE_NAME, sizeof(dn), dn, nullptr);
            if (std::string(dn).find("gfx906") != std::string::npos || std::string(name).find("AMD") != std::string::npos) {
                plat = p;
                dev = d;
                std::printf("device: %s on %s\n", dn, name);
                break;
            }
        }
        if (dev) break;
    }
    if (!dev) { std::printf("no AMD GPU\n"); return 2; }

    cl_int rc = CL_SUCCESS;
    cl_context ctx = clCreateContext(nullptr, 1, &dev, nullptr, nullptr, &rc);
    if (rc != CL_SUCCESS) { std::printf("ctx %d\n", rc); return 2; }
    cl_command_queue q = clCreateCommandQueueWithProperties(ctx, dev, nullptr, &rc);
    if (rc != CL_SUCCESS) { std::printf("queue %d\n", rc); return 2; }

    const char* sp = (const char*) src.data();
    size_t slen = src.size();
    cl_program prog = clCreateProgramWithSource(ctx, 1, &sp, &slen, &rc);
    if (rc != CL_SUCCESS) { std::printf("prog %d\n", rc); return 2; }
    const cl_int br = clBuildProgram(prog, 1, &dev, "-cl-std=CL2.0", nullptr, nullptr);
    if (br != CL_SUCCESS) { dump_log(prog, dev, "iq_row"); return 2; }
    cl_kernel k = clCreateKernel(prog, "iq_row", &rc);
    bool dbg_mode = false, lane_mode = false;
    if (rc != CL_SUCCESS) {
        k = clCreateKernel(prog, "iq_row_dbg", &rc);
        dbg_mode = (rc == CL_SUCCESS);
        if (rc != CL_SUCCESS) {
            k = clCreateKernel(prog, "iq_lane", &rc);
            lane_mode = (rc == CL_SUCCESS);
            if (rc != CL_SUCCESS) { std::printf("kernel %d\n", rc); return 2; }
        }
    }

    // ---- the buffers, exactly as the peer fills them
    const uint64_t blob_off = (blobfile.size() >= blk_head + 8) ? *(const uint64_t*) blobfile.data() : 0;
    const std::vector<uint8_t> wt(blobfile.begin() + blk_head, blobfile.end());
    uint64_t hdr[8] = {0};
    if (blobfile.size() >= 64) std::memcpy(hdr, blobfile.data(), 64);
    std::printf("peer hdr: ent_off %llu down_off %llu d_row %llu gu_row %llu d_chunk %llu gu_chunk %llu "
                "bytes %llu types %llu (blob_off field %llu)\n",
                (unsigned long long) hdr[0], (unsigned long long) hdr[1], (unsigned long long) hdr[2],
                (unsigned long long) hdr[3], (unsigned long long) hdr[4], (unsigned long long) hdr[5],
                (unsigned long long) hdr[6], (unsigned long long) hdr[7], (unsigned long long) blob_off);

    // kvalues_iq4nl, uploaded as signed bytes
    static const int8_t KV[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
    uint8_t kvb[16];
    for (int i = 0; i < 16; ++i) kvb[i] = (uint8_t) KV[i];
    uint32_t dummy[8] = {0};

    cl_mem mwt = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, wt.size(), (void*) wt.data(), &rc);
    cl_mem mxq = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, hq.size(), hq.data(), &rc);
    cl_mem mt2 = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(dummy), dummy, &rc);
    cl_mem mt3 = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(dummy), dummy, &rc);
    cl_mem mkv = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 16, kvb, &rc);
    const uint32_t N_ROW_E = (uint32_t) hdr[5] * 32u * 0;   // set below
    (void) N_ROW_E;
    const uint32_t ACT_BLK = 20;                      // n_ff / 32
    const uint32_t ROWB = (uint32_t) hdr[2];          // d_row
    const uint32_t EXTRA = (uint32_t) hdr[1];         // down_off
    const uint32_t OUTSTRIDE = 2560;                  // n_embd
    const uint32_t NROWE = 2560;
    const uint32_t BPE = (NROWE + 3u) / 4u;           // IQ_R == 4
    cl_mem mout = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, (size_t) OUTSTRIDE * 4, nullptr, &rc);
    cl_mem meo = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(uint64_t),
                                (void*) &hdr[0], &rc);   // ent_off = the slot's own offset in the arena
    uint32_t act0 = 0;
    cl_mem mea = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 4, &act0, &rc);

    int ai = 0;
    clSetKernelArg(k, ai++, sizeof(cl_mem), &mwt);
    clSetKernelArg(k, ai++, sizeof(cl_mem), &mxq);
    clSetKernelArg(k, ai++, sizeof(cl_mem), &mt2);
    clSetKernelArg(k, ai++, sizeof(cl_mem), &mt3);
    clSetKernelArg(k, ai++, sizeof(cl_mem), &mkv);
    clSetKernelArg(k, ai++, sizeof(cl_mem), &mout);
    clSetKernelArg(k, ai++, sizeof(cl_mem), &meo);
    clSetKernelArg(k, ai++, sizeof(cl_mem), &mea);
    const uint32_t n_ent = 1;
    clSetKernelArg(k, ai++, sizeof(n_ent), &n_ent);
    clSetKernelArg(k, ai++, sizeof(BPE), &BPE);
    clSetKernelArg(k, ai++, sizeof(NROWE), &NROWE);
    clSetKernelArg(k, ai++, sizeof(ROWB), &ROWB);
    clSetKernelArg(k, ai++, sizeof(EXTRA), &EXTRA);
    clSetKernelArg(k, ai++, sizeof(ACT_BLK), &ACT_BLK);
    clSetKernelArg(k, ai++, sizeof(OUTSTRIDE), &OUTSTRIDE);
    cl_mem mdbg = nullptr;
    if (dbg_mode) {
        mdbg = clCreateBuffer(ctx, CL_MEM_READ_WRITE, 1024 * sizeof(float), nullptr, &rc);
        clSetKernelArg(k, ai++, sizeof(cl_mem), &mdbg);
    }

    // NOTE: the peer uploads the expert into the arena at slot_off and then passes ent_off = slot_off, with
    // `wt` being the whole arena.  Here `wt` is the single expert, so ent_off must be 0; hdr[0] is patched.
    uint64_t e0 = 0;
    clEnqueueWriteBuffer(q, meo, CL_TRUE, 0, sizeof(e0), &e0, 0, nullptr, nullptr);

    // the global size is in WORK-ITEMS: bpe work-groups of wg lanes, which is what the peer passes too
    const size_t lsz = 64, gsz = (size_t) BPE * lsz;
    std::printf("launch: bpe %u grid %zu lsz %zu n_row_e %u out_stride %u row_bytes %u extra %u act_blk %u\n",
                BPE, gsz, lsz, NROWE, OUTSTRIDE, ROWB, EXTRA, ACT_BLK);
    rc = clEnqueueNDRangeKernel(q, k, 1, nullptr, &gsz, &lsz, 0, nullptr, nullptr);
    if (rc != CL_SUCCESS) { std::printf("ndrange %d\n", rc); return 2; }
    std::vector<float> out((size_t) OUTSTRIDE);
    clEnqueueReadBuffer(q, mout, CL_TRUE, 0, out.size() * 4, out.data(), 0, nullptr, nullptr);

    std::printf("kernel row 0..%d: ", n_show - 1);
    for (int i = 0; i < n_show; ++i) std::printf("%.6g ", (double) out[(size_t) i]);
    std::printf("\n");
    if (n_show == 0) {                      // every row, so a whole-column comparison is possible
        if (FILE* fh = std::fopen("repro_row.bin", "wb")) {
            std::fwrite(out.data(), 4, out.size(), fh);
            std::fclose(fh);
        }
        std::printf("wrote repro_row.bin (%zu floats)\n", out.size());
    }

    if (lane_mode) {
        std::printf("\nlane contributions, outv[r*IQ_WG + lane] (r = 0..3), lanes 0..39:\n");
        for (int r = 0; r < 4; ++r) {
            std::printf(" r%d:", r);
            for (int l = 0; l < 40; ++l) {
                if (l % 4 == 0) std::printf("\n  ");
                std::printf("%4d %-13.7g", l, (double) out[(size_t) r * 64 + l]);
            }
            std::printf("\n");
        }
        return 0;
    }

    if (dbg_mode) {
        std::vector<float> dbg(1024, 0.0f);
        clEnqueueReadBuffer(q, mdbg, CL_TRUE, 0, dbg.size() * 4, dbg.data(), 0, nullptr, nullptr);
        std::printf("\nlane  k  kbx  m   si        dw          dx         qa          qb          acc0\n");
        for (int l = 0; l < 40; ++l) {
            const float* d = &dbg[(size_t) l * 8];
            uint32_t qa = 0, qb = 0;
            std::memcpy(&qa, &d[5], 4);
            std::memcpy(&qb, &d[6], 4);
            std::printf("%4d %4d %4d %3d  %-10.6g %-10.6g %-10.6g 0x%08x 0x%08x %-10.6g\n", l, (int) d[4],
                        (int) d[3], (int) d[4] & 1, (double) d[0], (double) d[1], (double) d[2], qa, qb,
                        (double) d[7]);
        }
        std::printf("\nraw acc per lane (pre-tree):\n");
        for (int l = 0; l < 64; ++l) std::printf("%4d %-12.6g%s", l, (double) dbg[512 + l], (l % 4 == 3) ? "\n" : "");
        std::printf("\n");
    }
    return 0;
}
