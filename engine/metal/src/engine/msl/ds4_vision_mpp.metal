// metal/src/engine/msl/ds4_vision_mpp.metal - the vision tower's GEMM on the Metal 4 tensor ops (MetalPerformancePrimitives matmul2d),
// BF16 operands read in place with their row strides, FP32 accumulation.  Same contract as vis_gemm in ds4_vision.metal
// (Y[r][o] = sum_k X[r][k] W(o, k)) without alpha: the host folds the attention scale into the softmax.  A separate file, so a Mac
// whose compiler lacks bfloat tensors keeps the plain kernel (DS4_VISION_NO_MPP=1 asks for that too).
//
// Strata's own; the idea of BF16 weights read in place by a tensor-op matmul follows metal/dense_mm.metal of this tree.
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp::tensor_ops;

struct VisMppArgs {
    device const bfloat* X;
    device const bfloat* W;
    device uchar* Y;
    int M, N, K, ldx, ldw, ldy;
};

// WN: W is [K][N] (o contiguous), else [N][K].  OB: Y is BF16, else FP32.
template <int TM, int TN, int SG, bool WN, bool OB>
kernel void vis_mpp(constant VisMppArgs& a [[buffer(0)]], uint3 tg [[threadgroup_position_in_grid]]) {
    using E2 = dextents<int32_t, 2>;
    tensor<device bfloat, E2, tensor_inline> tA((device bfloat*) a.X, E2(a.K, a.M), array<int32_t, 2>{1, a.ldx});
    constexpr auto desc = matmul2d_descriptor(TM, TN, dynamic_length_v<int>, false, !WN, false);
    matmul2d<desc, execution_simdgroups<SG>> op;
    auto mA = tA.slice(0, int(tg.y) * TM);
    if (WN) {
        tensor<device bfloat, E2, tensor_inline> tB((device bfloat*) a.W, E2(a.N, a.K), array<int32_t, 2>{1, a.ldw});
        auto mB = tB.slice(int(tg.x) * TN, 0);
        if (OB) {
            tensor<device bfloat, E2, tensor_inline> tY((device bfloat*) a.Y, E2(a.N, a.M), array<int32_t, 2>{1, a.ldy});
            auto mY = tY.slice(int(tg.x) * TN, int(tg.y) * TM);
            op.run(mA, mB, mY);
        } else {
            tensor<device float, E2, tensor_inline> tY((device float*) a.Y, E2(a.N, a.M), array<int32_t, 2>{1, a.ldy});
            auto mY = tY.slice(int(tg.x) * TN, int(tg.y) * TM);
            op.run(mA, mB, mY);
        }
    } else {
        tensor<device bfloat, E2, tensor_inline> tB((device bfloat*) a.W, E2(a.K, a.N), array<int32_t, 2>{1, a.ldw});
        auto mB = tB.slice(0, int(tg.x) * TN);
        if (OB) {
            tensor<device bfloat, E2, tensor_inline> tY((device bfloat*) a.Y, E2(a.N, a.M), array<int32_t, 2>{1, a.ldy});
            auto mY = tY.slice(int(tg.x) * TN, int(tg.y) * TM);
            op.run(mA, mB, mY);
        } else {
            tensor<device float, E2, tensor_inline> tY((device float*) a.Y, E2(a.N, a.M), array<int32_t, 2>{1, a.ldy});
            auto mY = tY.slice(int(tg.x) * TN, int(tg.y) * TM);
            op.run(mA, mB, mY);
        }
    }
}

template [[host_name("vis_mpp_k_f32_128_64")]] kernel decltype(vis_mpp<128, 64, 4, false, false>) vis_mpp<128, 64, 4, false, false>;
template [[host_name("vis_mpp_k_f32_64_64")]] kernel decltype(vis_mpp<64, 64, 4, false, false>) vis_mpp<64, 64, 4, false, false>;
template [[host_name("vis_mpp_n_f32_64_64")]] kernel decltype(vis_mpp<64, 64, 4, true, false>) vis_mpp<64, 64, 4, true, false>;
template [[host_name("vis_mpp_n_bf_64_64")]] kernel decltype(vis_mpp<64, 64, 4, true, true>) vis_mpp<64, 64, 4, true, true>;
