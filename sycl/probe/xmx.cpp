// xmx.cpp: which joint_matrix (XMX) shapes and types this device offers.  icpx -fsycl xmx.cpp -o xmx && ./xmx
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/matrix/matrix.hpp>
#include <cstdio>
int main() {
    namespace exp = sycl::ext::oneapi::experimental;
    sycl::queue q;
    auto d = q.get_device();
    std::printf("%s, sub-group sizes:", d.get_info<sycl::info::device::name>().c_str());
    for (auto s : d.get_info<sycl::info::device::sub_group_sizes>()) std::printf(" %zu", s);
    std::printf("\n");
    auto combos = d.get_info<exp::info::device::matrix_combinations>();
    std::printf("%zu joint_matrix combinations (M N K  A B C D):\n", combos.size());
    auto name = [](exp::matrix::matrix_type t) {
        using T = exp::matrix::matrix_type;
        switch (t) { case T::bf16: return "bf16"; case T::fp16: return "fp16"; case T::tf32: return "tf32"; case T::fp32: return "fp32";
                     case T::fp64: return "fp64"; case T::sint8: return "s8"; case T::uint8: return "u8"; case T::sint16: return "s16";
                     case T::uint16: return "u16"; case T::sint32: return "s32"; case T::uint32: return "u32"; case T::sint64: return "s64";
                     case T::uint64: return "u64"; default: return "?"; }
    };
    for (auto& c : combos)
        std::printf("  %2zu %2zu %2zu  %s %s %s %s  (max %zu %zu %zu)\n", c.msize, c.nsize, c.ksize, name(c.atype), name(c.btype),
                    name(c.ctype), name(c.dtype), c.max_msize, c.max_nsize, c.max_ksize);
    return 0;
}
