#include "spin8/query_adapter.h"
#include "spin8/query_v3.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>
extern "C" {
#include "../third_party/sha256/sha256.h"
int spin8_query_adapter_c_check(const spin8_query_adapter *);
}

namespace {
void check(bool ok, const char *message) {
  if (!ok) {
    std::fprintf(stderr, "query adapter: %s\n", message);
    std::exit(1);
  }
}
std::string hex(const uint8_t *data) {
  constexpr char alphabet[] = "0123456789abcdef";
  std::string result;
  for (size_t i = 0; i < 32; ++i) {
    result += alphabet[data[i] >> 4];
    result += alphabet[data[i] & 15];
  }
  return result;
}
void digest(const char *text, uint8_t *out) {
  spin8_sidecar_sha256_hash(out, reinterpret_cast<const unsigned char *>(text),
                            std::strlen(text));
}
void hashes() {
  std::array<uint8_t, 32> value{};
  digest("", value.data());
  check(hex(value.data()) ==
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        "SHA empty");
  digest("abc", value.data());
  check(hex(value.data()) ==
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        "SHA abc");
  const std::string long_value(1000000, 'a');
  digest(long_value.c_str(), value.data());
  check(hex(value.data()) ==
            "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0",
        "SHA multiblock million a");
}
float pattern(size_t i, size_t section) {
  if (i % 97 == 0) {
    const uint32_t negative_zero = 0x80000000;
    float value;
    std::memcpy(&value, &negative_zero, sizeof(value));
    return value;
  }
  return static_cast<float>(static_cast<int>(i % 31) - 15 +
                            static_cast<int>(section)) /
         1024.f;
}
void pattern_check(const float *values, size_t count, size_t section) {
  for (size_t i = 0; i < count; ++i) {
    const auto expected = pattern(i, section);
    check(std::memcmp(values + i, &expected, sizeof(float)) == 0,
          "independent per-head F32 bits including negative zero");
  }
}
void behavior(const spin8_query_adapter_layer_view &v) {
  // Compare the borrowed F32 parameters with a separately calculated MLP and
  // rational chart for EACH head. This does not duplicate the parser.
  const size_t width = v.spec.head_width;
  std::vector<double> q(width);
  for (size_t i = 0; i < width; ++i)
    q[i] = (static_cast<int>(i % 13) - 6) / 16.;
  for (size_t head = 0; head < v.spec.q_heads; ++head) {
    std::vector<double> win(v.win + head * 8 * width,
                            v.win + (head + 1) * 8 * width);
    std::array<double, 8> bin{}, hidden{}, actual{};
    std::array<double, 56> wout{};
    std::array<double, 7> bout{}, tangent{};
    for (size_t j = 0; j < 8; ++j) {
      bin[j] = v.bin[head * 8 + j];
      double sum = bin[j];
      for (size_t k = 0; k < width; ++k)
        sum += win[j * width + k] * q[k];
      hidden[j] = std::tanh(sum);
    }
    double norm = 0;
    for (size_t j = 0; j < 7; ++j) {
      bout[j] = v.bout[head * 7 + j];
      tangent[j] = bout[j];
      for (size_t k = 0; k < 8; ++k) {
        wout[j * 8 + k] = v.wout[head * 56 + j * 8 + k];
        tangent[j] += wout[j * 8 + k] * hidden[k];
      }
      norm += tangent[j] * tangent[j];
    }
    const double tau = std::tan(std::acos(-1.) / 8);
    const double scale = tau / std::sqrt(tau * tau + norm);
    double u_norm = 0;
    for (auto &value : tangent) {
      value *= scale;
      u_norm += value * value;
    }
    const double c0 = 1 / std::sqrt(1 + u_norm);
    spin8_query_v3_controller *controller = nullptr;
    check(spin8_query_v3_controller_create(
              3, width, win.data(), win.size() * 8, bin.data(), sizeof(bin),
              wout.data(), sizeof(wout), bout.data(), sizeof(bout),
              &controller) == SPIN8_OK,
          "controller from immutable view");
    check(spin8_query_v3_controller_spinors(controller, SPIN8_F64, q.data(),
                                            q.size() * 8, width, actual.data(),
                                            sizeof(actual), 8, 1) == SPIN8_OK,
          "controller spinor");
    check(std::abs(actual[0] - c0) < 1e-14, "independent rational chart c0");
    for (size_t j = 0; j < 7; ++j)
      check(std::abs(actual[j + 1] - tangent[j] * c0) < 1e-14,
            "independent rational chart tangent");
    std::vector<double> transformed(width);
    check(spin8_query_v3_controller_apply(
              controller, SPIN8_F64, q.data(), q.size() * 8, width,
              transformed.data(), transformed.size() * 8, width, 1) == SPIN8_OK,
          "view apply");
    for (size_t j = width - width % 8; j < width; ++j)
      check(transformed[j] == q[j], "tail width copied exactly");
    if (std::all_of(wout.begin(), wout.end(),
                    [](double x) { return x == 0; }) &&
        std::all_of(bout.begin(), bout.end(), [](double x) { return x == 0; }))
      check(transformed == q, "zero output parameters give exact identity");
    spin8_query_v3_controller_free(controller);
  }
}
spin8_query_adapter_layer_spec specs[] = {{2, 24, 4, 256}, {7, 2, 1, 11}};
spin8_query_adapter_expectations expectations(uint32_t count, uint8_t *models) {
  spin8_query_adapter_expectations e{};
  e.abi_version = SPIN8_QUERY_ADAPTER_ABI_VERSION;
  e.struct_bytes = sizeof(e);
  e.model_count = count;
  e.layer_count = 2;
  e.model_digests = models;
  e.model_digests_bytes = count * 32;
  digest("profile", e.profile_digest);
  digest("position", e.position_digest);
  e.layers = specs;
  e.layers_bytes = sizeof(specs);
  return e;
}
void overflow_alias_regressions() {
  alignas(spin8_query_adapter *) std::array<uint8_t, 64> bytes{};
  alignas(spin8_query_adapter *) std::array<uint8_t, 64> models{};
  alignas(spin8_query_adapter *) spin8_query_adapter_layer_spec layers[] = {
      {2, 24, 4, 256}, {7, 2, 1, 11}};
  std::fill(bytes.begin(), bytes.end(), uint8_t{0xa5});
  std::fill(models.begin(), models.end(), uint8_t{0x5a});
  auto e = expectations(2, models.data());
  e.layers = layers;
  const auto bytes_before = bytes;
  for (const size_t capacity : {SIZE_MAX, size_t(0)})
    check(spin8_query_adapter_load(
              bytes.data(), capacity, &e, sizeof(e),
              reinterpret_cast<spin8_query_adapter **>(bytes.data())) ==
                  SPIN8_INVALID_ARGUMENT &&
              bytes == bytes_before,
          "overflow/zero file range with result alias preserves input");
  const auto e_before = e;
  for (const size_t capacity : {SIZE_MAX, size_t(0)})
    check(spin8_query_adapter_load(bytes.data(), bytes.size(), &e, capacity,
                                   reinterpret_cast<spin8_query_adapter **>(
                                       &e)) == SPIN8_INVALID_ARGUMENT &&
              std::memcmp(&e, &e_before, sizeof(e)) == 0,
          "overflow/zero expectation range with result alias preserves "
          "descriptor");
  const auto models_before = models;
  const auto layers_before =
      std::array<spin8_query_adapter_layer_spec, 2>{layers[0], layers[1]};
  for (const size_t capacity : {SIZE_MAX, size_t(0)}) {
    auto nested = e;
    nested.model_count = 0;
    nested.model_digests_bytes = capacity;
    check(spin8_query_adapter_load(
              bytes.data(), bytes.size(), &nested, sizeof(nested),
              reinterpret_cast<spin8_query_adapter **>(models.data())) ==
                  SPIN8_INVALID_ARGUMENT &&
              models == models_before,
          "invalid nested model range cannot bypass result alias rejection");
    nested = e;
    nested.layer_count = 0;
    nested.layers_bytes = capacity;
    check(spin8_query_adapter_load(
              bytes.data(), bytes.size(), &nested, sizeof(nested),
              reinterpret_cast<spin8_query_adapter **>(layers)) ==
                  SPIN8_INVALID_ARGUMENT &&
              std::memcmp(layers, layers_before.data(), sizeof(layers)) == 0,
          "invalid nested layer range cannot bypass result alias rejection");
  }
  auto *out = reinterpret_cast<spin8_query_adapter *>(uintptr_t(1));
  check(spin8_query_adapter_load(
            bytes.data(), bytes.size(),
            reinterpret_cast<const spin8_query_adapter_expectations *>(
                UINTPTR_MAX - 7),
            sizeof(e), &out) == SPIN8_INVALID_ARGUMENT &&
            out == reinterpret_cast<spin8_query_adapter *>(uintptr_t(1)),
        "unrepresentable descriptor rejected without dereference or output "
        "write");
}
void failed(const std::vector<uint8_t> &data,
            spin8_query_adapter_expectations e,
            size_t descriptor_bytes = sizeof(spin8_query_adapter_expectations),
            bool untouched = false) {
  auto *out = reinterpret_cast<spin8_query_adapter *>(uintptr_t(1));
  check(spin8_query_adapter_load(data.data(), data.size(), &e, descriptor_bytes,
                                 &out) != SPIN8_OK,
        "bad expected contract rejected");
  check(out == (untouched
                    ? reinterpret_cast<spin8_query_adapter *>(uintptr_t(1))
                    : nullptr),
        "failure result follows disjointness guarantee");
}
void boundaries(const std::vector<uint8_t> &data,
                const spin8_query_adapter_expectations &expected) {
  auto e = expected;
  e.model_digests_bytes = 0;
  failed(data, e);
  e = expected;
  e.layers_bytes = 0;
  failed(data, e);
  e = expected;
  e.model_count = 257;
  failed(data, e);
  e = expected;
  e.layer_count = UINT32_MAX;
  failed(data, e);
  e = expected;
  e.abi_version = 99;
  failed(data, e);
  e = expected;
  e.struct_bytes = 0;
  failed(data, e);
  e = expected;
  e.reserved = 1;
  failed(data, e);
  e = expected;
  e.profile_digest[0] ^= 1;
  failed(data, e);
  e = expected;
  e.position_digest[0] ^= 1;
  failed(data, e);
  e = expected;
  std::memset(e.position_digest, 0, 32);
  failed(data, e);
  e = expected;
  e.layer_count = 1;
  failed(data, e);
  failed(data, expected, sizeof(e) - 1);
  std::array<uint8_t, 64> models{};
  std::memcpy(models.data(), expected.model_digests,
              expected.model_digests_bytes);
  models[0] ^= 1;
  e = expected;
  e.model_digests = models.data();
  failed(data, e);
  std::fill(models.begin(), models.end(), uint8_t{0});
  failed(data, e);
  spin8_query_adapter_layer_spec wrong_specs[] = {specs[1], specs[0]};
  e = expected;
  e.layers = wrong_specs;
  failed(data, e);
  wrong_specs[0] = specs[0];
  wrong_specs[1] = specs[1];
  wrong_specs[0].kv_heads = 3;
  failed(data, e);
  e = expected;
  e.layers = reinterpret_cast<const spin8_query_adapter_layer_spec *>(
      reinterpret_cast<const uint8_t *>(specs) + 1);
  failed(data, e);
  e = expected;
  e.layers =
      reinterpret_cast<const spin8_query_adapter_layer_spec *>(UINTPTR_MAX - 7);
  failed(data, e, sizeof(e), true);
  spin8_query_adapter *out =
      reinterpret_cast<spin8_query_adapter *>(uintptr_t(1));
  check(spin8_query_adapter_load(data.data(), SIZE_MAX, &expected,
                                 sizeof(expected),
                                 &out) == SPIN8_INVALID_ARGUMENT &&
            out == reinterpret_cast<spin8_query_adapter *>(uintptr_t(1)),
        "unrepresentable file range leaves result untouched");
  check(spin8_query_adapter_load(
            data.data(), SPIN8_QUERY_ADAPTER_MAX_BYTES + size_t(1), &expected,
            sizeof(expected), &out) == SPIN8_INVALID_ARGUMENT &&
            !out,
        "total file limit before read");
  alignas(spin8_query_adapter_expectations) std::array<uint8_t, sizeof(e) + 1>
      misaligned{};
  std::memcpy(misaligned.data() + 1, &expected, sizeof(expected));
  check(spin8_query_adapter_load(
            data.data(), data.size(),
            reinterpret_cast<const spin8_query_adapter_expectations *>(
                misaligned.data() + 1),
            sizeof(expected), &out) == SPIN8_INVALID_ARGUMENT &&
            !out,
        "descriptor alignment");
  auto alias = data;
  const auto before = alias;
  check(spin8_query_adapter_load(
            alias.data(), alias.size(), &expected, sizeof(expected),
            reinterpret_cast<spin8_query_adapter **>(alias.data())) ==
            SPIN8_INVALID_ARGUMENT,
        "aliased result rejected");
  check(alias == before, "aliased failure leaves input unchanged");
  e = expected;
  const auto e_before = e;
  check(spin8_query_adapter_load(data.data(), data.size(), &e, sizeof(e),
                                 reinterpret_cast<spin8_query_adapter **>(
                                     &e)) == SPIN8_INVALID_ARGUMENT &&
            std::memcmp(&e, &e_before, sizeof(e)) == 0,
        "result alias cannot overwrite expectations");
  auto unaligned = data;
  unaligned.insert(unaligned.begin(), 0);
  check(spin8_query_adapter_load(unaligned.data() + 1, data.size(), &expected,
                                 sizeof(expected), &out) == SPIN8_OK,
        "unaligned wire bytes accepted");
  spin8_query_adapter_free(out);
}
} // namespace

int main(int argc, char **argv) {
  hashes();
  overflow_alias_regressions();
  spin8_query_v3_basis_report basis{};
  check(spin8_query_v3_basis_audit(&basis) == SPIN8_OK,
        "actual core basis audit");
  if (argc == 1) {
    spin8_query_adapter *out =
        reinterpret_cast<spin8_query_adapter *>(uintptr_t(1));
    check(spin8_query_adapter_load(nullptr, 0, nullptr, 0, &out) ==
                  SPIN8_INVALID_ARGUMENT &&
              !out,
          "null C boundary");
    return 0;
  }
  check(argc >= 4, "fixture arguments");
  std::ifstream file(argv[2], std::ios::binary);
  check(bool(file), "open fixture");
  std::vector<uint8_t> data((std::istreambuf_iterator<char>(file)),
                            std::istreambuf_iterator<char>());
  std::array<uint8_t, 64> models{};
  digest("model-A", models.data());
  digest("model-B", models.data() + 32);
  const auto expected =
      expectations(static_cast<uint32_t>(std::atoi(argv[3])), models.data());
  spin8_query_adapter *adapter =
      reinterpret_cast<spin8_query_adapter *>(uintptr_t(1));
  const auto status = spin8_query_adapter_load(
      data.data(), data.size(), &expected, sizeof(expected), &adapter);
  if (std::string(argv[1]) == "reject") {
    check(status != SPIN8_OK && !adapter,
          "malformed fixture rejected without result leakage");
    return 0;
  }
  check(status == SPIN8_OK && adapter, "Python golden accepted");
  spin8_query_adapter_info info{};
  check(spin8_query_adapter_get_info(adapter, &info, sizeof(info)) == SPIN8_OK,
        "metadata view");
  check(argc == 5 && hex(info.content_digest) == argv[4],
        "independent Python hashlib full identity");
  check(spin8_query_adapter_c_check(adapter) == 0,
        "actual C translation unit consumes adapter");
  const auto original = data;
  std::fill(data.begin(), data.end(), uint8_t{0});
  const bool identity = std::string(argv[1]) == "identity";
  for (uint32_t i = 0; i < 2; ++i) {
    spin8_query_adapter_layer_view view{};
    check(spin8_query_adapter_get_layer(adapter, i, &view, sizeof(view)) ==
              SPIN8_OK,
          "layer view");
    const float *arrays[] = {view.win, view.bin, view.wout, view.bout};
    const size_t sizes[] = {view.win_bytes, view.bin_bytes, view.wout_bytes,
                            view.bout_bytes};
    for (size_t j = 0; j < 4; ++j) {
      if (!identity) {
        pattern_check(arrays[j], sizes[j] / sizeof(float), j);
      } else {
        for (size_t k = 0; k < sizes[j] / sizeof(float); ++k)
          check(arrays[j][k] == (j == 0 ? (k % 2 ? 0.125f : -0.125f) : 0.f),
                "nonzero Win and zero output initialization");
      }
    }
    behavior(view);
    check(spin8_query_adapter_get_layer(
              adapter, i,
              reinterpret_cast<spin8_query_adapter_layer_view *>(
                  const_cast<float *>(view.win)),
              sizeof(view)) == SPIN8_INVALID_ARGUMENT,
          "borrowed payload cannot be getter output");
  }
  spin8_query_adapter_free(adapter);
  boundaries(original, expected);
  return 0;
}
