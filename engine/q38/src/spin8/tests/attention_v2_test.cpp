#include "spin8/attention_v2.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

static size_t cases = 0;
static double worst_error = 0;

static void check(bool ok, const char *message) {
  if (!ok) {
    std::fprintf(stderr, "FAIL: %s\n", message);
    std::exit(1);
  }
}

static void identity_attention() {
  uint32_t map[] = {0};
  spin8_attention_v2_config config{};
  config.abi_version = SPIN8_ATTENTION_V2_ABI_VERSION;
  config.struct_bytes = sizeof(config);
  config.semantics = SPIN8_ATTENTION_V2_SELF;
  config.dtype = SPIN8_F64;
  config.source_id = 42;
  config.n_q_heads = config.n_kv_heads = 1;
  config.d_qk = 2;
  config.d_v = 3;
  config.q_to_kv = map;
  config.q_to_kv_bytes = sizeof(map);
  spin8_attention_v2_plan *plan = nullptr;
  check(spin8_attention_v2_plan_create(&config, sizeof(config), &plan) ==
            SPIN8_OK,
        "create identity plan");
  double q[] = {1, 0}, k[] = {1, 0, 0, 1}, v[] = {1, 2, 3, 5, 6, 7};
  double out[] = {-99, -99, -99};
  uint8_t mask[] = {1, 1};
  spin8_attention_v2_request request{};
  request.abi_version = SPIN8_ATTENTION_V2_ABI_VERSION;
  request.struct_bytes = sizeof(request);
  request.q_source_id = request.kv_source_id = 42;
  request.position_stage = SPIN8_ATTENTION_V2_POST_POSITION;
  request.n_query = 1;
  request.n_key = 2;
  request.scale = 0.7;
  request.logit_order = SPIN8_ATTENTION_V2_SCALE_BIAS_SOFTCAP;
  request.q = {q, sizeof(q), 16, 16, 8};
  request.k = {k, sizeof(k), 16, 16, 8};
  request.v = {v, sizeof(v), 24, 24, 8};
  request.mask = {mask, sizeof(mask), 2, 2, 1};
  request.output = {out, sizeof(out), 24, 24, 8};
  check(spin8_attention_v2_execute(plan, &request, sizeof(request)) == SPIN8_OK,
        "execute identity attention");
  const double p = std::exp(0.7) / (std::exp(0.7) + 1);
  for (size_t i = 0; i < 3; ++i)
    check(std::abs(out[i] - (p * v[i] + (1 - p) * v[3 + i])) <= 1e-11,
          "identity scalar oracle");
  spin8_attention_v2_plan_free(plan);
  ++cases;
}

// Deliberately unaligned data, independent positive byte strides and padding.
struct Buffer {
  size_t nt, nh, nc, item, ts, hs, cs, capacity;
  std::vector<unsigned char> bytes;
  Buffer(size_t t, size_t h, size_t c, size_t size = 8,
         bool interleaved = false)
      : nt(t), nh(h), nc(c), item(size) {
    cs = size + 3;
    hs = c * cs + 7;
    if (interleaved) {
      hs = size + 5;
      cs = h * hs + 7;
    }
    ts = (h - 1) * hs + (c - 1) * cs + size + 13;
    capacity = (t - 1) * ts + (h - 1) * hs + (c - 1) * cs + size;
    bytes.assign(capacity + 12, 0xa5);
  }
  size_t offset(size_t t, size_t h, size_t c) const {
    return 1 + t * ts + h * hs + c * cs;
  }
  void put(size_t t, size_t h, size_t c, double value) {
    std::memcpy(bytes.data() + offset(t, h, c), &value, 8);
  }
  double get(size_t t, size_t h, size_t c) const {
    double value;
    std::memcpy(&value, bytes.data() + offset(t, h, c), 8);
    return value;
  }
  void allow(size_t t, size_t h, size_t c, bool value) {
    bytes[offset(t, h, c)] = value ? 7 : 0;
  }
  bool allowed(size_t t, size_t h, size_t c) const {
    return bytes[offset(t, h, c)] != 0;
  }
  spin8_attention_v2_input input() const {
    return {bytes.data() + 1, capacity, ts, hs, cs};
  }
  spin8_attention_v2_output output() {
    return {bytes.data() + 1, capacity, ts, hs, cs};
  }
  void sentinels() const {
    std::vector<bool> touched(bytes.size());
    for (size_t t = 0; t < nt; ++t)
      for (size_t h = 0; h < nh; ++h)
        for (size_t c = 0; c < nc; ++c)
          for (size_t b = 0; b < item; ++b)
            touched[offset(t, h, c) + b] = true;
    for (size_t i = 0; i < bytes.size(); ++i)
      check(touched[i] || bytes[i] == 0xa5, "strided output guard/padding");
  }
};

struct Fixture {
  Buffer q, k, v, mask, bias, gate, out;
  std::vector<uint32_t> map;
  spin8_attention_v2_config config{};
  spin8_attention_v2_request request{};
  Fixture(size_t dq, size_t dv, size_t heads, size_t kvheads)
      : q(3, heads, dq), k(5, kvheads, dq, 8, true), v(5, kvheads, dv),
        mask(3, heads, 5, 1, true), bias(3, heads, 5),
        gate(3, heads, dv, 8, true), out(3, heads, dv, 8, true), map(heads) {
    for (size_t h = 0; h < heads; ++h)
      map[h] = static_cast<uint32_t>((h * 2 + kvheads - 1) % kvheads);
    config.abi_version = SPIN8_ATTENTION_V2_ABI_VERSION;
    config.struct_bytes = sizeof(config);
    config.semantics = SPIN8_ATTENTION_V2_SELF;
    config.dtype = SPIN8_F64;
    config.source_id = 471;
    config.n_q_heads = heads;
    config.n_kv_heads = kvheads;
    config.d_qk = dq;
    config.d_v = dv;
    config.q_to_kv = map.data();
    config.q_to_kv_bytes = map.size() * sizeof(uint32_t);
    request.abi_version = SPIN8_ATTENTION_V2_ABI_VERSION;
    request.struct_bytes = sizeof(request);
    request.position_stage = SPIN8_ATTENTION_V2_POST_POSITION;
    request.q_source_id = request.kv_source_id = config.source_id;
    request.n_query = q.nt;
    request.n_key = k.nt;
    request.scale = 0.37; // Deliberately not 1/sqrt(d).
    request.softcap = 1.3;
    request.q = q.input();
    request.k = k.input();
    request.v = v.input();
    request.mask = mask.input();
    request.bias = bias.input();
    request.gate = gate.input();
    request.output = out.output();
    for (Buffer *b : {&q, &k, &v, &gate})
      for (size_t t = 0; t < b->nt; ++t)
        for (size_t h = 0; h < b->nh; ++h)
          for (size_t c = 0; c < b->nc; ++c)
            b->put(t, h, c,
                   std::sin(0.13 *
                            static_cast<double>(1 + t * 71 + h * 31 + c * 7)) +
                       0.17 * static_cast<double>(h + t));
    for (size_t t = 0; t < q.nt; ++t)
      for (size_t h = 0; h < heads; ++h)
        for (size_t j = 0; j < k.nt; ++j) {
          mask.allow(t, h, j, j == 0 || (t + h + j) % 3 != 0);
          bias.put(t, h, j, 0.6 * static_cast<double>(j) - 1.1);
        }
  }
  spin8_attention_v2_plan *make() const {
    spin8_attention_v2_plan *p = nullptr;
    check(spin8_attention_v2_plan_create(&config, sizeof(config), &p) ==
                  SPIN8_OK &&
              p,
          "create geometry plan");
    return p;
  }
  // Independent scalar FP64 original-space oracle: no production transforms.
  std::vector<double> oracle() const {
    std::vector<double> result(q.nt * q.nh * v.nc);
    for (size_t t = 0; t < q.nt; ++t)
      for (size_t h = 0; h < q.nh; ++h) {
        std::vector<double> scores(k.nt);
        double maximum = -std::numeric_limits<double>::infinity();
        for (size_t j = 0; j < k.nt; ++j) {
          double dot = 0;
          for (size_t c = 0; c < q.nc; ++c)
            dot += q.get(t, h, c) * k.get(j, map[h], c);
          double s = request.scale * dot;
          const double b = request.bias.data ? bias.get(t, h, j) : 0;
          if (request.logit_order == SPIN8_ATTENTION_V2_SCALE_BIAS_SOFTCAP)
            s += b;
          if (request.softcap)
            s = request.softcap * std::tanh(s / request.softcap);
          if (request.logit_order == SPIN8_ATTENTION_V2_SCALE_SOFTCAP_BIAS)
            s += b;
          scores[j] = s;
          if (mask.allowed(t, h, j))
            maximum = std::max(maximum, s);
        }
        double sum = 0;
        for (size_t j = 0; j < k.nt; ++j) {
          scores[j] = mask.allowed(t, h, j) ? std::exp(scores[j] - maximum) : 0;
          sum += scores[j];
        }
        for (size_t c = 0; c < v.nc; ++c) {
          double value = 0;
          for (size_t j = 0; j < k.nt; ++j)
            value += (scores[j] / sum) * v.get(j, map[h], c);
          if (request.gate.data)
            value *= gate.get(t, h, c);
          result[(t * q.nh + h) * v.nc + c] = value;
        }
      }
    return result;
  }
  void compare(spin8_attention_v2_plan *p) {
    const auto expected = oracle();
    const auto saved_q = q.bytes, saved_k = k.bytes, saved_v = v.bytes;
    const auto saved_mask = mask.bytes, saved_bias = bias.bytes,
               saved_gate = gate.bytes;
    check(spin8_attention_v2_execute(p, &request, sizeof(request)) == SPIN8_OK,
          "strided attention execution");
    for (size_t t = 0; t < q.nt; ++t)
      for (size_t h = 0; h < q.nh; ++h)
        for (size_t c = 0; c < v.nc; ++c) {
          const double error =
              std::abs(out.get(t, h, c) - expected[(t * q.nh + h) * v.nc + c]);
          check(std::isfinite(error) && error <= 1e-11,
                "FP64 scalar oracle <=1e-11");
          worst_error = std::max(worst_error, error);
        }
    check(q.bytes == saved_q && k.bytes == saved_k && v.bytes == saved_v &&
              mask.bytes == saved_mask && bias.bytes == saved_bias &&
              gate.bytes == saved_gate,
          "all input bytes unchanged");
    out.sentinels();
    ++cases;
  }
};

static void geometry_oracle() {
  const size_t widths[] = {64, 80, 96, 128, 130, 192, 256};
  const size_t value_widths[] = {130, 96, 80, 256, 192, 64, 128};
  for (size_t i = 0; i < 7; ++i)
    for (size_t topology = 0; topology < 3; ++topology)
      for (uint32_t order = 0; order < 2; ++order)
        for (int cap = 0; cap < 2; ++cap) {
          Fixture f(widths[i], value_widths[i], topology == 0 ? 3 : 5,
                    topology == 2 ? 1 : 3);
          f.request.logit_order = order;
          f.request.softcap = cap ? 1.3 : 0;
          auto *p = f.make();
          f.compare(p);
          spin8_attention_v2_plan_free(p);
        }
}

static void validation() {
  Fixture f(8, 9, 5, 3);
  auto *p = f.make();
  auto bad_request = [&](spin8_attention_v2_request r, spin8_status status,
                         const char *name,
                         size_t bytes = sizeof(spin8_attention_v2_request)) {
    const auto saved = f.out.bytes;
    check(spin8_attention_v2_execute(p, &r, bytes) == status, name);
    check(f.out.bytes == saved, "failure leaves entire output unchanged");
    ++cases;
  };
  auto bad_config = [&](spin8_attention_v2_config c, spin8_status status,
                        const char *name,
                        size_t bytes = sizeof(spin8_attention_v2_config)) {
    auto *out = p;
    check(spin8_attention_v2_plan_create(&c, bytes, &out) == status, name);
    check(out == p, "failed create leaves plan output unchanged");
    ++cases;
  };
  auto c = f.config;
  c.abi_version = 99;
  bad_config(c, SPIN8_UNSUPPORTED, "config ABI");
  c = f.config;
  c.struct_bytes--;
  bad_config(c, SPIN8_INVALID_ARGUMENT, "config size");
  bad_config(f.config, SPIN8_INVALID_ARGUMENT, "config short span",
             sizeof(c) - 1);
  for (size_t i = 0; i < 4; ++i) {
    c = f.config;
    c.reserved[i] = 1;
    bad_config(c, SPIN8_INVALID_ARGUMENT, "config reserved");
  }
  for (uint32_t dtype :
       {uint32_t(SPIN8_F32), uint32_t(SPIN8_F16), uint32_t(SPIN8_BF16), 99u}) {
    c = f.config;
    c.dtype = dtype;
    bad_config(c, SPIN8_UNSUPPORTED, "dtype rejected");
  }
  c = f.config;
  c.codec = 1;
  bad_config(c, SPIN8_UNSUPPORTED, "codec rejected");
  c = f.config;
  c.semantics = 2;
  bad_config(c, SPIN8_UNSUPPORTED, "cross-attention rejected");
  c = f.config;
  c.basis = 99;
  bad_config(c, SPIN8_UNSUPPORTED, "unknown basis");
  c = f.config;
  c.source_id = 0;
  bad_config(c, SPIN8_INVALID_ARGUMENT, "zero source");
  c = f.config;
  c.n_q_heads = 0;
  bad_config(c, SPIN8_INVALID_ARGUMENT, "zero q heads");
  c = f.config;
  c.n_kv_heads = 0;
  bad_config(c, SPIN8_INVALID_ARGUMENT, "zero kv heads");
  c = f.config;
  c.d_qk = 0;
  bad_config(c, SPIN8_INVALID_ARGUMENT, "zero qk width");
  c = f.config;
  c.d_v = 0;
  bad_config(c, SPIN8_INVALID_ARGUMENT, "zero v width");
  c = f.config;
  c.d_qk = SIZE_MAX;
  bad_config(c, SPIN8_INVALID_ARGUMENT, "dimension overflow");
  c = f.config;
  c.q_to_kv_bytes--;
  bad_config(c, SPIN8_INVALID_ARGUMENT, "mapping capacity");
  c = f.config;
  c.q_to_kv = nullptr;
  bad_config(c, SPIN8_INVALID_ARGUMENT, "null mapping");
  f.map[0] = 3;
  bad_config(f.config, SPIN8_INVALID_ARGUMENT, "mapping out of range");
  f.map[0] = 2;
  c = f.config;
  c.view = 1;
  bad_config(c, SPIN8_INVALID_ARGUMENT, "identity ignores no view");
  c = f.config;
  c.hadamard_block = 128;
  bad_config(c, SPIN8_INVALID_ARGUMENT, "identity ignores no H");
  auto r = f.request;
  r.abi_version++;
  bad_request(r, SPIN8_UNSUPPORTED, "request ABI");
  r = f.request;
  r.struct_bytes--;
  bad_request(r, SPIN8_INVALID_ARGUMENT, "request size");
  bad_request(f.request, SPIN8_INVALID_ARGUMENT, "request short span",
              sizeof(r) - 1);
  for (size_t i = 0; i < 4; ++i) {
    r = f.request;
    r.reserved[i] = 1;
    bad_request(r, SPIN8_INVALID_ARGUMENT, "request reserved");
  }
  r = f.request;
  r.q_source_id++;
  bad_request(r, SPIN8_UNSUPPORTED, "query source mismatch");
  r = f.request;
  r.kv_source_id++;
  bad_request(r, SPIN8_UNSUPPORTED, "KV source mismatch");
  r = f.request;
  r.position_stage = 0;
  bad_request(r, SPIN8_UNSUPPORTED, "pre-position QK rejected");
  r = f.request;
  r.logit_order = 77;
  bad_request(r, SPIN8_UNSUPPORTED, "unknown logit order");
  r = f.request;
  r.softcap = -1;
  bad_request(r, SPIN8_INVALID_ARGUMENT, "negative softcap");
  r = f.request;
  r.n_query = 0;
  bad_request(r, SPIN8_INVALID_ARGUMENT, "empty queries");
  r = f.request;
  r.n_key = 0;
  bad_request(r, SPIN8_INVALID_ARGUMENT, "empty keys");
  r = f.request;
  r.n_key = SIZE_MAX;
  bad_request(r, SPIN8_INVALID_ARGUMENT, "token overflow");
  for (auto field :
       {&spin8_attention_v2_request::q, &spin8_attention_v2_request::k,
        &spin8_attention_v2_request::v, &spin8_attention_v2_request::mask,
        &spin8_attention_v2_request::bias, &spin8_attention_v2_request::gate}) {
    r = f.request;
    (r.*field).capacity_bytes--;
    bad_request(r, SPIN8_INVALID_ARGUMENT, "short input capacity");
    r = f.request;
    (r.*field).token_stride_bytes = 0;
    bad_request(r, SPIN8_INVALID_ARGUMENT, "zero token stride");
    r = f.request;
    (r.*field).head_stride_bytes = 0;
    bad_request(r, SPIN8_INVALID_ARGUMENT, "zero head stride");
    r = f.request;
    (r.*field).channel_stride_bytes = 0;
    bad_request(r, SPIN8_INVALID_ARGUMENT, "zero channel stride");
    r = f.request;
    (r.*field).token_stride_bytes = SIZE_MAX;
    bad_request(r, SPIN8_INVALID_ARGUMENT, "stride overflow");
    r = f.request;
    (r.*field).capacity_bytes = SIZE_MAX;
    bad_request(r, SPIN8_INVALID_ARGUMENT, "address overflow");
    r = f.request;
    (r.*field).data = nullptr;
    bad_request(r, SPIN8_INVALID_ARGUMENT, "null input with nonzero layout");
    r = f.request;
    (r.*field).data = f.out.bytes.data() + 1;
    (r.*field).capacity_bytes = f.out.capacity;
    bad_request(r, SPIN8_INVALID_ARGUMENT, "output/input alias");
  }
  r = f.request;
  r.output.capacity_bytes--;
  bad_request(r, SPIN8_INVALID_ARGUMENT, "short output");
  r = f.request;
  r.output.channel_stride_bytes = 1;
  bad_request(r, SPIN8_INVALID_ARGUMENT, "overlapping output elements");
  r = f.request;
  r.output.token_stride_bytes = SIZE_MAX;
  bad_request(r, SPIN8_INVALID_ARGUMENT, "output overflow");
  r = f.request;
  r.output.data = nullptr;
  bad_request(r, SPIN8_INVALID_ARGUMENT, "null output");
  r = f.request;
  r.mask = {};
  bad_request(r, SPIN8_INVALID_ARGUMENT, "required mask");
  const double special[] = {std::numeric_limits<double>::quiet_NaN(),
                            std::numeric_limits<double>::infinity(),
                            -std::numeric_limits<double>::infinity()};
  for (double x : special) {
    r = f.request;
    r.scale = x;
    bad_request(r, SPIN8_NONFINITE, "nonfinite scale");
    r = f.request;
    r.softcap = x;
    bad_request(r, SPIN8_NONFINITE, "nonfinite softcap");
    for (Buffer *b : {&f.q, &f.k, &f.v, &f.bias, &f.gate}) {
      const double old = b->get(b->nt - 1, b->nh - 1, b->nc - 1);
      b->put(b->nt - 1, b->nh - 1, b->nc - 1, x);
      bad_request(f.request, SPIN8_NONFINITE, "nonfinite last tensor element");
      b->put(b->nt - 1, b->nh - 1, b->nc - 1, old);
    }
  }
  for (size_t j = 0; j < f.k.nt; ++j)
    f.mask.allow(2, 4, j, false);
  bad_request(f.request, SPIN8_INVALID_ARGUMENT,
              "late all-masked row is transactional");
  for (size_t j = 0; j < f.k.nt; ++j)
    f.mask.allow(2, 4, j, true);
  f.q.put(2, 4, 7, std::numeric_limits<double>::max());
  f.k.put(4, f.map[4], 7, std::numeric_limits<double>::max());
  bad_request(f.request, SPIN8_NONFINITE, "finite dot arithmetic overflow");
  spin8_attention_v2_plan_free(p);
  spin8_attention_v2_plan_free(nullptr);
  Fixture optional(8, 9, 5, 3);
  optional.request.bias = {};
  optional.request.gate = {};
  optional.request.scale = -0.2;
  p = optional.make();
  optional.compare(p);
  optional.request.scale = 0;
  optional.compare(p);
  // Plan must not borrow the mutable caller mapping.
  const auto saved_map = optional.map;
  std::fill(optional.map.begin(), optional.map.end(), UINT32_MAX);
  check(spin8_attention_v2_execute(p, &optional.request,
                                   sizeof(optional.request)) == SPIN8_OK,
        "mapping copied into plan");
  optional.map = saved_map;
  const auto expected = optional.oracle();
  for (size_t t = 0; t < optional.q.nt; ++t)
    for (size_t h = 0; h < optional.q.nh; ++h)
      for (size_t ch = 0; ch < optional.v.nc; ++ch)
        check(
            std::abs(optional.out.get(t, h, ch) -
                     expected[(t * optional.q.nh + h) * optional.v.nc + ch]) <=
                1e-11,
            "mapping copy result");
  ++cases;
  spin8_attention_v2_plan_free(p);
}

static void fixed_basis() {
  const size_t widths[] = {64, 80, 96, 128, 130, 192, 256};
  const size_t value_widths[] = {130, 96, 80, 256, 192, 64, 128};
  for (size_t i = 0; i < 7; ++i)
    for (size_t topology = 0; topology < 3; ++topology)
      for (uint32_t view = 0; view < 3; ++view)
        for (uint32_t block : {0u, 128u}) {
          Fixture f(widths[i], value_widths[i], topology == 0 ? 3 : 5,
                    topology == 2 ? 1 : 3);
          spin8_plane planes[] = {{0, 2, 0.73}, {4, 7, -0.38}, {1, 6, 1.23}};
          f.config.basis = SPIN8_ATTENTION_V2_FIXED;
          f.config.view = view;
          f.config.hadamard_block = block;
          f.config.planes = planes;
          f.config.plane_count = 3;
          f.config.planes_bytes = sizeof(planes);
          auto *p = f.make();
          // Mutation and end-of-scope lifetime must not affect the owned plan.
          planes[0].angle = std::numeric_limits<double>::quiet_NaN();
          for (uint32_t order = 0; order < 2; ++order) {
            f.request.logit_order = order;
            f.compare(p);
          }
          spin8_attention_v2_plan_free(p);
        }
  Fixture f(130, 128, 5, 3);
  spin8_plane plane{0, 1, 0.7};
  f.config.basis = SPIN8_ATTENTION_V2_FIXED;
  f.config.planes = &plane;
  f.config.plane_count = 1;
  f.config.planes_bytes = sizeof(plane);
  auto bad = [&](spin8_attention_v2_config c, spin8_status expected,
                 const char *name) {
    spin8_attention_v2_plan *p = nullptr;
    check(spin8_attention_v2_plan_create(&c, sizeof(c), &p) == expected && !p,
          name);
    ++cases;
  };
  auto c = f.config;
  c.view = 3;
  bad(c, SPIN8_UNSUPPORTED, "multi-view/unknown view rejected");
  c = f.config;
  c.hadamard_block = 8;
  bad(c, SPIN8_UNSUPPORTED, "non-H128 block rejected");
  c = f.config;
  c.planes_bytes--;
  bad(c, SPIN8_INVALID_ARGUMENT, "plane capacity");
  c = f.config;
  c.planes = nullptr;
  bad(c, SPIN8_INVALID_ARGUMENT, "null planes");
  c = f.config;
  c.plane_count = SIZE_MAX;
  bad(c, SPIN8_INVALID_ARGUMENT, "plane count overflow");
  plane.angle = std::numeric_limits<double>::infinity();
  bad(f.config, SPIN8_NONFINITE, "nonfinite plane");
  plane.angle = 0.7;
  plane.a = 8;
  bad(f.config, SPIN8_INVALID_ARGUMENT, "invalid plane");
  f.config.planes = nullptr;
  f.config.plane_count = f.config.planes_bytes = 0;
  f.config.hadamard_block = 128;
  auto *p = f.make();
  f.compare(p); // H128 with identity Spin8 is still a real transform path.
  spin8_attention_v2_plan_free(p);
}

static double difference(const std::vector<double> &a,
                         const std::vector<double> &b) {
  check(a.size() == b.size(), "oracle shape");
  double error = 0;
  for (size_t i = 0; i < a.size(); ++i)
    error = std::max(error, std::abs(a[i] - b[i]));
  return error;
}

// Dense test-only transform, independently applying coefficients and H parity.
static std::vector<double> dense_basis(std::vector<double> x,
                                       const std::array<double, 64> &m,
                                       bool inverse, bool h128) {
  auto hadamard = [&] {
    const auto input = x;
    for (size_t base = 0; base + 128 <= x.size(); base += 128)
      for (size_t i = 0; i < 128; ++i) {
        double sum = 0;
        for (size_t j = 0; j < 128; ++j) {
          size_t bits = i & j;
          int parity = 0;
          while (bits) {
            parity ^= 1;
            bits &= bits - 1;
          }
          sum += input[base + j] * (parity ? -1. : 1.);
        }
        x[base + i] = sum / std::sqrt(128.);
      }
  };
  if (h128 && !inverse)
    hadamard();
  const auto input = x;
  for (size_t base = 0; base + 8 <= x.size(); base += 8)
    for (size_t i = 0; i < 8; ++i) {
      x[base + i] = 0;
      for (size_t j = 0; j < 8; ++j)
        x[base + i] += m[inverse ? j * 8 + i : i * 8 + j] * input[base + j];
    }
  if (h128 && inverse)
    hadamard();
  return x;
}

static void negative_controls() {
  Fixture f(130, 130, 5, 3);
  spin8_plane planes[] = {{0, 2, 0.73}, {4, 7, -0.38}, {1, 6, 1.23}};
  spin8_plan *spin = nullptr;
  check(spin8_plan_create(planes, 3, &spin) == SPIN8_OK,
        "control audited transform");
  std::array<double, 64> vector{}, plus{};
  check(spin8_plan_matrix(spin, SPIN8_VECTOR, vector.data()) == SPIN8_OK &&
            spin8_plan_matrix(spin, SPIN8_SPIN_PLUS, plus.data()) == SPIN8_OK,
        "control matrices");
  spin8_plan_free(spin);
  const auto expected = f.oracle();
  const auto qbytes = f.q.bytes, kbytes = f.k.bytes;
  for (Buffer *b : {&f.q, &f.k})
    for (size_t t = 0; t < b->nt; ++t)
      for (size_t h = 0; h < b->nh; ++h) {
        std::vector<double> row(b->nc);
        for (size_t ch = 0; ch < b->nc; ++ch)
          row[ch] = b->get(t, h, ch);
        row = dense_basis(row, b == &f.q ? vector : plus, false, true);
        for (size_t ch = 0; ch < b->nc; ++ch)
          b->put(t, h, ch, row[ch]);
      }
  const double basis_error = difference(expected, f.oracle());
  check(basis_error > 1e-4, "wrong Q/K bases negative control must diverge");
  f.q.bytes = qbytes;
  f.k.bytes = kbytes;
  f.request.gate = {};
  const auto ungated = f.oracle();
  f.request.gate = f.gate.input();
  std::vector<double> wrong_gate(ungated.size());
  for (size_t t = 0; t < f.q.nt; ++t)
    for (size_t h = 0; h < f.q.nh; ++h) {
      const size_t start = (t * f.q.nh + h) * f.v.nc;
      std::vector<double> row(ungated.begin() + start,
                              ungated.begin() + start + f.v.nc);
      row = dense_basis(row, plus, false, true);
      for (size_t ch = 0; ch < f.v.nc; ++ch)
        row[ch] *= f.gate.get(t, h, ch);
      row = dense_basis(row, plus, true, true);
      std::copy(row.begin(), row.end(), wrong_gate.begin() + start);
    }
  const double gate_error = difference(expected, wrong_gate);
  check(gate_error > 1e-4, "gate-before-inverse negative control must diverge");
  // Compare the actual path on the SAME sensitive fixture.
  f.config.basis = SPIN8_ATTENTION_V2_FIXED;
  f.config.view = SPIN8_SPIN_PLUS;
  f.config.hadamard_block = 128;
  f.config.planes = planes;
  f.config.plane_count = 3;
  f.config.planes_bytes = sizeof(planes);
  auto *p = f.make();
  f.compare(p);
  spin8_attention_v2_plan_free(p);
  ++cases;
  std::printf(
      "Negative controls: wrong Q/K basis %.9g; gate-before-inverse %.9g\n",
      basis_error, gate_error);
}

static void transform_witnesses() {
  // Orthogonal output invariance alone cannot detect an identity shortcut.
  // Finite inputs whose REAL transform overflows provide an observable witness.
  for (int which = 0; which < 4; ++which) {
    Fixture f(128, 128, 1, 1);
    f.request.bias = {};
    f.request.gate = {};
    f.request.softcap = 0;
    for (Buffer *b : {&f.q, &f.k, &f.v})
      for (size_t t = 0; t < b->nt; ++t)
        for (size_t ch = 0; ch < b->nc; ++ch)
          b->put(t, 0, ch, 0);
    Buffer *huge = which == 0 ? &f.q : which == 1 ? &f.k : &f.v;
    for (size_t ch = 0; ch < (which == 3 ? 128u : 2u); ++ch)
      huge->put(0, 0, ch, std::numeric_limits<double>::max());
    for (size_t t = 0; t < f.q.nt; ++t)
      for (size_t j = 0; j < f.k.nt; ++j)
        f.mask.allow(t, 0, j, j == 0);
    auto *p = f.make();
    f.compare(p);
    spin8_attention_v2_plan_free(p);
    spin8_plane plane{0, 1, 0.7853981633974483};
    f.config.basis = SPIN8_ATTENTION_V2_FIXED;
    if (which == 3)
      f.config.hadamard_block = 128;
    else {
      f.config.planes = &plane;
      f.config.plane_count = 1;
      f.config.planes_bytes = sizeof(plane);
    }
    p = f.make();
    const auto before = f.out.bytes;
    check(spin8_attention_v2_execute(p, &f.request, sizeof(f.request)) ==
              SPIN8_NONFINITE,
          "actual Q/K/V or H128 transform overflow witness");
    check(before == f.out.bytes, "transform failure is transactional");
    spin8_attention_v2_plan_free(p);
    ++cases;
  }
}

static void boundary_supplement() {
  Fixture f(8, 9, 5, 3);
  auto *p = f.make();
  for (auto field :
       {&spin8_attention_v2_request::q, &spin8_attention_v2_request::k,
        &spin8_attention_v2_request::v, &spin8_attention_v2_request::mask,
        &spin8_attention_v2_request::bias, &spin8_attention_v2_request::gate})
    for (size_t shift : {size_t(0), size_t(8)}) {
      auto r = f.request;
      const size_t capacity =
          std::max((r.*field).capacity_bytes + shift, r.output.capacity_bytes);
      std::vector<unsigned char> shared(capacity + 16, 0x5a);
      (r.*field).data = shared.data() + shift;
      r.output.data = shared.data();
      const auto before = shared;
      check(spin8_attention_v2_execute(p, &r, sizeof(r)) ==
                SPIN8_INVALID_ARGUMENT,
            "exact/partial output alias with valid capacities");
      check(shared == before, "alias rejection preserves input/output storage");
      ++cases;
    }
  check(spin8_attention_v2_execute(nullptr, &f.request, sizeof(f.request)) ==
            SPIN8_INVALID_ARGUMENT,
        "null plan");
  check(spin8_attention_v2_execute(p, nullptr, sizeof(f.request)) ==
            SPIN8_INVALID_ARGUMENT,
        "null request");
  check(spin8_attention_v2_plan_create(nullptr, sizeof(f.config), &p) ==
            SPIN8_INVALID_ARGUMENT,
        "null config");
  check(spin8_attention_v2_plan_create(&f.config, sizeof(f.config), nullptr) ==
            SPIN8_INVALID_ARGUMENT,
        "null create output");
  cases += 4;
  auto c = f.config;
  const auto original_config = c;
  check(spin8_attention_v2_plan_create(
            &c, sizeof(c),
            reinterpret_cast<spin8_attention_v2_plan **>(&c.reserved[0])) ==
            SPIN8_INVALID_ARGUMENT,
        "plan output aliases config");
  check(std::memcmp(&c, &original_config, sizeof(c)) == 0,
        "aliased config unchanged");
  alignas(spin8_attention_v2_plan *) std::array<uint32_t, 8> map{};
  c = f.config;
  c.q_to_kv = map.data();
  c.q_to_kv_bytes = sizeof(map);
  check(spin8_attention_v2_plan_create(
            &c, sizeof(c),
            reinterpret_cast<spin8_attention_v2_plan **>(map.data())) ==
            SPIN8_INVALID_ARGUMENT,
        "plan output aliases mapping");
  check(std::all_of(map.begin(), map.end(), [](uint32_t x) { return x == 0; }),
        "aliased map unchanged");
  alignas(spin8_attention_v2_plan *) spin8_plane plane{0, 1, 0.3};
  const auto original_plane = plane;
  c = f.config;
  c.basis = SPIN8_ATTENTION_V2_FIXED;
  c.planes = &plane;
  c.plane_count = 1;
  c.planes_bytes = sizeof(plane);
  check(spin8_attention_v2_plan_create(
            &c, sizeof(c),
            reinterpret_cast<spin8_attention_v2_plan **>(&plane)) ==
            SPIN8_INVALID_ARGUMENT,
        "plan output aliases planes");
  check(std::memcmp(&plane, &original_plane, sizeof(plane)) == 0,
        "aliased plane unchanged");
  cases += 3;
  const auto saved_output = f.out.bytes;
  f.mask.allow(2, 4, 4, false);
  f.bias.put(2, 4, 4, std::numeric_limits<double>::infinity());
  check(spin8_attention_v2_execute(p, &f.request, sizeof(f.request)) ==
                SPIN8_NONFINITE &&
            f.out.bytes == saved_output,
        "masked nonfinite bias rejected without writes");
  ++cases;
  spin8_attention_v2_plan_free(p);

  Fixture small(1, 1, 1, 1);
  p = small.make();
  auto r = small.request;
  r.output.data = &r;
  r.output.capacity_bytes = sizeof(r);
  const auto before = r;
  check(spin8_attention_v2_execute(p, &r, sizeof(r)) == SPIN8_INVALID_ARGUMENT,
        "output aliases request descriptor");
  check(std::memcmp(&r, &before, sizeof(r)) == 0, "aliased request unchanged");
  ++cases;
  r = small.request;
  r.n_query = 1;
  r.output.data = p;
  r.output.capacity_bytes = 8;
  check(spin8_attention_v2_execute(p, &r, sizeof(r)) == SPIN8_INVALID_ARGUMENT,
        "output aliases opaque plan");
  ++cases;
  // The plan remains usable after rejected aliases.
  small.compare(p);
  for (size_t t = 0; t < small.v.nt; ++t)
    small.v.put(t, 0, 0, 2);
  small.gate.put(2, 0, 0, std::numeric_limits<double>::max());
  const auto old_output = small.out.bytes;
  check(spin8_attention_v2_execute(p, &small.request, sizeof(small.request)) ==
                SPIN8_NONFINITE &&
            small.out.bytes == old_output,
        "late gate overflow leaves earlier rows unchanged");
  ++cases;
  spin8_attention_v2_plan_free(p);

  // Input-only aliases are valid; Q and K may share read-only bytes.
  Fixture shared(8, 9, 5, 5);
  shared.request.q = shared.k.input();
  for (size_t t = 0; t < shared.q.nt; ++t)
    for (size_t h = 0; h < shared.q.nh; ++h)
      for (size_t ch = 0; ch < shared.q.nc; ++ch)
        shared.q.put(t, h, ch, shared.k.get(t, h, ch));
  p = shared.make();
  shared.compare(p);
  spin8_attention_v2_plan_free(p);
}

int main() {
  identity_attention();
  geometry_oracle();
  validation();
  fixed_basis();
  negative_controls();
  transform_witnesses();
  boundary_supplement();
  std::printf("PASS attention_v2: %zu cases; worst absolute error %.17g\n",
              cases, worst_error);
}
