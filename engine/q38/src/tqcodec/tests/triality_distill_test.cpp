/* Synthetic one-head training integration. This does not load a model or
 * establish Transformer task quality. Train and held-out queries are separate.
 * A small explicit orthogonal position map exercises pre-position insertion;
 * it is not a claim of qwen35 MRoPE configuration compatibility. */
#include "spin8/query_v3.h"
#include "tqcodec/cache.h"
#include "tqcodec/distill.h"
#include <array>
#include <cmath>
#include <cstdio>
#include <memory>
#include <stdexcept>

namespace {
constexpr size_t d = 16, n = 9, samples = 8;
using Vec = std::array<double, d>;
struct Weights {
  std::array<double, 8 * d> win{};
  std::array<double, 8> bin{};
  std::array<double, 56> wout{};
  std::array<double, 7> bout{};
};
void need(bool condition, const char *message) {
  if (!condition)
    throw std::runtime_error(message);
}
void position(const Vec &input, Vec &out, size_t pos, bool inverse = false) {
  for (size_t i = 0; i < d; i += 2) {
    const double theta =
        (inverse ? -1 : 1) * double(pos) * .025 * double(i + 1);
    const double c = std::cos(theta), s = std::sin(theta);
    out[i] = c * input[i] - s * input[i + 1];
    out[i + 1] = s * input[i] + c * input[i + 1];
  }
}
template <size_t count>
void add(std::array<double, count> &to, const std::array<double, count> &from,
         double factor) {
  for (size_t i = 0; i < count; ++i)
    to[i] += factor * from[i];
}
void add(Weights &to, const Weights &from, double factor) {
  add(to.win, from.win, factor);
  add(to.bin, from.bin, factor);
  add(to.wout, from.wout, factor);
  add(to.bout, from.bout, factor);
}
struct Fixture {
  std::array<Vec, n> keys{}, values{}, kbar{}, vhat{};
  std::array<Vec, samples> queries{};
  std::array<std::array<double, d * d>, 3> transforms{};
  std::unique_ptr<tq_cache, decltype(&tq_cache_destroy)> cache{
      nullptr, tq_cache_destroy};
  Fixture() {
    spin8_plane planes[] = {{0, 1, .37}, {2, 5, -.29}, {1, 7, .53}};
    spin8_plan *raw = nullptr;
    need(spin8_plan_create(planes, 3, &raw) == SPIN8_OK, "strict plan");
    std::unique_ptr<spin8_plan, decltype(&spin8_plan_free)> plan(
        raw, spin8_plan_free);
    double h[64];
    need(spin8_hadamard_matrix(37, 0, h) == SPIN8_OK, "signed Hadamard");
    for (int r = 0; r < 3; ++r) {
      double b[64];
      need(spin8_plan_matrix(raw, static_cast<spin8_view>(r), b) == SPIN8_OK,
           "same Spin lift");
      for (size_t block = 0; block < d; block += 8)
        for (size_t i = 0; i < 8; ++i)
          for (size_t j = 0; j < 8; ++j)
            for (size_t k = 0; k < 8; ++k)
              transforms[r][(block + i) * d + block + j] +=
                  b[i * 8 + k] * h[k * 8 + j];
    }
    double kb[16], vb[4];
    for (size_t i = 0; i < 16; ++i)
      kb[i] = -1 + 2. * double(i) / 15;
    for (size_t i = 0; i < 4; ++i)
      vb[i] = -1 + 2. * double(i) / 3;
    tq_cache_config cfg{};
    cfg.width = d;
    cfg.capacity_rows = n;
    cfg.key_bits = 4;
    cfg.value_bits = 2;
    cfg.key_centroids = kb;
    cfg.value_centroids = vb;
    for (int r = 0; r < 3; ++r) {
      cfg.transforms[r] = transforms[r].data();
      cfg.alpha[r] = 1. / 3;
    }
    tq_cache *cache_raw = nullptr;
    need(tq_cache_create(&cfg, &cache_raw) == TQ_OK, "physical cache");
    cache.reset(cache_raw);
    for (size_t j = 0; j < n; ++j) {
      Vec pre{};
      for (size_t i = 0; i < d; ++i) {
        pre[i] = std::sin(.37 * double(1 + j * d + i)) +
                 .3 * std::cos(.19 * double(i + j));
        values[j][i] = .5 * std::cos(.41 * double(j * d + i)) + .2;
      }
      position(pre, keys[j], j);
      need(tq_cache_append(cache.get(), keys[j].data(), d, values[j].data(),
                           d) == TQ_OK,
           "append 3K/sharedV");
      for (unsigned r = 0; r < 4; ++r) {
        uint8_t payload[16];
        double norm;
        Vec decoded{};
        need(tq_cache_copy_row(cache.get(), j, r, payload, sizeof(payload),
                               &norm) == TQ_OK,
             "read actual cache bytes");
        need(tq_decode_f64(payload, sizeof(payload), d, r < 3 ? kb : vb,
                           r < 3 ? 4u : 2u, norm, decoded.data()) == TQ_OK,
             "decode actual cache bytes");
        if (r < 3) {
          for (size_t i = 0; i < d; ++i)
            for (size_t k = 0; k < d; ++k)
              kbar[j][i] += transforms[r][k * d + i] * decoded[k] / 3;
        } else
          vhat[j] = decoded;
      }
    }
    for (size_t s = 0; s < samples; ++s) {
      double norm = 0;
      for (size_t i = 0; i < d; ++i) {
        queries[s][i] = std::sin(.13 * double(3 + s * d + i)) +
                        .3 * std::cos(.29 * double(i + s));
        norm += queries[s][i] * queries[s][i];
      }
      for (double &x : queries[s])
        x *= std::sqrt(double(d) / norm);
    }
  }
};
std::unique_ptr<spin8_query_v3_controller,
                decltype(&spin8_query_v3_controller_free)>
controller(const Weights &w) {
  spin8_query_v3_controller *raw = nullptr;
  need(spin8_query_v3_controller_create(
           SPIN8_QUERY_V3_VERSION, d, w.win.data(), sizeof(w.win), w.bin.data(),
           sizeof(w.bin), w.wout.data(), sizeof(w.wout), w.bout.data(),
           sizeof(w.bout), &raw) == SPIN8_OK,
       "create controller snapshot");
  return {raw, spin8_query_v3_controller_free};
}
tq_f64_rows view(const std::array<Vec, n> &matrix) {
  return {matrix[0].data(), n, d, sizeof(Vec), sizeof(matrix)};
}
double evaluate(const Fixture &f, const Weights &w, size_t first, size_t count,
                Weights *gradient = nullptr, bool quantized = true) {
  auto ctl = controller(w);
  double sum = 0;
  if (gradient)
    *gradient = {};
  for (size_t s = first; s < first + count; ++s) {
    Vec adapted{}, qt{}, qs{}, dq_post{}, dy{}, dq{};
    need(spin8_query_v3_controller_apply(
             ctl.get(), SPIN8_F64, f.queries[s].data(), sizeof(Vec), d,
             adapted.data(), sizeof(Vec), d, 1) == SPIN8_OK,
         "pre-position Q adapter");
    position(f.queries[s], qt, n + s);
    position(adapted, qs, n + s);
    double scratch[3 * d];
    tq_distill_loss loss{};
    need(tq_distill_head_f64(
             qt.data(), sizeof(Vec), qs.data(), sizeof(Vec), view(f.keys),
             view(quantized ? f.kbar : f.keys), view(f.values),
             view(quantized ? f.vhat : f.values), nullptr, 0,
             1 / std::sqrt(double(d)), 1, .1, 1e-6, scratch, sizeof(scratch),
             &loss, dq_post.data(), sizeof(Vec)) == TQ_OK,
         "bounded attention distillation");
    sum += loss.total;
    if (gradient) {
      // The adapter is before the position operation; transpose its derivative.
      position(dq_post, dy, n + s, true);
      Weights g{};
      const spin8_query_v3_gradients_f64 outputs{
          dq.data(),     sizeof(dq),    g.win.data(),  sizeof(g.win),
          g.bin.data(),  sizeof(g.bin), g.wout.data(), sizeof(g.wout),
          g.bout.data(), sizeof(g.bout)};
      need(spin8_query_v3_controller_vjp_f64(ctl.get(), f.queries[s].data(),
                                             sizeof(Vec), dy.data(),
                                             sizeof(Vec), &outputs) == SPIN8_OK,
           "complete input-dependent VJP");
      add(*gradient, g, 1. / double(count));
    }
  }
  return sum / double(count);
}
} // namespace
int main() {
  try {
    const Fixture fixture;
    Weights weights{};
    for (size_t i = 0; i < weights.win.size(); ++i)
      weights.win[i] = .08 * std::sin(double(i + 1));
    Weights gradient{};
    const double original = evaluate(fixture, weights, 0, 4, &gradient);
    const double held_original = evaluate(fixture, weights, 4, 4);
    need(original > 0, "codec baseline differs from exactKV");
    need(evaluate(fixture, weights, 0, 4, nullptr, false) == 0,
         "identity init matches exactKV");
    double norm2 = 0;
    for (double g : gradient.wout)
      norm2 += g * g;
    need(norm2 > 1e-12, "first-step Wout receives attention gradient");
    for (double g : gradient.win)
      need(g == 0, "first-step Win gradient is zero as expected");
    // Independent five-point difference checks the whole pre-position adapter
    // -> RoPE-like map -> three-view reconstruction -> one-softmax loss chain.
    constexpr double step = 1e-4;
    for (size_t p : {size_t{0}, size_t{13}, size_t{55}}) {
      Weights probe = weights;
      probe.wout[p] = 2 * step;
      const double p2 = evaluate(fixture, probe, 0, 4);
      probe.wout[p] = step;
      const double p1 = evaluate(fixture, probe, 0, 4);
      probe.wout[p] = -step;
      const double m1 = evaluate(fixture, probe, 0, 4);
      probe.wout[p] = -2 * step;
      const double m2 = evaluate(fixture, probe, 0, 4);
      need(std::abs(gradient.wout[p] -
                    (-p2 + 8 * p1 - 8 * m1 + m2) / (12 * step)) < 2e-8,
           "full-chain parameter gradient matches numerical oracle");
    }
    double current = original;
    size_t accepted = 0;
    for (size_t epoch = 0; epoch < 20; ++epoch) {
      evaluate(fixture, weights, 0, 4, &gradient);
      bool improved = false;
      for (double rate = .1; rate >= 1e-6; rate *= .5) {
        Weights candidate = weights;
        add(candidate, gradient, -rate);
        const double next = evaluate(fixture, candidate, 0, 4);
        if (next < current) {
          weights = candidate;
          current = next;
          improved = true;
          ++accepted;
          break;
        }
      }
      if (!improved)
        break;
    }
    need(accepted > 0 && current < original - 1e-8,
         "synthetic training reduces training loss");
    const double held_final = evaluate(fixture, weights, 4, 4);
    std::printf("synthetic one-head distillation only: train %.17g -> %.17g; "
                "held-out %.17g -> %.17g; accepted %zu\n",
                original, current, held_original, held_final, accepted);
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "FAIL synthetic distillation: %s\n", error.what());
    return 1;
  }
}
