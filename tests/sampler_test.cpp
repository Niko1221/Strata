// CPU-only test of the host sampler.   g++ -std=c++17 -I../src/qwen36 sampler_test.cpp -o sampler_test && ./sampler_test
#include <cstdio>
#include <map>
#include "sampler.hpp"
using namespace q36;
static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++fails; } } while (0)

int main() {
    std::vector<float> lg(100, 0.f);
    lg[42] = 5.f; lg[7] = 4.f;
    { SamplingParams p; Sampler s(p); auto l = lg; CHECK(s.sample(l, {}) == 42); }                 // temperature absent = greedy
    { SamplingParams p; p.temperature = 0.f; p.repeat_penalty = 2.f; Sampler s(p);                // penalty turns 5 -> 2.5 < 4
      auto l = lg; CHECK(s.sample(l, {42}) == 7); }
    { SamplingParams p; p.presence_penalty = 3.f; Sampler s(p); auto l = lg; CHECK(s.sample(l, {42}) == 7); }   // 5-3=2 < 4
    { SamplingParams p; p.freq_penalty = 1.f; Sampler s(p); auto l = lg; CHECK(s.sample(l, {42, 42}) == 7); }  // 5-2=3 < 4
    { SamplingParams p; p.penalty_last_n = 1; p.repeat_penalty = 2.f; Sampler s(p);               // 42 is outside the window
      auto l = lg; CHECK(s.sample(l, {42, 1}) == 42); }
    { SamplingParams p; p.temperature = 1.f; p.top_k = 1; p.seed = 3; Sampler s(p);                // top_k 1 = greedy
      for (int i = 0; i < 50; ++i) { auto l = lg; CHECK(s.sample(l, {}) == 42); } }
    { SamplingParams p; p.temperature = 1.f; p.min_p = 0.5f; p.seed = 5; Sampler s(p);             // p(7)/p(42)=e^-1=0.37 < 0.5
      for (int i = 0; i < 200; ++i) { auto l = lg; CHECK(s.sample(l, {}) == 42); } }
    { SamplingParams p; p.temperature = 1.f; p.top_p = 0.5f; p.seed = 9; Sampler s(p);             // p(42) alone is > 0.5 of the top-64 mass
      std::vector<float> l2(100, -10.f); l2[1] = 10.f; l2[2] = 0.f;
      for (int i = 0; i < 100; ++i) { auto l = l2; CHECK(s.sample(l, {}) == 1); } }
    { SamplingParams p; p.temperature = 1.f; p.seed = 11; Sampler s(p); std::map<int, int> cnt;     // plain softmax over the top 64
      for (int i = 0; i < 4000; ++i) { auto l = lg; ++cnt[s.sample(l, {})]; }
      const double r = (double)cnt[7] / cnt[42];  // expected e^-1 = 0.368
      CHECK(r > 0.30 && r < 0.44); CHECK(cnt.size() > 2); }
    { SamplingParams p; p.temperature = 0.7f; p.seed = 42; Sampler a(p), b(p); auto l1 = lg, l2 = lg;  // same seed, same draws
      for (int i = 0; i < 20; ++i) { l1 = lg; l2 = lg; CHECK(a.sample(l1, {}) == b.sample(l2, {})); } }
    std::printf(fails ? "sampler_test: %d FAILED\n" : "sampler_test: all passed (%d failures)\n", fails);
    return fails != 0;
}
