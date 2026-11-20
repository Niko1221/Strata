#include "strata/qwen35/phase_trace.hpp"
#include <string>
#include <thread>

using namespace strata::qwen35;
int main() {
  FILE *output = std::tmpfile();
  if (!output)
    return 1;
  int failures = 0;
  auto check = [&](bool ok) {
    if (!ok)
      ++failures;
  };
  {
    Q36PhaseTrace phase(Q36Phase::prefill, 7, true, output);
    {
      Q36PhaseLayer layer(3);
      Q36ReadTrace read(37);
      read.complete(true);
      q36_phase_h2d(37);
    }
    {
      Q36PhaseLayer layer(4);
      Q36ReadTrace read(13);
      q36_phase_d2d(11);
    }
    check(phase.total().read_calls == 2 &&
          phase.total().read_completed_bytes == 37);
    check(phase.total().read_requested_bytes == 50 &&
          phase.total().read_failed_calls == 1);
    {
      Q36PhaseTrace disabled(Q36Phase::warmup, 0, false, output);
      Q36ReadTrace unobserved(99);
      unobserved.complete(true);
      q36_phase_h2d(100);
    }
    std::thread other([] {
      Q36ReadTrace unobserved(1000);
      unobserved.complete(true);
      q36_phase_h2d(1000);
    });
    other.join();
    check(phase.total().expert_h2d_submitted_bytes == 37);
    check(phase.total().read_calls == 2);
    phase.input_tokens = 12;
    phase.reused_tokens = 4;
    phase.processed_tokens = 8;
    phase.finish();
    phase.finish();
    check(q36_active_phase == nullptr);
  }
  {
    Q36PhaseTrace incomplete(Q36Phase::cancel, 8, true, output);
  }
  check(q36_active_phase == nullptr);
  std::rewind(output);
  std::string text;
  char line[2048];
  while (std::fgets(line, sizeof(line), output))
    text += line;
  std::fclose(output);
  check(text.find("request=7 phase=prefill state=complete layer=-1") !=
        std::string::npos);
  check(text.find("read_calls=2 read_requested_bytes=50 "
                  "read_completed_bytes=37 read_failed_calls=1") !=
        std::string::npos);
  check(text.find("input_tokens=12 reused_tokens=4 processed_tokens=8") !=
        std::string::npos);
  check(text.find("phase=cancel state=incomplete") != std::string::npos);
  check(text.find("phase=warmup") == std::string::npos);
  check(text.find("layer=3") != std::string::npos &&
        text.find("layer=4") != std::string::npos);
  if (failures)
    std::fprintf(stderr, "phase trace: %d failures\n", failures);
  return failures ? 1 : 0;
}
