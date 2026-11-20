#include "tqcodec/calibrate.h"
#include "tqcodec/cache.h"
#include "tqcodec/distill.h"
#include "tqcodec/transport.h"
#include <math.h>
#include <stdio.h>

int main(void) {
  const double book[4] = {-1., 0., 1., 2.};
  const double input[8] = {1., 0., 0., 0., 0., 0., 0., 0.};
  uint8_t packed[2] = {0, 0};
  double norm = 0., decoded[8] = {0}, score = 0.;
  tq_size size = {0, 0, 0};
  size_t scratch_bytes = 0;
  if (tq_block_size(8, 2, &size) != TQ_OK || size.payload_bytes != 2 ||
      tq_encode_f64(input, 8, book, 2, packed, sizeof(packed), &norm) != TQ_OK ||
      tq_decode_f64(packed, sizeof(packed), 8, book, 2, norm, decoded) != TQ_OK ||
      tq_score_f64(input, packed, sizeof(packed), 8, book, 2, norm, &score) != TQ_OK ||
      fabs(score-1.) > 1e-15 || decoded[0] != 1. ||
      tq_distill_workspace_size(8, 8, &scratch_bytes) != TQ_OK ||
      scratch_bytes != 24*sizeof(double)) {
    fputs("C public API smoke failed\n", stderr);
    return 1;
  }
  /* These calls cross the C linker boundary into the new opaque-plan ABI. */
  if (tq_transport_plan_audit(NULL, NULL) != TQ_INVALID ||
      tq_transport_plan_bytes(NULL, NULL) != TQ_INVALID) {
    fputs("C transport API smoke failed\n", stderr);
    return 1;
  }
  puts("C public headers and linked APIs passed");
  return 0;
}
