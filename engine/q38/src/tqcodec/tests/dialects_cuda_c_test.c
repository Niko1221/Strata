#include "tqcodec/dialects_cuda.h"
#include <string.h>
int main(void) {
  tqc_dialect_cuda_v1_info info = {0};
  tqc_dialect_cuda_v1_request request = {0};
  unsigned id;
  info.version = 1;
  info.size = sizeof(info);
  request.version = 1;
  request.size = sizeof(request);
  for (id = 1; id <= 5; ++id) {
    if (tqc_dialect_cuda_v1_query(id, &info) != TQC_D_OK ||
        info.block_elements != 128 || info.block_bytes != 18 + id * 16 ||
        info.operation_mask != 126 || info.compute_capability != 120 ||
        info.precision != TQC_D_F32 || !info.columns_128 || !info.columns_256 ||
        strcmp(info.source_revision,
               "643956891dfdf51aecdbd05ca508dd21e21e8195") != 0)
      return 1;
  }
  if (tqc_dialect_cuda_v1_query(6, &info) != TQC_D_UNSUPPORTED)
    return 2;
  return tqc_dialect_cuda_v1_enqueue(&request) == TQC_D_UNSUPPORTED ? 0 : 3;
}
