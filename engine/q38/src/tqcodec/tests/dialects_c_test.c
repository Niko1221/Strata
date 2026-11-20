#include "tqcodec/dialects.h"
#include <string.h>
int main(void) {
  float x[128] = {0};
  unsigned char bytes[66] = {0};
  tqc_dialect_v1_info info = {0};
  tqc_dialect_v1_request req = {0};
  tqc_dialect_v1_receipt result = {0};
  info.version = 1;
  info.size = sizeof(info);
  if (tqc_dialect_v1_query(TQC_THETOM_64395689_TURBO4, &info))
    return 1;
  if (info.block_bytes != 66 || info.block_elements != 128 || info.cuda)
    return 2;
  req.version = 1;
  req.size = sizeof(req);
  req.dialect = TQC_THETOM_64395689_TURBO4;
  req.operation = TQC_D_ENCODE;
  req.device = TQC_D_CPU;
  req.precision = TQC_D_F32;
  req.rows = 1;
  req.columns = 128;
  req.src.data = x;
  req.src.capacity_bytes = sizeof(x);
  req.src.row_stride_bytes = sizeof(x);
  req.dst.data = bytes;
  req.dst.capacity_bytes = sizeof(bytes);
  req.dst.row_stride_bytes = sizeof(bytes);
  result.version = 1;
  result.size = sizeof(result);
  if (tqc_dialect_v1_execute(&req, &result))
    return 3;
  if (result.payload_bytes != 66 || bytes[0] != 0 || bytes[1] != 0)
    return 4;
  req.dst.capacity_bytes = 65;
  if (tqc_dialect_v1_execute(&req, &result) != TQC_D_CAPACITY)
    return 5;
  return 0;
}
