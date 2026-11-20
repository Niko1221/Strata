#include "spin8/attention_v2.h"
#include <math.h>
#include <stdio.h>

int main(void) {
  spin8_attention_v2_config c = {0};
  spin8_attention_v2_request r = {0};
  spin8_attention_v2_plan *p = NULL;
  uint32_t map[1] = {0};
  double q[8] = {0}, k[8] = {0}, v[3] = {2, -3, 4}, out[3] = {0};
  uint8_t mask[1] = {1};
  spin8_plane plane = {0, 1, 0.3};
  c.abi_version = SPIN8_ATTENTION_V2_ABI_VERSION;
  c.struct_bytes = sizeof(c);
  c.semantics = SPIN8_ATTENTION_V2_SELF;
  c.dtype = SPIN8_F64;
  c.basis = SPIN8_ATTENTION_V2_FIXED;
  c.view = SPIN8_SPIN_MINUS;
  c.source_id = 8;
  c.n_q_heads = c.n_kv_heads = 1;
  c.d_qk = 8;
  c.d_v = 3;
  c.q_to_kv = map;
  c.q_to_kv_bytes = sizeof(map);
  c.planes = &plane;
  c.plane_count = 1;
  c.planes_bytes = sizeof(plane);
  if (spin8_attention_v2_plan_create(&c, sizeof(c), &p) != SPIN8_OK)
    return 1;
  r.abi_version = SPIN8_ATTENTION_V2_ABI_VERSION;
  r.struct_bytes = sizeof(r);
  r.position_stage = SPIN8_ATTENTION_V2_POST_POSITION;
  r.q_source_id = r.kv_source_id = 8;
  r.n_query = r.n_key = 1;
  r.scale = 0.25;
  r.q.data = q;
  r.q.capacity_bytes = sizeof(q);
  r.q.token_stride_bytes = r.q.head_stride_bytes = sizeof(q);
  r.q.channel_stride_bytes = sizeof(double);
  r.k = r.q;
  r.k.data = k;
  r.v = r.q;
  r.v.data = v;
  r.v.capacity_bytes = sizeof(v);
  r.v.token_stride_bytes = r.v.head_stride_bytes = sizeof(v);
  r.mask.data = mask;
  r.mask.capacity_bytes = sizeof(mask);
  r.mask.token_stride_bytes = r.mask.head_stride_bytes =
      r.mask.channel_stride_bytes = 1;
  r.output.data = out;
  r.output.capacity_bytes = sizeof(out);
  r.output.token_stride_bytes = r.output.head_stride_bytes = sizeof(out);
  r.output.channel_stride_bytes = sizeof(double);
  if (spin8_attention_v2_execute(p, &r, sizeof(r)) != SPIN8_OK)
    return 2;
  if (fabs(out[0] - 2) > 1e-11 || fabs(out[1] + 3) > 1e-11 ||
      fabs(out[2] - 4) > 1e-11)
    return 3;
  r.reserved[3] = 1;
  if (spin8_attention_v2_execute(p, &r, sizeof(r)) != SPIN8_INVALID_ARGUMENT ||
      out[1] != -3)
    return 4;
  spin8_attention_v2_plan_free(p);
  puts("PASS attention_v2 pure C ABI: fixed plan, execute, reserved rejection");
  return 0;
}
