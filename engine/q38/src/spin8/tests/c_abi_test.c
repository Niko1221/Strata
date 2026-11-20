#include "spin8/spin8.h"
#include "spin8/query_v3.h"
int main(void) {
  spin8_capabilities caps;
  if (spin8_get_capabilities(0) != SPIN8_INVALID_ARGUMENT ||
      spin8_get_capabilities(&caps) != SPIN8_OK || caps.abi_version != 1 ||
      !(caps.cpu_dtype_mask & (1u << SPIN8_F64)) || caps.representation_dimension != 8)
    return 4;
  spin8_plan *plan = 0;
  double x[8] = {1, 2, 3, 4, 5, 6, 7, 8}, y[8];
  if (spin8_plan_create(0, 0, &plan) != SPIN8_OK)
    return 1;
  if (spin8_transform(plan, SPIN8_VECTOR, SPIN8_F64, x, 8, y, 8, 1, 8, 0) !=
      SPIN8_OK)
    return 2;
  spin8_plan_free(plan);
  {
    double a[7] = {0}, c[8];
    spin8_query_v3_basis_report report;
    if (SPIN8_QUERY_V3_VERSION != 3u ||
        spin8_query_v3_basis_audit(&report) != SPIN8_OK ||
        spin8_query_v3_spinor_from_tangent(a, c) != SPIN8_OK || c[0] != 1)
      return 5;
  }
  return y[7] == 8 ? 0 : 3;
}
