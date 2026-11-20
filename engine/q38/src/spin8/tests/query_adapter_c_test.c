#include "spin8/query_adapter.h"
#include <string.h>

/* This file is compiled as C, and consumes an object loaded by the C++ test. */
int spin8_query_adapter_c_check(const spin8_query_adapter *adapter) {
  spin8_query_adapter_info info;
  spin8_query_adapter_layer_view view;
  unsigned char before[sizeof(view)];
  if (spin8_query_adapter_get_info(adapter, &info, sizeof(info)) != SPIN8_OK ||
      info.abi_version != SPIN8_QUERY_ADAPTER_ABI_VERSION ||
      info.layer_count != 2)
    return 1;
  if (spin8_query_adapter_get_layer(adapter, 0, &view, sizeof(view)) !=
          SPIN8_OK ||
      view.spec.q_heads != 24 || view.spec.kv_heads != 4 ||
      view.spec.head_width != 256 || view.win_bytes != 24u * 8u * 256u * 4u)
    return 2;
  memcpy(before, &view, sizeof(view));
  if (spin8_query_adapter_get_layer(adapter, 2, &view, sizeof(view)) !=
          SPIN8_INVALID_ARGUMENT ||
      memcmp(before, &view, sizeof(view)) != 0)
    return 3;
  if (spin8_query_adapter_get_layer(adapter, 0, &view, sizeof(view) - 1) !=
          SPIN8_INVALID_ARGUMENT ||
      memcmp(before, &view, sizeof(view)) != 0)
    return 4;
  return 0;
}
