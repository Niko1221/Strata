from sq10 import base, mx, unpack_head

mx.set_cache_limit(64 * 1024 * 1024)
weight, scales = unpack_head(compress=True)
mx.save_safetensors(str(base / 'draft-head-mxfp8.safetensors'), {'weight': weight, 'scales': scales})
