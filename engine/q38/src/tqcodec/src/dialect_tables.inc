// Private arithmetic/table material derived from TheTom
// 643956891dfdf51aecdbd05ca508dd21e21e8195. See DIALECT_LICENSE.txt. No GGML
// ABI or dependency is exported.
static const float CENTROIDS_2BIT[4] = {-0.133462f, -0.039994f, 0.039994f,
                                        0.133462f};
static const float CENTROIDS_3BIT[8] = {-0.190207f, -0.118786f, -0.066822f,
                                        -0.021663f, 0.021663f,  0.066822f,
                                        0.118786f,  0.190207f};
static const float TURBO6_CENTROIDS[64] = {
    -0.330935f, -0.286417f, -0.257865f, -0.236198f, -0.218435f, -0.203203f,
    -0.189753f, -0.177626f, -0.166522f, -0.156230f, -0.146600f, -0.137517f,
    -0.128895f, -0.120663f, -0.112765f, -0.105157f, -0.097801f, -0.090663f,
    -0.083717f, -0.076940f, -0.070310f, -0.063809f, -0.057422f, -0.051133f,
    -0.044929f, -0.038798f, -0.032729f, -0.026710f, -0.020733f, -0.014787f,
    -0.008863f, -0.002953f, 0.002953f,  0.008863f,  0.014787f,  0.020733f,
    0.026710f,  0.032729f,  0.038798f,  0.044929f,  0.051133f,  0.057422f,
    0.063809f,  0.070310f,  0.076940f,  0.083717f,  0.090663f,  0.097801f,
    0.105157f,  0.112765f,  0.120663f,  0.128895f,  0.137517f,  0.146600f,
    0.156230f,  0.166522f,  0.177626f,  0.189753f,  0.203203f,  0.218435f,
    0.236198f,  0.257865f,  0.286417f,  0.330935f};
static const float TURBO6_MID[63] = {
    -0.308676f, -0.272141f, -0.247031f, -0.227316f, -0.210819f, -0.196478f,
    -0.183690f, -0.172074f, -0.161376f, -0.151415f, -0.142059f, -0.133206f,
    -0.124779f, -0.116714f, -0.108961f, -0.101479f, -0.094232f, -0.087190f,
    -0.080329f, -0.073625f, -0.067060f, -0.060616f, -0.054277f, -0.048031f,
    -0.041864f, -0.035763f, -0.029719f, -0.023722f, -0.017760f, -0.011825f,
    -0.005908f, 0.000000f,  0.005908f,  0.011825f,  0.017760f,  0.023722f,
    0.029719f,  0.035763f,  0.041864f,  0.048031f,  0.054277f,  0.060616f,
    0.067060f,  0.073625f,  0.080329f,  0.087190f,  0.094232f,  0.101479f,
    0.108961f,  0.116714f,  0.124779f,  0.133206f,  0.142059f,  0.151415f,
    0.161376f,  0.172074f,  0.183690f,  0.196478f,  0.210819f,  0.227316f,
    0.247031f,  0.272141f,  0.308676f};
static const float TURBO5_CENTROIDS[32] = {
    -0.288236f, -0.237892f, -0.204892f, -0.179348f, -0.158003f, -0.139353f,
    -0.122569f, -0.107141f, -0.092730f, -0.079097f, -0.066064f, -0.053491f,
    -0.041269f, -0.029304f, -0.017515f, -0.005827f, 0.005827f,  0.017515f,
    0.029304f,  0.041269f,  0.053491f,  0.066064f,  0.079097f,  0.092730f,
    0.107141f,  0.122569f,  0.139353f,  0.158003f,  0.179348f,  0.204892f,
    0.237892f,  0.288236f};
static const float TURBO5_MID[31] = {
    -0.263064f, -0.221392f, -0.192120f, -0.168675f, -0.148678f, -0.130961f,
    -0.114855f, -0.099935f, -0.085914f, -0.072580f, -0.059778f, -0.047380f,
    -0.035287f, -0.023410f, -0.011671f, 0.000000f,  0.011671f,  0.023410f,
    0.035287f,  0.047380f,  0.059778f,  0.072580f,  0.085914f,  0.099935f,
    0.114855f,  0.130961f,  0.148678f,  0.168675f,  0.192120f,  0.221392f,
    0.263064f};
static const float turbo_cpu_s1[128] = {
    -1, 1,  1,  -1, -1, 1,  -1, 1,  -1, -1, 1,  1,  1,  1,  1,  1,  1,  -1, 1,
    -1, 1,  -1, -1, 1,  1,  1,  -1, 1,  1,  -1, -1, -1, -1, 1,  1,  -1, 1,  1,
    -1, 1,  -1, 1,  1,  -1, -1, 1,  -1, 1,  1,  1,  1,  -1, -1, -1, -1, -1, 1,
    -1, 1,  1,  1,  1,  -1, 1,  -1, -1, 1,  -1, -1, -1, 1,  -1, -1, -1, 1,  -1,
    -1, -1, 1,  1,  1,  -1, -1, 1,  1,  1,  -1, -1, 1,  1,  -1, 1,  1,  -1, 1,
    -1, -1, 1,  1,  -1, 1,  -1, 1,  -1, 1,  1,  1,  1,  -1, 1,  -1, 1,  1,  -1,
    1,  1,  -1, -1, -1, -1, -1, 1,  1,  -1, 1,  1,  -1, 1};
static const float turbo_cpu_s2[128] = {
    1,  1,  1,  1,  -1, 1,  1,  -1, 1,  -1, -1, -1, 1,  -1, -1, -1, 1,  1,  -1,
    -1, 1,  -1, 1,  -1, 1,  -1, -1, 1,  -1, 1,  1,  1,  1,  1,  -1, -1, -1, 1,
    -1, -1, -1, -1, -1, -1, 1,  1,  1,  -1, 1,  -1, 1,  1,  1,  -1, -1, 1,  -1,
    -1, -1, -1, -1, -1, 1,  1,  1,  -1, 1,  -1, -1, -1, -1, 1,  -1, 1,  -1, 1,
    -1, -1, 1,  1,  -1, 1,  -1, 1,  1,  -1, 1,  -1, -1, -1, -1, 1,  -1, -1, 1,
    -1, 1,  -1, 1,  1,  1,  -1, -1, 1,  -1, 1,  -1, 1,  1,  -1, -1, 1,  -1, 1,
    -1, 1,  1,  -1, 1,  -1, 1,  -1, -1, -1, -1, -1, 1,  -1};
static const float CENTROIDS_4BIT[16] = {
    -0.241529f, -0.182877f, -0.143016f, -0.111036f, -0.083292f, -0.058050f,
    -0.034299f, -0.011349f, 0.011349f,  0.034299f,  0.058050f,  0.083292f,
    0.111036f,  0.143016f,  0.182877f,  0.241529f};
static const float TQ3_0_CENTROIDS[8] = {-1.996684f, -1.291398f, -0.740341f,
                                         -0.247508f, 0.230106f,  0.725222f,
                                         1.277503f,  1.988943f};
static const float TQ4_0_CENTROIDS[16] = {
    -2.732590f, -2.069017f, -1.618046f, -1.256231f, -0.942340f, -0.656759f,
    -0.388048f, -0.128395f, 0.128395f,  0.388048f,  0.656759f,  0.942340f,
    1.256231f,  1.618046f,  2.069017f,  2.732590f,
};
static const float TQ3_0_SIGNS[32] = {
    +1.0f, -1.0f, +1.0f, -1.0f, +1.0f, +1.0f, -1.0f, +1.0f, -1.0f, -1.0f, +1.0f,
    -1.0f, +1.0f, +1.0f, -1.0f, +1.0f, -1.0f, -1.0f, +1.0f, -1.0f, +1.0f, -1.0f,
    -1.0f, +1.0f, -1.0f, +1.0f, +1.0f, -1.0f, +1.0f, -1.0f, -1.0f, +1.0f,
};
#define TQ_BLOCK_SIZE 32
#define TQ_INV_SQRT32 0.17677669529663688f
#define QK_TQ3_0 32
#define QK_TQ4_1S 32
struct block_tq3_1s {
  uint16_t d0, d1;
  uint8_t qs[12];
};
struct block_tq4_1s {
  uint16_t d0, d1;
  uint8_t qs[16];
};
static int nearest_centroid_2bit(float val) {
  /* Binary search on midpoints: {-0.133, -0.040, 0.040, 0.133} */
  if (val < -0.086728f)
    return 0; /* midpoint(-0.133, -0.040) */
  if (val < 0.000000f)
    return 1; /* midpoint(-0.040, 0.040) */
  if (val < 0.086728f)
    return 2; /* midpoint(0.040, 0.133) */
  return 3;
}
static int nearest_centroid_3bit(float val) {
  /* 8 centroids, find nearest via midpoints */
  if (val < -0.154496f)
    return 0;
  if (val < -0.092804f)
    return 1;
  if (val < -0.044243f)
    return 2;
  if (val < 0.000000f)
    return 3;
  if (val < 0.044243f)
    return 4;
  if (val < 0.092804f)
    return 5;
  if (val < 0.154496f)
    return 6;
  return 7;
}
static int nearest_centroid_4bit(float val) {
  /* 16 centroids, optimal for N(0, 1/sqrt(128)), find nearest via midpoints */
  if (val < -0.212203f)
    return 0;
  if (val < -0.162947f)
    return 1;
  if (val < -0.127026f)
    return 2;
  if (val < -0.097164f)
    return 3;
  if (val < -0.070671f)
    return 4;
  if (val < -0.046174f)
    return 5;
  if (val < -0.022824f)
    return 6;
  if (val < 0.000000f)
    return 7;
  if (val < 0.022824f)
    return 8;
  if (val < 0.046174f)
    return 9;
  if (val < 0.070671f)
    return 10;
  if (val < 0.097164f)
    return 11;
  if (val < 0.127026f)
    return 12;
  if (val < 0.162947f)
    return 13;
  if (val < 0.212203f)
    return 14;
  return 15;
}
static int nearest_centroid_5bit(float val) {
  int lo = 0;
  int hi = 31;
  while (lo < hi) {
    const int mid = (lo + hi) / 2;
    if (val < TURBO5_MID[mid]) {
      hi = mid;
    } else {
      lo = mid + 1;
    }
  }
  return lo;
}
static int nearest_centroid_6bit(float val) {
  int lo = 0; /* candidate code range [lo, hi] */
  int hi = 63;
  while (lo < hi) {
    const int mid =
        (lo + hi) / 2; /* compare against the lo/hi split at TURBO6_MID[mid] */
    if (val < TURBO6_MID[mid]) {
      hi = mid;
    } else {
      lo = mid + 1;
    }
  }
  return lo;
}
static void tq3_0_rht_forward(float *buf) {
  for (int i = 0; i < TQ_BLOCK_SIZE; i++)
    buf[i] *= TQ3_0_SIGNS[i];
  for (int step = 1; step < TQ_BLOCK_SIZE; step <<= 1) {
    for (int i = 0; i < TQ_BLOCK_SIZE; i += step << 1) {
      for (int j = i; j < i + step; j++) {
        float a = buf[j], b = buf[j + step];
        buf[j] = a + b;
        buf[j + step] = a - b;
      }
    }
  }
  for (int i = 0; i < TQ_BLOCK_SIZE; i++)
    buf[i] *= TQ_INV_SQRT32;
}
static void tq3_0_rht_inverse(float *buf) {
  for (int step = 1; step < TQ_BLOCK_SIZE; step <<= 1) {
    for (int i = 0; i < TQ_BLOCK_SIZE; i += step << 1) {
      for (int j = i; j < i + step; j++) {
        float a = buf[j], b = buf[j + step];
        buf[j] = a + b;
        buf[j + step] = a - b;
      }
    }
  }
  for (int i = 0; i < TQ_BLOCK_SIZE; i++)
    buf[i] *= TQ_INV_SQRT32 * TQ3_0_SIGNS[i];
}
static int tq3_0_choose_index(float val) {
  /* Binary search on midpoints of TQ3_0_CENTROIDS */
  if (val < -1.644041f)
    return 0;
  if (val < -1.015870f)
    return 1;
  if (val < -0.493925f)
    return 2;
  if (val < -0.008701f)
    return 3;
  if (val < 0.477664f)
    return 4;
  if (val < 1.001363f)
    return 5;
  if (val < 1.633223f)
    return 6;
  return 7;
}
static int tq4_0_choose_index(float val) {
  /* Binary search on midpoints of TQ4_0_CENTROIDS */
  if (val < -2.400804f)
    return 0;
  if (val < -1.843532f)
    return 1;
  if (val < -1.437139f)
    return 2;
  if (val < -1.099286f)
    return 3;
  if (val < -0.799550f)
    return 4;
  if (val < -0.522404f)
    return 5;
  if (val < -0.258222f)
    return 6;
  if (val < 0.000000f)
    return 7;
  if (val < 0.258222f)
    return 8;
  if (val < 0.522404f)
    return 9;
  if (val < 0.799550f)
    return 10;
  if (val < 1.099286f)
    return 11;
  if (val < 1.437139f)
    return 12;
  if (val < 1.843532f)
    return 13;
  if (val < 2.400804f)
    return 14;
  return 15;
}
static void quantize_row_tq3_1s_ref(const float *x, block_tq3_1s *y, int k) {
  assert(k % QK_TQ3_0 == 0);
  const int nb = k / QK_TQ3_0;

  for (int block = 0; block < nb; block++) {
    const float *src_blk = x + block * QK_TQ3_0;
    block_tq3_1s *blk = &y[block];

    /* 1. Forward RHT */
    float buf[TQ_BLOCK_SIZE];
    memcpy(buf, src_blk, TQ_BLOCK_SIZE * sizeof(float));
    tq3_0_rht_forward(buf);

    /* 2. Split into two halves, compute RMS per half */
    float rms0 = 0.0f, rms1 = 0.0f;
    for (int j = 0; j < 16; j++)
      rms0 += buf[j] * buf[j];
    for (int j = 16; j < 32; j++)
      rms1 += buf[j] * buf[j];
    rms0 = sqrtf(rms0 / 16.0f);
    rms1 = sqrtf(rms1 / 16.0f);

    /* 3. Scale search (9 points) */
    static const float scales[] = {0.6f, 0.7f, 0.8f,  0.9f, 1.0f,
                                   1.1f, 1.2f, 1.35f, 1.5f};
    float best_d0 = rms0, best_d1 = rms1;
    float best_err = 1e30f;

    for (int si = 0; si < 9; si++) {
      float d0 = rms0 * scales[si];
      float d1 = rms1 * scales[si];
      float inv0 = (d0 > 1e-10f) ? 1.0f / d0 : 0.0f;
      float inv1 = (d1 > 1e-10f) ? 1.0f / d1 : 0.0f;

      float err = 0.0f;
      for (int j = 0; j < 16; j++) {
        int idx = tq3_0_choose_index(buf[j] * inv0);
        float diff = buf[j] - TQ3_0_CENTROIDS[idx] * d0;
        err += diff * diff;
      }
      for (int j = 16; j < 32; j++) {
        int idx = tq3_0_choose_index(buf[j] * inv1);
        float diff = buf[j] - TQ3_0_CENTROIDS[idx] * d1;
        err += diff * diff;
      }
      if (err < best_err) {
        best_err = err;
        best_d0 = d0;
        best_d1 = d1;
      }
    }

    /* 4. Iterative refinement (6 iterations) */
    for (int iter = 0; iter < 6; iter++) {
      float inv0 = (best_d0 > 1e-10f) ? 1.0f / best_d0 : 0.0f;
      float inv1 = (best_d1 > 1e-10f) ? 1.0f / best_d1 : 0.0f;

      float num0 = 0.0f, den0 = 0.0f;
      float num1 = 0.0f, den1 = 0.0f;
      for (int j = 0; j < 16; j++) {
        int idx = tq3_0_choose_index(buf[j] * inv0);
        float c = TQ3_0_CENTROIDS[idx];
        num0 += buf[j] * c;
        den0 += c * c;
      }
      for (int j = 16; j < 32; j++) {
        int idx = tq3_0_choose_index(buf[j] * inv1);
        float c = TQ3_0_CENTROIDS[idx];
        num1 += buf[j] * c;
        den1 += c * c;
      }
      if (den0 > 1e-10f)
        best_d0 = num0 / den0;
      if (den1 > 1e-10f)
        best_d1 = num1 / den1;
    }

    /* 5. Final quantize + pack */
    float inv0 = (best_d0 > 1e-10f) ? 1.0f / best_d0 : 0.0f;
    float inv1 = (best_d1 > 1e-10f) ? 1.0f / best_d1 : 0.0f;

    blk->d0 = half_encode(best_d0);
    blk->d1 = half_encode(best_d1);
    memset(blk->qs, 0, QK_TQ3_0 * 3 / 8);

    /* TQ3 packing: 4 groups of 8 indices packed into 3 bytes each */
    for (int g = 0; g < 4; g++) {
      uint8_t indices[8];
      for (int i = 0; i < 8; i++) {
        int j = g * 8 + i;
        float inv = (j < 16) ? inv0 : inv1;
        indices[i] = (uint8_t)tq3_0_choose_index(buf[j] * inv);
      }
      uint8_t *qp = blk->qs + g * 3;
      qp[0] =
          (indices[0] & 7) | ((indices[1] & 7) << 3) | ((indices[2] & 3) << 6);
      qp[1] = ((indices[2] >> 2) & 1) | ((indices[3] & 7) << 1) |
              ((indices[4] & 7) << 4) | ((indices[5] & 1) << 7);
      qp[2] = ((indices[5] >> 1) & 3) | ((indices[6] & 7) << 2) |
              ((indices[7] & 7) << 5);
    }
  }
}
static void dequantize_row_tq3_1s(const block_tq3_1s *x, float *y, int k) {
  assert(k % QK_TQ3_0 == 0);
  const int nb = k / QK_TQ3_0;

  for (int blk_i = 0; blk_i < nb; blk_i++) {
    float d0 = half_decode(x[blk_i].d0);
    float d1 = half_decode(x[blk_i].d1);

    /* Unpack 3-bit indices */
    float buf[32];
    for (int g = 0; g < 4; g++) {
      const uint8_t *qp = x[blk_i].qs + g * 3;
      uint8_t idx[8];
      idx[0] = qp[0] & 7;
      idx[1] = (qp[0] >> 3) & 7;
      idx[2] = ((qp[0] >> 6) | (qp[1] << 2)) & 7;
      idx[3] = (qp[1] >> 1) & 7;
      idx[4] = (qp[1] >> 4) & 7;
      idx[5] = ((qp[1] >> 7) | (qp[2] << 1)) & 7;
      idx[6] = (qp[2] >> 2) & 7;
      idx[7] = (qp[2] >> 5) & 7;

      for (int i = 0; i < 8; i++) {
        int j = g * 8 + i;
        float d = (j < 16) ? d0 : d1;
        buf[j] = TQ3_0_CENTROIDS[idx[i]] * d;
      }
    }

    /* Inverse RHT */
    tq3_0_rht_inverse(buf);

    memcpy(y + blk_i * QK_TQ3_0, buf, QK_TQ3_0 * sizeof(float));
  }
}
static void quantize_row_tq4_1s_ref(const float *x, block_tq4_1s *y, int k) {
  assert(k % QK_TQ4_1S == 0);
  const int nb = k / QK_TQ4_1S;

  for (int block = 0; block < nb; block++) {
    const float *src_blk = x + block * QK_TQ4_1S;
    block_tq4_1s *blk = &y[block];

    /* 1. Forward RHT */
    float buf[TQ_BLOCK_SIZE];
    memcpy(buf, src_blk, TQ_BLOCK_SIZE * sizeof(float));
    tq3_0_rht_forward(buf);

    /* 2. Split into two halves, compute RMS per half */
    float rms0 = 0.0f, rms1 = 0.0f;
    for (int j = 0; j < 16; j++)
      rms0 += buf[j] * buf[j];
    for (int j = 16; j < 32; j++)
      rms1 += buf[j] * buf[j];
    rms0 = sqrtf(rms0 / 16.0f);
    rms1 = sqrtf(rms1 / 16.0f);

    /* 3. Scale search (9 points) */
    static const float scales[] = {0.6f, 0.7f, 0.8f,  0.9f, 1.0f,
                                   1.1f, 1.2f, 1.35f, 1.5f};
    float best_d0 = rms0, best_d1 = rms1;
    float best_err = 1e30f;

    for (int si = 0; si < 9; si++) {
      float d0 = rms0 * scales[si];
      float d1 = rms1 * scales[si];
      float inv0 = (d0 > 1e-10f) ? 1.0f / d0 : 0.0f;
      float inv1 = (d1 > 1e-10f) ? 1.0f / d1 : 0.0f;

      float err = 0.0f;
      for (int j = 0; j < 16; j++) {
        int idx = tq4_0_choose_index(buf[j] * inv0);
        float diff = buf[j] - TQ4_0_CENTROIDS[idx] * d0;
        err += diff * diff;
      }
      for (int j = 16; j < 32; j++) {
        int idx = tq4_0_choose_index(buf[j] * inv1);
        float diff = buf[j] - TQ4_0_CENTROIDS[idx] * d1;
        err += diff * diff;
      }
      if (err < best_err) {
        best_err = err;
        best_d0 = d0;
        best_d1 = d1;
      }
    }

    /* 4. Iterative refinement (6 iterations) */
    for (int iter = 0; iter < 6; iter++) {
      float inv0 = (best_d0 > 1e-10f) ? 1.0f / best_d0 : 0.0f;
      float inv1 = (best_d1 > 1e-10f) ? 1.0f / best_d1 : 0.0f;

      float num0 = 0.0f, den0 = 0.0f;
      float num1 = 0.0f, den1 = 0.0f;
      for (int j = 0; j < 16; j++) {
        int idx = tq4_0_choose_index(buf[j] * inv0);
        float c = TQ4_0_CENTROIDS[idx];
        num0 += buf[j] * c;
        den0 += c * c;
      }
      for (int j = 16; j < 32; j++) {
        int idx = tq4_0_choose_index(buf[j] * inv1);
        float c = TQ4_0_CENTROIDS[idx];
        num1 += buf[j] * c;
        den1 += c * c;
      }
      if (den0 > 1e-10f)
        best_d0 = num0 / den0;
      if (den1 > 1e-10f)
        best_d1 = num1 / den1;
    }

    /* 5. Final quantize + pack (nibble packing) */
    float inv0 = (best_d0 > 1e-10f) ? 1.0f / best_d0 : 0.0f;
    float inv1 = (best_d1 > 1e-10f) ? 1.0f / best_d1 : 0.0f;

    blk->d0 = half_encode(best_d0);
    blk->d1 = half_encode(best_d1);
    memset(blk->qs, 0, QK_TQ4_1S / 2);

    for (int j = 0; j < QK_TQ4_1S; j++) {
      float inv = (j < 16) ? inv0 : inv1;
      int idx = tq4_0_choose_index(buf[j] * inv);
      blk->qs[j / 2] |= (uint8_t)((idx & 0xF) << ((j & 1) * 4));
    }
  }
}
static void dequantize_row_tq4_1s(const block_tq4_1s *x, float *y, int k) {
  assert(k % QK_TQ4_1S == 0);
  const int nb = k / QK_TQ4_1S;

  for (int blk_i = 0; blk_i < nb; blk_i++) {
    float d0 = half_decode(x[blk_i].d0);
    float d1 = half_decode(x[blk_i].d1);

    float buf[32];
    for (int j = 0; j < 32; j++) {
      uint8_t idx = (x[blk_i].qs[j / 2] >> ((j & 1) * 4)) & 0xF;
      float d = (j < 16) ? d0 : d1;
      buf[j] = TQ4_0_CENTROIDS[idx] * d;
    }

    /* Inverse RHT */
    tq3_0_rht_inverse(buf);

    memcpy(y + blk_i * QK_TQ4_1S, buf, QK_TQ4_1S * sizeof(float));
  }
}
