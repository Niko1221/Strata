/* Sha256.h -- SHA-256 Hash
2010-06-11 : Igor Pavlov : Public domain */

#ifndef SPIN8_SIDECAR_SHA256_H
#define SPIN8_SIDECAR_SHA256_H

#include <stdlib.h>
#include <stdint.h>

#define SHA256_DIGEST_SIZE 32

typedef struct spin8_sidecar_sha256_t
{
  uint32_t state[8];
  uint64_t count;
  unsigned char buffer[64];
} spin8_sidecar_sha256_t;

void spin8_sidecar_sha256_init(spin8_sidecar_sha256_t *p);
void spin8_sidecar_sha256_update(spin8_sidecar_sha256_t *p, const unsigned char *data, size_t size);
void spin8_sidecar_sha256_final(spin8_sidecar_sha256_t *p, unsigned char *digest);
void spin8_sidecar_sha256_hash(unsigned char *buf, const unsigned char *data, size_t size);

#endif
