#ifndef STRATA_URING_READER_H
#define STRATA_URING_READER_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
struct strata_uring;
/* Fixed-depth, single submitter/reaper.  Buffers must remain alive until reaped. */
struct strata_uring* strata_uring_open(unsigned depth);
int strata_uring_submit(struct strata_uring*, int fd, uint64_t offset, void* buf,
                        uint32_t length, uint64_t tag); /* 0 or negative errno */
int strata_uring_reap(struct strata_uring*, uint64_t* tag, int* result); /* 1 or 0 */
int strata_uring_fd(const struct strata_uring*);
void strata_uring_close(struct strata_uring*); /* caller must drain outstanding reads first */
#ifdef __cplusplus
}
#endif
#endif