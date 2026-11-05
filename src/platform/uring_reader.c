/* Linux io_uring READ: no liburing dependency; one submitter/reaper thread. */
#define _GNU_SOURCE
#include "strata/platform/uring_reader.h"
#include <errno.h>
#include <linux/io_uring.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

struct strata_uring {
    int fd;
    void* sq_map;
    void* cq_map;
    void* sqes_map;
    size_t sq_len, cq_len, sqes_len;
    unsigned *sq_head, *sq_tail, *sq_mask, *sq_array;
    unsigned *cq_head, *cq_tail, *cq_mask;
    struct io_uring_sqe* sqes;
    struct io_uring_cqe* cqes;
};

int strata_uring_fd(const struct strata_uring* r) { return r ? r->fd : -1; }

void strata_uring_close(struct strata_uring* r) {
    if (!r) return;
    if (r->sqes_map) munmap(r->sqes_map, r->sqes_len);
    if (r->cq_map && r->cq_map != r->sq_map) munmap(r->cq_map, r->cq_len);
    if (r->sq_map) munmap(r->sq_map, r->sq_len);
    if (r->fd >= 0) close(r->fd);
    free(r);
}

struct strata_uring* strata_uring_open(unsigned depth) {
    struct io_uring_params p;
    struct strata_uring* r;
    unsigned char *sq, *cq;
    if (depth < 2 || depth > 256 || (depth & (depth - 1))) return NULL;
    memset(&p, 0, sizeof p);
    r = (struct strata_uring*) calloc(1, sizeof *r);
    if (!r) return NULL;
    r->fd = -1;
    r->fd = (int) syscall(__NR_io_uring_setup, depth, &p);
    if (r->fd < 0) { strata_uring_close(r); return NULL; }
    r->sq_len = p.sq_off.array + p.sq_entries * sizeof(unsigned);
    r->cq_len = p.cq_off.cqes + p.cq_entries * sizeof(struct io_uring_cqe);
    if (p.features & IORING_FEAT_SINGLE_MMAP) {
        if (r->cq_len > r->sq_len) r->sq_len = r->cq_len;
        r->cq_len = r->sq_len;
    }
    r->sqes_len = p.sq_entries * sizeof(struct io_uring_sqe);
    r->sq_map = mmap(NULL, r->sq_len, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE,
                     r->fd, IORING_OFF_SQ_RING);
    if (r->sq_map == MAP_FAILED) { r->sq_map = NULL; strata_uring_close(r); return NULL; }
    if (p.features & IORING_FEAT_SINGLE_MMAP) r->cq_map = r->sq_map;
    else {
        r->cq_map = mmap(NULL, r->cq_len, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE,
                         r->fd, IORING_OFF_CQ_RING);
        if (r->cq_map == MAP_FAILED) { r->cq_map = NULL; strata_uring_close(r); return NULL; }
    }
    r->sqes_map = mmap(NULL, r->sqes_len, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE,
                       r->fd, IORING_OFF_SQES);
    if (r->sqes_map == MAP_FAILED) { r->sqes_map = NULL; strata_uring_close(r); return NULL; }
    sq = (unsigned char*) r->sq_map;
    cq = (unsigned char*) r->cq_map;
    r->sq_head = (unsigned*) (sq + p.sq_off.head);
    r->sq_tail = (unsigned*) (sq + p.sq_off.tail);
    r->sq_mask = (unsigned*) (sq + p.sq_off.ring_mask);
    r->sq_array = (unsigned*) (sq + p.sq_off.array);
    r->cq_head = (unsigned*) (cq + p.cq_off.head);
    r->cq_tail = (unsigned*) (cq + p.cq_off.tail);
    r->cq_mask = (unsigned*) (cq + p.cq_off.ring_mask);
    r->sqes = (struct io_uring_sqe*) r->sqes_map;
    r->cqes = (struct io_uring_cqe*) (cq + p.cq_off.cqes);
    return r;
}

int strata_uring_submit(struct strata_uring* r, int fd, uint64_t offset, void* buf,
                        uint32_t length, uint64_t tag) {
    unsigned tail, head, index;
    struct io_uring_sqe* sqe;
    int n;
    if (!r || fd < 0 || !buf || !length) return -EINVAL;
    tail = __atomic_load_n(r->sq_tail, __ATOMIC_RELAXED);
    head = __atomic_load_n(r->sq_head, __ATOMIC_ACQUIRE);
    if (tail - head >= *r->sq_mask + 1u) return -EAGAIN;
    index = tail & *r->sq_mask;
    sqe = &r->sqes[index];
    memset(sqe, 0, sizeof *sqe);
    sqe->opcode = IORING_OP_READ;
    sqe->fd = fd;
    sqe->off = offset;
    sqe->addr = (uint64_t) (uintptr_t) buf;
    sqe->len = length;
    sqe->user_data = tag;
    r->sq_array[index] = index;
    __atomic_store_n(r->sq_tail, tail + 1, __ATOMIC_RELEASE);
    do { n = (int) syscall(__NR_io_uring_enter, r->fd, 1, 0, 0, NULL, 0); } while (n < 0 && errno == EINTR);
    if (n == 1) return 0;
    /* If the SQ head moved, the request was consumed despite an interrupted enter. */
    if (__atomic_load_n(r->sq_head, __ATOMIC_ACQUIRE) != head) return 0;
    __atomic_store_n(r->sq_tail, tail, __ATOMIC_RELEASE);
    return -EIO;
}

int strata_uring_reap(struct strata_uring* r, uint64_t* tag, int* result) {
    unsigned head;
    const struct io_uring_cqe* cqe;
    if (!r || !tag || !result) return 0;
    head = __atomic_load_n(r->cq_head, __ATOMIC_RELAXED);
    if (head == __atomic_load_n(r->cq_tail, __ATOMIC_ACQUIRE)) return 0;
    cqe = &r->cqes[head & *r->cq_mask];
    *tag = cqe->user_data;
    *result = cqe->res;
    __atomic_store_n(r->cq_head, head + 1, __ATOMIC_RELEASE);
    return 1;
}