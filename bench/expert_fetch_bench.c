/* C reference for the pulsar milestone-2 benchmark: execute the same plan
 * file as fetch-bench (offset len per line) with liburing + O_DIRECT and
 * identical aligned-bracket reads, so Rust and C perform byte-identical
 * I/O. Build: gcc -O2 -o expert_fetch_bench expert_fetch_bench.c -luring
 * Run:   ./expert_fetch_bench <model.gguf> <plan-file> <qd> */
#define _GNU_SOURCE
#include <fcntl.h>
#include <liburing.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define ALIGN 4096ull

typedef struct {
    uint64_t offset, len;
} plan_read;

typedef struct {
    void *buf;
    uint64_t disk_len;
    size_t payload_off, payload_len;
    int busy;
} slot_t;

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: %s <model.gguf> <plan-file> <qd>\n", argv[0]);
        return 2;
    }
    const int qd = atoi(argv[3]);
    if (qd < 1 || qd > 512) {
        fprintf(stderr, "bad qd\n");
        return 2;
    }

    FILE *pf = fopen(argv[2], "r");
    if (!pf) {
        perror("plan");
        return 1;
    }
    size_t cap = 1 << 16, n = 0;
    plan_read *reads = malloc(cap * sizeof(*reads));
    while (fscanf(pf, "%llu %llu",
                  (unsigned long long *)&reads[n].offset,
                  (unsigned long long *)&reads[n].len) == 2) {
        if (++n == cap) {
            cap *= 2;
            reads = realloc(reads, cap * sizeof(*reads));
        }
    }
    fclose(pf);

    int fd = open(argv[1], O_RDONLY | O_DIRECT);
    if (fd < 0) {
        perror("open O_DIRECT");
        return 1;
    }

    struct io_uring ring;
    if (io_uring_queue_init((unsigned)qd * 2, &ring, 0) != 0) {
        perror("uring init");
        return 1;
    }

    slot_t *slots = calloc((size_t)qd, sizeof(*slots));
    uint64_t bytes_payload = 0, bytes_disk = 0, done = 0;
    uint8_t checksum = 0;
    size_t next = 0;
    int inflight = 0;
    const double t0 = now_sec();

    for (;;) {
        while (inflight < qd && next < n) {
            const plan_read r = reads[next];
            const uint64_t aligned_off = r.offset & ~(ALIGN - 1);
            const size_t payload_off = (size_t)(r.offset - aligned_off);
            const uint64_t disk_len =
                ((payload_off + r.len + ALIGN - 1) / ALIGN) * ALIGN;
            int s = -1;
            for (int i = 0; i < qd; i++) {
                if (!slots[i].busy) { s = i; break; }
            }
            slot_t *sl = &slots[s];
            if (!sl->buf || sl->disk_len < disk_len) {
                free(sl->buf);
                if (posix_memalign(&sl->buf, ALIGN, disk_len) != 0) {
                    fprintf(stderr, "oom\n");
                    return 1;
                }
                sl->disk_len = disk_len;
            }
            sl->payload_off = payload_off;
            sl->payload_len = (size_t)r.len;
            sl->busy = 1;
            struct io_uring_sqe *sqe = io_uring_get_sqe(&ring);
            io_uring_prep_read(sqe, fd, sl->buf, (unsigned)disk_len,
                               (off_t)aligned_off);
            io_uring_sqe_set_data64(sqe, (uint64_t)s);
            bytes_disk += disk_len;
            inflight++;
            next++;
        }
        if (inflight == 0) break;
        io_uring_submit_and_wait(&ring, 1);
        struct io_uring_cqe *cqe;
        unsigned head, seen = 0;
        io_uring_for_each_cqe(&ring, head, cqe) {
            slot_t *sl = &slots[io_uring_cqe_get_data64(cqe)];
            if (cqe->res < 0) {
                fprintf(stderr, "read: %s\n", strerror(-cqe->res));
                return 1;
            }
            checksum ^= ((uint8_t *)sl->buf)[sl->payload_off + sl->payload_len / 2];
            bytes_payload += sl->payload_len;
            done++;
            sl->busy = 0;
            inflight--;
            seen++;
        }
        io_uring_cq_advance(&ring, seen);
    }
    const double secs = now_sec() - t0;
    printf("c:    %llu reads, payload %.2f GiB, disk %.2f GiB, %.3f s, "
           "%.2f GB/s payload, %.2f GB/s disk, checksum %02x\n",
           (unsigned long long)done,
           (double)bytes_payload / (double)(1ull << 30),
           (double)bytes_disk / (double)(1ull << 30),
           secs,
           (double)bytes_payload / secs / 1e9,
           (double)bytes_disk / secs / 1e9,
           checksum);
    return 0;
}
