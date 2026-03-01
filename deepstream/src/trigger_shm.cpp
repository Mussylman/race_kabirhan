/**
 * trigger_shm.cpp — POSIX shared memory writer for trigger state.
 */

#include "trigger_shm.h"

#include <cstdio>
#include <cstring>
#include <atomic>
#include <chrono>

#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <semaphore.h>

namespace rv {

TriggerShmWriter::TriggerShmWriter() = default;

TriggerShmWriter::~TriggerShmWriter() {
    destroy();
}

bool TriggerShmWriter::create(uint32_t num_cameras) {
    num_cameras_ = num_cameras;

    shm_unlink(TRIGGER_SHM_NAME);
    sem_unlink(TRIGGER_SEM_NAME);

    shm_fd_ = shm_open(TRIGGER_SHM_NAME, O_CREAT | O_RDWR, 0666);
    if (shm_fd_ < 0) {
        perror("trigger shm_open");
        return false;
    }
    fchmod(shm_fd_, 0666);

    if (ftruncate(shm_fd_, TRIGGER_SHM_SIZE) < 0) {
        perror("trigger ftruncate");
        close(shm_fd_);
        shm_fd_ = -1;
        return false;
    }

    shm_raw_ = mmap(nullptr, TRIGGER_SHM_SIZE, PROT_READ | PROT_WRITE,
                     MAP_SHARED, shm_fd_, 0);
    if (shm_raw_ == MAP_FAILED) {
        perror("trigger mmap");
        close(shm_fd_);
        shm_fd_ = -1;
        shm_raw_ = nullptr;
        return false;
    }

    shm_ptr_ = reinterpret_cast<TriggerShmHeader*>(shm_raw_);
    std::memset(shm_ptr_, 0, TRIGGER_SHM_SIZE);
    shm_ptr_->num_cameras = num_cameras;

    sem_ = sem_open(TRIGGER_SEM_NAME, O_CREAT, 0666, 0);
    if (sem_ == SEM_FAILED) {
        perror("trigger sem_open");
        sem_ = nullptr;
        destroy();
        return false;
    }

    fprintf(stderr, "[TriggerSHM] Created '%s' (%zu bytes, %u cameras)\n",
            TRIGGER_SHM_NAME, TRIGGER_SHM_SIZE, num_cameras);
    return true;
}

void TriggerShmWriter::write(uint32_t active_mask,
                             const uint32_t detection_counts[MAX_CAMERAS]) {
    if (!shm_ptr_) return;

    shm_ptr_->active_mask = active_mask;
    std::memcpy(shm_ptr_->detection_counts, detection_counts,
                sizeof(uint32_t) * MAX_CAMERAS);
    shm_ptr_->timestamp_us = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now().time_since_epoch()
        ).count()
    );

    auto* seq = reinterpret_cast<std::atomic<uint64_t>*>(&shm_ptr_->write_seq);
    seq->fetch_add(1, std::memory_order_release);
    std::atomic_thread_fence(std::memory_order_release);

    if (sem_) {
        sem_post(static_cast<sem_t*>(sem_));
    }
}

void TriggerShmWriter::destroy() {
    if (shm_raw_ && shm_raw_ != MAP_FAILED) {
        munmap(shm_raw_, TRIGGER_SHM_SIZE);
        shm_raw_ = nullptr;
        shm_ptr_ = nullptr;
    }
    if (shm_fd_ >= 0) {
        close(shm_fd_);
        shm_fd_ = -1;
    }
    if (sem_) {
        sem_close(static_cast<sem_t*>(sem_));
        sem_ = nullptr;
    }

    shm_unlink(TRIGGER_SHM_NAME);
    sem_unlink(TRIGGER_SEM_NAME);
}

} // namespace rv
