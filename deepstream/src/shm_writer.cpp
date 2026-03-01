/**
 * shm_writer.cpp — POSIX shared memory writer implementation.
 */

#include "shm_writer.h"

#include <cstdio>
#include <cstring>
#include <atomic>
#include <mutex>

#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <semaphore.h>

namespace rv {

static std::mutex g_write_mutex;

ShmWriter::ShmWriter() = default;

ShmWriter::~ShmWriter() {
    destroy();
}

bool ShmWriter::create(uint32_t num_cameras, const std::vector<std::string>& cam_ids) {
    // Unlink any stale segments
    shm_unlink(SHM_NAME);
    sem_unlink(SEM_NAME);

    // Create shared memory (world-readable/writable for cross-user IPC)
    shm_fd_ = shm_open(SHM_NAME, O_CREAT | O_RDWR, 0666);
    if (shm_fd_ < 0) {
        perror("shm_open");
        return false;
    }
    fchmod(shm_fd_, 0666);

    if (ftruncate(shm_fd_, SHM_SIZE) < 0) {
        perror("ftruncate");
        close(shm_fd_);
        shm_fd_ = -1;
        return false;
    }

    shm_raw_ = mmap(nullptr, SHM_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd_, 0);
    if (shm_raw_ == MAP_FAILED) {
        perror("mmap");
        close(shm_fd_);
        shm_fd_ = -1;
        shm_raw_ = nullptr;
        return false;
    }

    // Initialize header
    shm_ptr_ = reinterpret_cast<ShmHeader*>(shm_raw_);
    std::memset(shm_ptr_, 0, SHM_SIZE);
    shm_ptr_->num_cameras = num_cameras;

    // Initialize camera slots
    for (uint32_t i = 0; i < num_cameras && i < MAX_CAMERAS; ++i) {
        if (i < cam_ids.size()) {
            init_camera_slot(shm_ptr_->cameras[i], cam_ids[i].c_str());
        }
    }

    // Create semaphore (initial value 0, world-accessible)
    sem_ = sem_open(SEM_NAME, O_CREAT, 0666, 0);
    if (sem_ == SEM_FAILED) {
        perror("sem_open");
        sem_ = nullptr;
        destroy();
        return false;
    }
    // Force permissions (umask may restrict 0666 on creation)
    chmod("/dev/shm/sem.rv_detections_sem", 0666);

    fprintf(stderr, "[ShmWriter] Created SHM '%s' (%zu bytes, %u cameras)\n",
            SHM_NAME, SHM_SIZE, num_cameras);
    return true;
}

void ShmWriter::write_camera(uint32_t cam_index, const CameraSlot& slot) {
    if (!shm_ptr_ || cam_index >= MAX_CAMERAS) return;

    std::lock_guard<std::mutex> lock(g_write_mutex);
    std::memcpy(&shm_ptr_->cameras[cam_index], &slot, sizeof(CameraSlot));
}

void ShmWriter::commit() {
    if (!shm_ptr_) return;

    // Atomic increment of write_seq
    auto* seq = reinterpret_cast<std::atomic<uint64_t>*>(&shm_ptr_->write_seq);
    seq->fetch_add(1, std::memory_order_release);

    // Memory fence to ensure all writes are visible
    std::atomic_thread_fence(std::memory_order_release);

    // Signal semaphore
    if (sem_) {
        sem_post(static_cast<sem_t*>(sem_));
    }
}

void ShmWriter::destroy() {
    if (shm_raw_ && shm_raw_ != MAP_FAILED) {
        munmap(shm_raw_, SHM_SIZE);
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

    shm_unlink(SHM_NAME);
    sem_unlink(SEM_NAME);
}

} // namespace rv
