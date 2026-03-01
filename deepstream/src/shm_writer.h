#pragma once
/**
 * shm_writer.h — POSIX shared memory writer for detection results.
 *
 * Creates /rv_detections shared memory segment and /rv_detections_sem
 * semaphore. Writes per-camera detection slots and notifies Python reader.
 */

#include "config.h"
#include <string>
#include <vector>

namespace rv {

class ShmWriter {
public:
    ShmWriter();
    ~ShmWriter();

    // Non-copyable
    ShmWriter(const ShmWriter&) = delete;
    ShmWriter& operator=(const ShmWriter&) = delete;

    /**
     * Create shared memory segment and semaphore.
     * @param num_cameras Number of cameras to configure slots for.
     * @param cam_ids     Camera IDs to initialize slots with.
     * @return true on success.
     */
    bool create(uint32_t num_cameras, const std::vector<std::string>& cam_ids);

    /**
     * Write detection results for one camera.
     * Thread-safe (uses internal mutex for multi-probe scenarios).
     */
    void write_camera(uint32_t cam_index, const CameraSlot& slot);

    /**
     * Atomically increment write_seq and signal the semaphore.
     * Call after writing all camera slots for this cycle.
     */
    void commit();

    /**
     * Cleanup shared memory and semaphore.
     */
    void destroy();

    bool is_valid() const { return shm_ptr_ != nullptr; }

private:
    int         shm_fd_   = -1;
    void*       shm_raw_  = nullptr;
    ShmHeader*  shm_ptr_  = nullptr;
    void*       sem_      = nullptr; // sem_t*
};

} // namespace rv
