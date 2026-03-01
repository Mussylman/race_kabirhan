#pragma once
/**
 * trigger_shm.h — POSIX shared memory writer for trigger activation state.
 *
 * Writes camera activation bitmask and per-camera detection counts to
 * /rv_trigger shared memory for Python consumption.
 */

#include "config.h"

namespace rv {

class TriggerShmWriter {
public:
    TriggerShmWriter();
    ~TriggerShmWriter();

    TriggerShmWriter(const TriggerShmWriter&) = delete;
    TriggerShmWriter& operator=(const TriggerShmWriter&) = delete;

    bool create(uint32_t num_cameras);
    void write(uint32_t active_mask, const uint32_t detection_counts[MAX_CAMERAS]);
    void destroy();

    bool is_valid() const { return shm_ptr_ != nullptr; }

private:
    int                shm_fd_  = -1;
    void*              shm_raw_ = nullptr;
    TriggerShmHeader*  shm_ptr_ = nullptr;
    void*              sem_     = nullptr; // sem_t*
    uint32_t           num_cameras_ = 0;
};

} // namespace rv
