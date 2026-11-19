#ifndef __audio_queue_h__
#define __audio_queue_h__

// audio_queue.h - lock-free queue between threads
//
// Role:      single producer / single consumer ring with batch publishing.
// Contents:  audio_spsc_queue<T, N>: push (producer, not visible yet), commit (publishes everything pushed so far with
//            one store), pop (consumer).
// Thread:    one producer thread, one consumer thread.
// Used by:   audio_mixer (commands game -> audio: one commit per game frame, so all layers of an event land in one
//            callback; notifications audio -> game).

#include <stdint.h>
#include <atomic>

template<typename T, uint32_t N>
struct audio_spsc_queue
{
    static_assert((N & (N - 1)) == 0, "N must be a power of two");

    T                       items[N];
    std::atomic<uint32_t>   head{0};        // consumer
    std::atomic<uint32_t>   tail{0};        // published by the producer
    uint32_t                pending = 0;    // producer-private write index

    bool push(const T& v)
    {
        if (pending - head.load(std::memory_order_acquire) == N) return false;
        items[pending & (N - 1)] = v;
        ++pending;
        return true;
    }

    void commit() { tail.store(pending, std::memory_order_release); }

    bool pop(T& v)
    {
        uint32_t h = head.load(std::memory_order_relaxed);
        if (h == tail.load(std::memory_order_acquire)) return false;
        v = items[h & (N - 1)];
        head.store(h + 1, std::memory_order_release);
        return true;
    }

    uint32_t size() const { return tail.load(std::memory_order_acquire) - head.load(std::memory_order_acquire); }
};

#endif
