#ifndef __audio_stream_h__
#define __audio_stream_h__

// audio_stream.h - streaming from files (IO thread)
//
// Role:      long sounds stay on disk; the IO thread fills a small ring per stream, the mixer plays from the ring.
//            IMA ADPCM WAV: the ring holds file blocks, the mixer decodes them. Ogg Vorbis: the IO thread decodes
//            (audio_vorbis.h) into mono int16 blocks of 1024 frames, the mixer copies them.
// Contents:
//   - audio_stream          one slot: state, format (from the file header), ring, produced / consumed counters
//   - audio_stream_system   slot pool, ring memory, Vorbis decoders with their memory and rings (all allocated once),
//                           IO thread, file IO (audio_file_io, stdio default)
// Thread:    game thread: open / close / state. IO thread: opens files, reads / decodes blocks. Audio thread (mixer):
//            consumes.
// Not here:  ADPCM decoding (the mixer, audio_adpcm.h), Vorbis decoding itself (audio_vorbis.h), the decision to
//            stream a sound (bank: sounds/sounds.json / meta.json).
// Rules:     one SPSC ring per stream (`produced` - IO thread, `consumed` - mixer); blocks strictly in play order (on loop
//            the IO thread wraps to block 0 itself). Underrun: the mixer outputs silence and does not move.
//            The format is found by the file itself ("OggS" -> Vorbis, else WAV). A Vorbis stream needs a free decoder
//            (max_vorbis) and the mixer rate; otherwise it opens in state `error` (`error` names the reason).
// Tests:     tests/audio_stream_test.cpp.
//
// Slot life cycle (state):
//   free -> requested (game: open) -> ready / error (IO: header parsed, ring filled) -> closing (game) -> free (IO)
// The game closes a stream only after the mixer has reported its voice finished.
// Seek = close + open at another frame; the ring never changes under a reading mixer.

#include <stdint.h>
#include <atomic>

#include "audio.h"          // audio_file_io

enum audio_stream_state : uint32_t {
    audio_stream_free,
    audio_stream_requested,
    audio_stream_ready,
    audio_stream_error,
    audio_stream_closing,
};

struct audio_stream {
    // set by the game before `requested`
    char                    path[256];
    uint64_t                start_frame         = 0;
    bool                    loop                = false;
    double                  opened_us           = 0;        // profiling: open -> ready

    // set by the IO thread before `ready`, immutable afterwards
    void*                   file                = nullptr;
    uint64_t                data_offset         = 0;
    uint32_t                block_align         = 0;
    uint32_t                samples_per_block   = 0;
    uint32_t                block_count         = 0;
    uint32_t                sample_rate         = 0;
    uint64_t                frames              = 0;
    uint32_t                ring_blocks         = 0;
    uint8_t*                ring                = nullptr;   // ring_blocks * block_align
    bool                    pcm                 = false;     // ring blocks are mono int16 (Vorbis), not ADPCM
    uint32_t                decoder             = ~0u;       // Vorbis decoder slot (IO thread)
    const char*             error               = nullptr;   // state error: why (static string)

    // IO thread <-> mixer
    std::atomic<uint32_t>   state{audio_stream_free};
    std::atomic<uint32_t>   produced{0}, consumed{0};

    // IO thread (read by the debug view)
    uint32_t                next_block          = 0;
    std::atomic<uint64_t>   reads{0}, bytes_read{0};       // file reads (ADPCM: one per block; Vorbis: 4 KB chunks)

    // mixer private (read by stats)
    std::atomic<uint64_t>   underrun_frames{0}, underrun_events{0};
    std::atomic<uint32_t>   fill_min{0};                    // ring blocks ahead of the mixer, minimum since ready
    bool                    in_underrun         = false;
};

struct audio_stream_desc {
    uint32_t                max_streams         = 16;
    uint32_t                ring_blocks         = 8;
    uint32_t                max_block_align     = 1024;
    audio_file_io*          io                  = nullptr;   // nullptr -> stdio
    uint32_t                idle_sleep_us       = 1000;
    // Vorbis: decoders shared by all slots (one per open Vorbis stream), each with its own memory and int16 ring
    uint32_t                max_vorbis          = 4;
    uint32_t                vorbis_memory       = 256 * 1024;
    uint32_t                vorbis_ring_blocks  = 16;       // x 1024 frames ~ 340 ms at 48 kHz
    uint32_t                sample_rate         = 0;        // a Vorbis file must have it (0: not checked)
};

class audio_stream_system
{
public:
    bool                init(const audio_stream_desc& desc);
    void                shutdown();

    // game thread
    audio_stream*       open(const char* path, uint64_t start_frame, bool loop);   // nullptr if no free slot
    void                close(audio_stream* s);                                     // only after its voice finished
    uint32_t            state(const audio_stream* s) const  { return s->state.load(std::memory_order_acquire); }
    uint64_t            underruns() const;
    audio_stream*       slot(uint32_t i)                    { return i < m_desc.max_streams ? &m_streams[i] : nullptr; }
    const audio_stream* slot(uint32_t i) const              { return i < m_desc.max_streams ? &m_streams[i] : nullptr; }
    uint32_t            max_streams() const                 { return m_desc.max_streams; }
    uint32_t            open_count() const;                 // slots not free
    size_t              memory_bytes() const;               // slots + rings + Vorbis decoders (memory, rings)

    // profiling, written by the IO thread (totals; the caller makes windows of them).
    // blocks / bytes / read_us: file reads (ADPCM block, Vorbis 4 KB chunk); decoded / decode_us: Vorbis blocks of
    // 1024 frames, decode time without the reads in it.
    struct io_totals { uint64_t blocks, bytes, starts, decoded; double read_us_sum, read_us_max, start_us_sum, start_us_max, decode_us_sum, decode_us_max; };
    io_totals           totals() const;
    void                reset_max()                         { m_read_us_max.store(0); m_start_us_max.store(0); m_decode_us_max.store(0); }   // game thread: a new window

private:
    struct vorbis_slot;
    void                io_thread();
    bool                serve(audio_stream& s);             // returns true if it did any IO
    bool                open_vorbis(audio_stream& s);       // IO thread: decoder, header, seek to the start block
    void                fill_block(audio_stream& s, uint32_t slot);     // ADPCM: read; Vorbis: decode
    bool                wrap(audio_stream& s);              // the end of the file: loop -> block 0 (true), else false
    void                count_read(audio_stream& s, size_t bytes, double us);
    static size_t       vorbis_read(void* user, uint64_t offset, void* dst, size_t size);

    audio_stream_desc   m_desc;
    audio_file_io       m_io                = {};
    audio_stream*       m_streams           = nullptr;
    uint8_t*            m_ring_memory       = nullptr;
    void*               m_thread            = nullptr;      // std::thread*
    std::atomic<bool>   m_running{false};
    vorbis_slot*        m_vorbis            = nullptr;      // max_vorbis
    uint8_t*            m_vorbis_memory     = nullptr;      // decoder memory + int16 rings of all of them
    std::atomic<uint64_t> m_blocks{0}, m_bytes{0}, m_starts{0}, m_decoded{0};
    std::atomic<double> m_read_us_sum{0}, m_read_us_max{0}, m_start_us_sum{0}, m_start_us_max{0}, m_decode_us_sum{0}, m_decode_us_max{0};
    void                read_block(audio_stream& s, uint32_t slot);     // one ring block from the file, timed
};

#endif
