#ifndef __audio_vorbis_h__
#define __audio_vorbis_h__

// audio_vorbis.h - Ogg Vorbis decoder for streams (stb_vorbis, patched copy in external/stb)
//
// Role:      one Vorbis file -> mono int16 at the file's rate, in fixed memory (no malloc), with sample-exact seeking.
// Contents:
//   - audio_vorbis_source   where the bytes come from: read_at(offset) + length (the stream IO, or memory)
//   - audio_vorbis_decoder  open (memory / source) -> info (channels, rate, frames), seek, read (mono int16), close
// Thread:    whoever owns the decoder; the stream system uses it on the IO thread only. One decoder = one thread.
// Not here:  rings, file handles, the choice of a decoder (audio_stream.cpp); the decision to stream (bank).
// Rules:     1 or 2 channels (2 are mixed down: (L + R) / 2 - audio_mix::mono_s16_store); the stream system also
//            requires the mixer rate. Decoder memory (setup: codebooks, tables + work buffers) is the caller's block;
//            a file needing more fails to open with "needs more memory".
// Uses:      external/stb/stb_vorbis.c (SIMD + IO callbacks: see the note at its top), audio_mix.h (mono_s16_store).
// Tests:     tests/audio_vorbis_test.cpp (SIMD == scalar, source == memory, seek, memory limit, speed),
//            tests/audio_stream_test.cpp (Vorbis streams).

#include <stdint.h>
#include <stddef.h>

namespace audio_vorbis {
static const uint32_t k_block_frames = 1024;            // mono int16 block of a Vorbis stream (ring slot, envelope step)
static const size_t   k_default_memory = 256 * 1024;    // decoder memory; 190-215 KB for the files measured
}

struct audio_vorbis_source {
    void*       user        = nullptr;
    size_t      (*read_at)(void* user, uint64_t offset, void* dst, size_t size) = nullptr;   // bytes read, 0: end / error
    uint64_t    length      = 0;                        // file size in bytes
};

class audio_vorbis_decoder
{
public:
    audio_vorbis_decoder() = default;
    ~audio_vorbis_decoder() { close(); }
    audio_vorbis_decoder(const audio_vorbis_decoder&) = delete;
    audio_vorbis_decoder& operator=(const audio_vorbis_decoder&) = delete;

    // memory / bytes: the decoder's whole working memory (kept until close). error: a static string on failure.
    bool        open(const audio_vorbis_source& src, void* memory, size_t bytes, const char** error);
    bool        open_memory(const void* data, size_t size, void* memory, size_t bytes, const char** error);
    void        close();
    bool        is_open() const                 { return m_v != nullptr; }

    uint32_t    channels() const                { return m_channels; }
    uint32_t    sample_rate() const             { return m_rate; }
    uint64_t    frames() const                  { return m_frames; }
    size_t      memory_used() const             { return m_memory_used; }   // setup + work memory the file needs

    bool        seek(uint64_t frame);           // sample exact; the next read starts at `frame`
    uint32_t    read(int16_t* dst, uint32_t n); // mono int16; fewer than n only at the end of the file

private:
    bool        finish_open(const char** error);

    struct stb_vorbis* m_v              = nullptr;
    uint32_t    m_channels              = 0;
    uint32_t    m_rate                  = 0;
    uint64_t    m_frames                = 0;
    size_t      m_memory_used           = 0;
    float       m_tmp[2][audio_vorbis::k_block_frames];   // planar floats from stb, then mixed down
};

#endif
