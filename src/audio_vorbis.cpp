// audio_vorbis.cpp - Vorbis decoder (summary in audio_vorbis.h). Also compiles stb_vorbis for the module.
// Parts: stb_vorbis build settings, open (source / memory), seek, read.

#include "audio_vorbis.h"
#include "audio_mix.h"

// stb_vorbis: no stdio, no push API (the pull API seeks), reads through IO callbacks, at most 2 channels
#define STB_VORBIS_NO_STDIO
#define STB_VORBIS_NO_PUSHDATA_API
#define STB_VORBIS_IO_CALLBACKS
#define STB_VORBIS_MAX_CHANNELS 2
#if defined(_MSC_VER)
    #pragma warning(push)
    #pragma warning(disable: 4244 4245 4456 4457 4701 4100 4505)
#elif defined(__GNUC__)
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wunused-function"
    #pragma GCC diagnostic ignored "-Wunused-value"
#endif
#include "stb/stb_vorbis.c"
#if defined(_MSC_VER)
    #pragma warning(pop)
#elif defined(__GNUC__)
    #pragma GCC diagnostic pop
#endif

namespace {

const char* open_error(int e)
{
    switch (e) {
    case VORBIS_outofmem:               return "the decoder needs more memory (audio_desc_t::vorbis_memory)";
    case VORBIS_too_many_channels:      return "more than 2 channels";
    case VORBIS_unexpected_eof:         return "file is empty or truncated";
    case VORBIS_missing_capture_pattern:
    case VORBIS_invalid_first_page:     return "not an Ogg Vorbis file";
    default:                            return "damaged or unsupported Ogg Vorbis file";
    }
}

} // namespace

bool audio_vorbis_decoder::open(const audio_vorbis_source& src, void* memory, size_t bytes, const char** error)
{
    close();
    stb_vorbis_io io{ src.user, src.read_at, src.length };
    stb_vorbis_alloc a{ (char*)memory, int(bytes) };
    int e = 0;
    m_v = stb_vorbis_open_io(&io, &e, &a);
    if (!m_v) { if (error) *error = open_error(e); return false; }
    return finish_open(error);
}

bool audio_vorbis_decoder::open_memory(const void* data, size_t size, void* memory, size_t bytes, const char** error)
{
    close();
    stb_vorbis_alloc a{ (char*)memory, int(bytes) };
    int e = 0;
    m_v = stb_vorbis_open_memory((const unsigned char*)data, int(size), &e, &a);
    if (!m_v) { if (error) *error = open_error(e); return false; }
    return finish_open(error);
}

bool audio_vorbis_decoder::finish_open(const char** error)
{
    stb_vorbis_info info = stb_vorbis_get_info(m_v);
    m_channels = uint32_t(info.channels);
    m_rate = info.sample_rate;
    m_memory_used = size_t(info.setup_memory_required) + info.temp_memory_required;
    m_frames = uint64_t(stb_vorbis_stream_length_in_samples64(m_v));     // reads the last page
    if (m_channels < 1 || m_channels > 2) { if (error) *error = "1 or 2 channels expected"; close(); return false; }
    if (!m_frames) { if (error) *error = "no length (last page not found)"; close(); return false; }
    return true;
}

void audio_vorbis_decoder::close()
{
    if (m_v) stb_vorbis_close(m_v);       // the memory block is the caller's: nothing is freed
    m_v = nullptr;
    m_channels = m_rate = 0;
    m_frames = 0;
    m_memory_used = 0;
}

bool audio_vorbis_decoder::seek(uint64_t frame)
{
    if (!m_v || frame >= m_frames) return false;
    return frame == 0 ? stb_vorbis_seek_start(m_v) != 0 : stb_vorbis_seek64(m_v, int64_t(frame)) != 0;
}

uint32_t audio_vorbis_decoder::read(int16_t* dst, uint32_t n)
{
    uint32_t done = 0;
    float* planes[2] = { m_tmp[0], m_tmp[1] };
    while (m_v && done < n) {
        int want = int(n - done < audio_vorbis::k_block_frames ? n - done : audio_vorbis::k_block_frames);
        int got = stb_vorbis_get_samples_float(m_v, int(m_channels), planes, want);
        if (got <= 0) break;
        audio_mix::mono_s16_store(dst + done, m_tmp[0], m_channels == 2 ? m_tmp[1] : nullptr, uint32_t(got));
        done += uint32_t(got);
    }
    return done;
}
