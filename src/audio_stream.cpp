// audio_stream.cpp - streaming implementation (summary in audio_stream.h).
// Parts: stdio IO, init / shutdown, open / close (game thread), totals, IO thread: block fill (ADPCM read, Vorbis decode),
// Vorbis open, serve (open + prefill, refill, close), io_thread.

#include "audio_stream.h"
#include "audio_adpcm.h"
#include "audio_vorbis.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <new>
#include <thread>
#include <chrono>

namespace {

double now_us() { return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

void*  stdio_open(void*, const char* path) { return fopen(path, "rb"); }
void   stdio_close(void*, void* f) { fclose((FILE*)f); }
size_t stdio_read_at(void*, void* f, uint64_t off, void* dst, size_t size)
{
#if defined(_WIN32)
    if (_fseeki64((FILE*)f, int64_t(off), SEEK_SET)) return 0;
#else
    if (fseeko((FILE*)f, off_t(off), SEEK_SET)) return 0;
#endif
    return fread(dst, 1, size, (FILE*)f);
}
uint64_t stdio_size(void*, void* f)
{
#if defined(_WIN32)
    if (_fseeki64((FILE*)f, 0, SEEK_END)) return 0;
    return uint64_t(_ftelli64((FILE*)f));
#else
    if (fseeko((FILE*)f, 0, SEEK_END)) return 0;
    return uint64_t(ftello((FILE*)f));
#endif
}

void atomic_max(std::atomic<double>& m, double v)
{
    for (double cur = m.load(std::memory_order_relaxed); v > cur && !m.compare_exchange_weak(cur, v);) {}   // the game thread resets it
}

// IO thread, single writer: plain load + store
void atomic_add(std::atomic<double>& a, double v)     { a.store(a.load(std::memory_order_relaxed) + v, std::memory_order_relaxed); }
void atomic_add(std::atomic<uint64_t>& a, uint64_t v) { a.store(a.load(std::memory_order_relaxed) + v, std::memory_order_relaxed); }

const uint32_t k_pcm_block_bytes = audio_vorbis::k_block_frames * sizeof(int16_t);

} // namespace

// one Vorbis decoder: its memory, its int16 ring, the stream it serves (IO thread only)
struct audio_stream_system::vorbis_slot {
    audio_vorbis_decoder    dec;
    uint8_t*                memory      = nullptr;
    uint8_t*                ring        = nullptr;
    audio_stream_system*    sys         = nullptr;
    audio_stream*           stream      = nullptr;      // nullptr: free
    double                  read_us     = 0;            // file read time inside the current decode
};

bool audio_stream_system::init(const audio_stream_desc& desc)
{
    m_desc = desc;
    if (desc.io) m_io = *desc.io;
    else m_io = audio_file_io{ nullptr, stdio_open, stdio_read_at, stdio_close, stdio_size };

    m_streams = (audio_stream*)calloc(m_desc.max_streams, sizeof(audio_stream));
    m_ring_memory = (uint8_t*)calloc(size_t(m_desc.max_streams) * m_desc.ring_blocks, m_desc.max_block_align);
    const size_t per_vorbis = size_t(m_desc.vorbis_memory) + size_t(m_desc.vorbis_ring_blocks) * k_pcm_block_bytes;
    if (m_desc.max_vorbis) {
        m_vorbis = new (std::nothrow) vorbis_slot[m_desc.max_vorbis];
        m_vorbis_memory = (uint8_t*)calloc(m_desc.max_vorbis, per_vorbis);
    }
    if (!m_streams || !m_ring_memory || (m_desc.max_vorbis && (!m_vorbis || !m_vorbis_memory))) { shutdown(); return false; }
    for (uint32_t i = 0; i < m_desc.max_streams; ++i) {
        new (&m_streams[i]) audio_stream();
        m_streams[i].ring = m_ring_memory + size_t(i) * m_desc.ring_blocks * m_desc.max_block_align;
    }
    for (uint32_t i = 0; i < m_desc.max_vorbis; ++i) {
        m_vorbis[i].memory = m_vorbis_memory + size_t(i) * per_vorbis;
        m_vorbis[i].ring = m_vorbis[i].memory + m_desc.vorbis_memory;
        m_vorbis[i].sys = this;
    }
    m_running = true;
    m_thread = new std::thread([this] { io_thread(); });
    return true;
}

void audio_stream_system::shutdown()
{
    m_running = false;
    if (m_thread) { ((std::thread*)m_thread)->join(); delete (std::thread*)m_thread; m_thread = nullptr; }
    if (m_streams) {
        for (uint32_t i = 0; i < m_desc.max_streams; ++i) {
            if (m_streams[i].file) m_io.close(m_io.user, m_streams[i].file);
            m_streams[i].~audio_stream();
        }
    }
    delete[] m_vorbis; m_vorbis = nullptr;          // decoders close themselves; their memory is the block below
    free(m_vorbis_memory); m_vorbis_memory = nullptr;
    free(m_streams); m_streams = nullptr;
    free(m_ring_memory); m_ring_memory = nullptr;
}

audio_stream* audio_stream_system::open(const char* path, uint64_t start_frame, bool loop)
{
    for (uint32_t i = 0; i < m_desc.max_streams; ++i) {
        audio_stream& s = m_streams[i];
        if (s.state.load(std::memory_order_acquire) != audio_stream_free) continue;
        size_t len = strlen(path);
        if (len >= sizeof(s.path)) return nullptr;
        memcpy(s.path, path, len + 1);
        s.start_frame = start_frame;
        s.loop = loop;
        s.opened_us = now_us();
        s.state.store(audio_stream_requested, std::memory_order_release);
        return &s;
    }
    return nullptr;
}

void audio_stream_system::close(audio_stream* s)
{
    if (!s) return;
    uint32_t st = s->state.load(std::memory_order_acquire);
    if (st == audio_stream_ready || st == audio_stream_error || st == audio_stream_requested)
        s->state.store(audio_stream_closing, std::memory_order_release);
}

uint32_t audio_stream_system::open_count() const
{
    uint32_t n = 0;
    for (uint32_t i = 0; i < m_desc.max_streams; ++i) n += m_streams[i].state.load(std::memory_order_acquire) != audio_stream_free;
    return n;
}

size_t audio_stream_system::memory_bytes() const
{
    return size_t(m_desc.max_streams) * (sizeof(audio_stream) + size_t(m_desc.ring_blocks) * m_desc.max_block_align)
         + size_t(m_desc.max_vorbis) * (sizeof(vorbis_slot) + m_desc.vorbis_memory + size_t(m_desc.vorbis_ring_blocks) * k_pcm_block_bytes);
}

audio_stream_system::io_totals audio_stream_system::totals() const
{
    io_totals t;
    t.blocks = m_blocks.load(std::memory_order_relaxed); t.bytes = m_bytes.load(std::memory_order_relaxed); t.starts = m_starts.load(std::memory_order_relaxed);
    t.decoded = m_decoded.load(std::memory_order_relaxed);
    t.read_us_sum = m_read_us_sum.load(std::memory_order_relaxed); t.read_us_max = m_read_us_max.load(std::memory_order_relaxed);
    t.start_us_sum = m_start_us_sum.load(std::memory_order_relaxed); t.start_us_max = m_start_us_max.load(std::memory_order_relaxed);
    t.decode_us_sum = m_decode_us_sum.load(std::memory_order_relaxed); t.decode_us_max = m_decode_us_max.load(std::memory_order_relaxed);
    return t;
}

uint64_t audio_stream_system::underruns() const
{
    uint64_t n = 0;
    for (uint32_t i = 0; i < m_desc.max_streams; ++i) n += m_streams[i].underrun_events.load(std::memory_order_relaxed);
    return n;
}

// ---------------------------------------------------------------------------
// IO thread: blocks
// ---------------------------------------------------------------------------
void audio_stream_system::count_read(audio_stream& s, size_t bytes, double us)
{
    s.reads.fetch_add(1, std::memory_order_relaxed);
    s.bytes_read.fetch_add(bytes, std::memory_order_relaxed);
    atomic_add(m_blocks, 1);
    atomic_add(m_bytes, bytes);
    atomic_add(m_read_us_sum, us);
    atomic_max(m_read_us_max, us);
}

// ADPCM: one block into ring slot `slot`
void audio_stream_system::read_block(audio_stream& s, uint32_t slot)
{
    const double t0 = now_us();
    m_io.read_at(m_io.user, s.file, s.data_offset + uint64_t(s.next_block) * s.block_align, s.ring + size_t(slot) * s.block_align, s.block_align);
    count_read(s, s.block_align, now_us() - t0);
    ++s.next_block;
}

// the decoder's file reads (inside stb_vorbis): counted as reads, their time kept out of the decode time
size_t audio_stream_system::vorbis_read(void* user, uint64_t offset, void* dst, size_t size)
{
    vorbis_slot& v = *(vorbis_slot*)user;
    audio_stream_system& sys = *v.sys;
    const double t0 = now_us();
    size_t got = sys.m_io.read_at(sys.m_io.user, v.stream->file, offset, dst, size);
    const double us = now_us() - t0;
    v.read_us += us;
    sys.count_read(*v.stream, got, us);
    return got;
}

void audio_stream_system::fill_block(audio_stream& s, uint32_t slot)
{
    if (!s.pcm) { read_block(s, slot); return; }
    vorbis_slot& v = m_vorbis[s.decoder];
    int16_t* dst = (int16_t*)(s.ring + size_t(slot) * s.block_align);
    const uint64_t first = uint64_t(s.next_block) * s.samples_per_block;
    const uint32_t want = uint32_t(s.frames - first < s.samples_per_block ? s.frames - first : s.samples_per_block);
    const double t0 = now_us();
    v.read_us = 0;
    uint32_t got = v.dec.read(dst, want);
    const double us = now_us() - t0 - v.read_us;
    if (got < s.samples_per_block) memset(dst + got, 0, (s.samples_per_block - got) * sizeof(int16_t));    // the last block / a short file
    atomic_add(m_decoded, 1);
    atomic_add(m_decode_us_sum, us);
    atomic_max(m_decode_us_max, us);
    ++s.next_block;
}

bool audio_stream_system::wrap(audio_stream& s)
{
    if (!s.loop) return false;
    s.next_block = 0;
    if (s.pcm) m_vorbis[s.decoder].dec.seek(0);
    return true;
}

// ---------------------------------------------------------------------------
// IO thread: Vorbis open
// ---------------------------------------------------------------------------
bool audio_stream_system::open_vorbis(audio_stream& s)
{
    vorbis_slot* v = nullptr;
    for (uint32_t i = 0; i < m_desc.max_vorbis && !v; ++i) if (!m_vorbis[i].stream) { v = &m_vorbis[i]; s.decoder = i; }
    if (!v) { s.error = "no free Vorbis decoder (audio_desc_t::max_vorbis_streams)"; return false; }
    uint64_t size = m_io.size ? m_io.size(m_io.user, s.file) : 0;
    if (!size) {                                    // no size callback: the last readable byte by bisection
        uint8_t b;
        uint64_t lo = 0, hi = uint64_t(1) << 40;
        while (hi - lo > 1) { uint64_t mid = lo + (hi - lo) / 2; (m_io.read_at(m_io.user, s.file, mid, &b, 1) == 1 ? lo : hi) = mid; }
        size = lo + 1;
    }
    v->stream = &s;
    audio_vorbis_source src;
    src.user = v; src.read_at = vorbis_read; src.length = size;
    const char* err = nullptr;
    if (!v->dec.open(src, v->memory, m_desc.vorbis_memory, &err)) { s.error = err; v->stream = nullptr; return false; }
    if (m_desc.sample_rate && v->dec.sample_rate() != m_desc.sample_rate) {
        s.error = "Vorbis file is not at the mixer rate";
        v->dec.close(); v->stream = nullptr; return false;
    }
    s.pcm               = true;
    s.block_align       = k_pcm_block_bytes;
    s.samples_per_block = audio_vorbis::k_block_frames;
    s.frames            = v->dec.frames();
    s.block_count       = uint32_t((s.frames + s.samples_per_block - 1) / s.samples_per_block);
    s.sample_rate       = v->dec.sample_rate();
    s.ring              = v->ring;
    s.ring_blocks       = m_desc.vorbis_ring_blocks;
    s.data_offset       = 0;
    const uint32_t start_block = uint32_t(s.start_frame / s.samples_per_block);
    if (start_block >= s.block_count || (start_block && !v->dec.seek(uint64_t(start_block) * s.samples_per_block))) {
        s.error = "start position past the end";
        v->dec.close(); v->stream = nullptr; return false;
    }
    s.next_block = start_block;
    return true;
}

// ---------------------------------------------------------------------------
// IO thread: one pass over a stream
// ---------------------------------------------------------------------------
bool audio_stream_system::serve(audio_stream& s)
{
    switch (s.state.load(std::memory_order_acquire)) {
    case audio_stream_requested: {
        s.error = nullptr;
        s.pcm = false;
        s.decoder = ~0u;
        s.ring = m_ring_memory + size_t(&s - m_streams) * m_desc.ring_blocks * m_desc.max_block_align;    // a Vorbis stream swaps it
        s.reads.store(0, std::memory_order_relaxed);
        s.bytes_read.store(0, std::memory_order_relaxed);
        // open + header (the data chunk may sit behind other chunks: read up to 4 KB)
        s.file = m_io.open(m_io.user, s.path);
        uint8_t head[4096];
        size_t got = s.file ? m_io.read_at(m_io.user, s.file, 0, head, sizeof(head)) : 0;
        bool ok;
        if (got >= 4 && !memcmp(head, "OggS", 4)) {
            ok = open_vorbis(s);
        } else {
            audio_adpcm::clip c;
            ok = got > 0 && audio_adpcm::parse_wav(head, got, c) && c.block_align <= m_desc.max_block_align && c.samples_per_block;
            uint32_t start_block = ok ? uint32_t(s.start_frame / c.samples_per_block) : 0;
            if (ok && start_block >= c.block_count) { ok = false; s.error = "start position past the end"; }
            else if (!ok) s.error = s.file ? "not an IMA ADPCM WAV or Ogg Vorbis file" : "cannot open the file";
            if (ok) {
                s.data_offset       = uint64_t(c.blocks - head);
                s.block_align       = c.block_align;
                s.samples_per_block = c.samples_per_block;
                s.block_count       = c.block_count;
                s.sample_rate       = c.sample_rate;
                s.frames            = c.frames;
                s.ring_blocks       = m_desc.ring_blocks;
                s.next_block        = start_block;
            }
        }
        if (!ok) {
            if (s.file) { m_io.close(m_io.user, s.file); s.file = nullptr; }
            s.state.store(audio_stream_error, std::memory_order_release);
            return true;
        }
        s.produced.store(0, std::memory_order_relaxed);
        s.consumed.store(0, std::memory_order_relaxed);
        s.underrun_frames.store(0, std::memory_order_relaxed);
        s.underrun_events.store(0, std::memory_order_relaxed);
        s.in_underrun = false;
        // prefill the whole ring before the stream is reported ready (= pending start)
        while (s.produced.load(std::memory_order_relaxed) < s.ring_blocks) {
            if (s.next_block >= s.block_count && !wrap(s)) break;
            fill_block(s, s.produced.load(std::memory_order_relaxed) % s.ring_blocks);
            s.produced.fetch_add(1, std::memory_order_release);
        }
        s.fill_min.store(s.produced.load(std::memory_order_relaxed), std::memory_order_relaxed);
        {
            const double us = now_us() - s.opened_us;
            atomic_add(m_starts, 1);
            atomic_add(m_start_us_sum, us);
            atomic_max(m_start_us_max, us);
        }
        s.state.store(audio_stream_ready, std::memory_order_release);
        return true;
    }
    case audio_stream_ready: {
        bool any = false;
        while (s.produced.load(std::memory_order_relaxed) - s.consumed.load(std::memory_order_acquire) < s.ring_blocks) {
            if (s.next_block >= s.block_count && !wrap(s)) break;
            if (s.state.load(std::memory_order_acquire) != audio_stream_ready) break;
            fill_block(s, s.produced.load(std::memory_order_relaxed) % s.ring_blocks);
            any = true;
            s.produced.fetch_add(1, std::memory_order_release);
        }
        return any;
    }
    case audio_stream_closing:
        if (s.pcm && s.decoder < m_desc.max_vorbis) { m_vorbis[s.decoder].dec.close(); m_vorbis[s.decoder].stream = nullptr; }
        if (s.file) { m_io.close(m_io.user, s.file); s.file = nullptr; }
        s.pcm = false;
        s.decoder = ~0u;
        s.state.store(audio_stream_free, std::memory_order_release);
        return true;
    default:
        return false;
    }
}

void audio_stream_system::io_thread()
{
    while (m_running.load(std::memory_order_acquire)) {
        bool any = false;
        for (uint32_t i = 0; i < m_desc.max_streams; ++i) any |= serve(m_streams[i]);
        if (!any) std::this_thread::sleep_for(std::chrono::microseconds(m_desc.idle_sleep_us));
    }
}
