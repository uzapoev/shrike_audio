// audio_mixer.cpp - mixer implementation (summary in audio_mixer.h).
// Parts: init / device, stats, commands (apply), voice rendering (fetch_block, render_voice), period (render_period), render.

#include "audio_mixer.h"
#include "audio_mix.h"
#include "audio_stream.h"

#include <stdlib.h>
#include <new>
#include <string.h>
#include <chrono>

// miniaudio is used only for the playback device.
#define MA_NO_DECODING
#define MA_NO_ENCODING
#define MA_NO_GENERATION
#define MA_NO_RESOURCE_MANAGER
#define MA_NO_NODE_GRAPH
#define MA_NO_ENGINE
#define MINIAUDIO_IMPLEMENTATION
#include "miniaudio/miniaudio.h"

namespace {

double now_us() { return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

size_t align32(size_t x) { return (x + 31) & ~size_t(31); }

void device_callback(ma_device* dev, void* out, const void*, ma_uint32 frames)
{
    ((audio_mixer*)dev->pUserData)->render((float*)out, frames);
}

} // namespace

// ---------------------------------------------------------------------------
// init / shutdown
// ---------------------------------------------------------------------------
bool audio_mixer::init(const audio_mixer_desc& desc)
{
    m_desc = desc;
    if (!m_desc.plugin.process) m_desc.plugin = audio_plugin_builtin_pan();
    const uint32_t n = m_desc.period_frames;
    if (n == 0 || m_desc.max_buses == 0) return false;

    // one block of memory, every buffer 32-byte aligned
    size_t voices    = align32(sizeof(voice) * m_desc.max_voices);
    size_t caches    = align32(sizeof(int16_t) * (m_desc.max_samples_per_block + 1)) * m_desc.max_voices;
    size_t instances = align32(sizeof(instance) * m_desc.max_instances);
    size_t buses     = align32(sizeof(bus) * m_desc.max_buses);
    size_t bus_bufs  = align32(sizeof(float) * n) * 2 * m_desc.max_buses;
    size_t mono      = align32(sizeof(float) * n) * 5;      // 4 instance buffers + a silent dummy lane
    size_t fifo      = align32(sizeof(float) * n * 2);
    size_t bus_in    = align32(sizeof(float) * n) * m_desc.max_buses;
    size_t zero      = align32(sizeof(float) * n);
    size_t reverbs   = align32(sizeof(audio_reverb) * m_desc.max_reverbs) + align32(sizeof(bool) * m_desc.max_reverbs)
                     + align32(sizeof(float) * audio_reverb::memory_floats()) * m_desc.max_reverbs;
    size_t total     = voices + caches + instances + buses + bus_bufs + mono + fifo + bus_in + zero + reverbs + 32;
    m_memory_bytes = total;
    m_reverb_bytes = reverbs;
    m_memory = calloc(1, total);
    if (!m_memory) return false;
    uint8_t* p = (uint8_t*)(((uintptr_t)m_memory + 31) & ~uintptr_t(31));

    m_voices = (voice*)p; p += voices;
    for (uint32_t i = 0; i < m_desc.max_voices; ++i) {
        new (&m_voices[i]) voice();
        m_voices[i].cache = (int16_t*)p;
        p += align32(sizeof(int16_t) * (m_desc.max_samples_per_block + 1));
    }
    m_instances = (instance*)p; p += instances;
    for (uint32_t i = 0; i < m_desc.max_instances; ++i) new (&m_instances[i]) instance();
    m_buses = (bus*)p; p += buses;
    for (uint32_t i = 0; i < m_desc.max_buses; ++i) {
        new (&m_buses[i]) bus();
        m_buses[i].l = (float*)p; p += align32(sizeof(float) * n);
        m_buses[i].r = (float*)p; p += align32(sizeof(float) * n);
        m_buses[i].in = (float*)p; p += align32(sizeof(float) * n);
    }
    m_buses[0].used = true;
    for (int k = 0; k < 4; ++k) { m_mono[k] = (float*)p; p += align32(sizeof(float) * n); }
    m_silent = (float*)p; p += align32(sizeof(float) * n);
    m_zero = (float*)p; p += zero;
    m_reverbs = (audio_reverb*)p; p += align32(sizeof(audio_reverb) * m_desc.max_reverbs);
    m_reverb_used = (bool*)p; p += align32(sizeof(bool) * m_desc.max_reverbs);
    for (uint32_t i = 0; i < m_desc.max_reverbs; ++i) {
        new (&m_reverbs[i]) audio_reverb();
        m_reverbs[i].attach((float*)p);
        p += align32(sizeof(float) * audio_reverb::memory_floats());
    }
    m_fifo = (float*)p; p += fifo;

    return m_desc.plugin.init ? m_desc.plugin.init(m_desc.plugin.user, m_desc.sample_rate, n, m_desc.max_instances) : true;
}

void audio_mixer::shutdown()
{
    stop_device();
    if (m_desc.plugin.shutdown) m_desc.plugin.shutdown(m_desc.plugin.user);
    free(m_memory);
    m_memory = nullptr;
    m_voices = nullptr; m_instances = nullptr; m_buses = nullptr;
}

bool audio_mixer::start_device()
{
    ma_device* dev = (ma_device*)calloc(1, sizeof(ma_device));
    ma_device_config dc = ma_device_config_init(ma_device_type_playback);
    dc.playback.format   = ma_format_f32;
    dc.playback.channels = 2;
    dc.sampleRate        = m_desc.sample_rate;
    dc.periodSizeInFrames = m_desc.period_frames;
    dc.dataCallback      = device_callback;
    dc.pUserData         = this;
    if (ma_device_init(nullptr, &dc, dev) != MA_SUCCESS) { free(dev); return false; }
    if (ma_device_start(dev) != MA_SUCCESS) { ma_device_uninit(dev); free(dev); return false; }
    m_device = dev;
    return true;
}

void audio_mixer::stop_device()
{
    if (!m_device) return;
    ma_device_uninit((ma_device*)m_device);
    free(m_device);
    m_device = nullptr;
}

audio_mixer_stats audio_mixer::stats()
{
    audio_mixer_stats s = {};
    s.periods            = m_periods.load();
    s.commands           = m_commands.load();
    s.play_on_busy_voice = m_play_on_busy.load();
    s.notes_dropped      = m_notes_dropped.load();
    s.stream_underruns   = m_underruns.load();
    uint64_t cnt         = m_period_count_window.exchange(0);
    s.period_us_max      = m_period_us_max.exchange(0);
    double sum           = m_period_us_sum.exchange(0);
    s.period_us_avg      = cnt ? sum / double(cnt) : 0.0;
    s.budget_us          = m_desc.period_frames * 1e6 / m_desc.sample_rate;
    return s;
}

// ---------------------------------------------------------------------------
// commands (audio thread)
// ---------------------------------------------------------------------------
void audio_mixer::unlink(uint32_t index)
{
    voice& v = m_voices[index];
    if (v.instance >= m_desc.max_instances) return;
    uint32_t* link = &m_instances[v.instance].head;
    while (*link != ~0u && *link != index) link = &m_voices[*link].next;
    if (*link == index) *link = v.next;
    v.next = ~0u;
    v.instance = ~0u;
}

void audio_mixer::finish(uint32_t index, audio_mixer_note_reason reason)
{
    voice& v = m_voices[index];
    if (!v.active) return;
    v.active = false;
    unlink(index);
    if (!m_notes.push(audio_mixer_note{ index, reason })) m_notes_dropped.fetch_add(1, std::memory_order_relaxed);
}

float audio_mixer::fade_at(const voice& v, uint32_t pos) const
{
    if (pos >= v.fade_len) return v.fade_to;
    return v.fade_from + (v.fade_to - v.fade_from) * (float(pos) / float(v.fade_len));
}

void audio_mixer::apply(const audio_mixer_cmd& c)
{
    switch (c.type) {
    case audio_mixer_cmd_voice_play: {
        if (c.target >= m_desc.max_voices || c.play.instance >= m_desc.max_instances || (!c.play.clip == !c.play.stream)) return;
        voice& v = m_voices[c.target];
        if (v.active) { m_play_on_busy.fetch_add(1, std::memory_order_relaxed); unlink(c.target); }
        uint64_t frames  = c.play.clip ? c.play.clip->frames : c.play.stream->frames;
        uint32_t spb     = c.play.clip ? c.play.clip->samples_per_block : c.play.stream->samples_per_block;
        if (spb == 0 || spb > m_desc.max_samples_per_block) return;
        if (c.play.stream && c.play.stream->state.load(std::memory_order_acquire) != audio_stream_ready) return;
        v.clip         = c.play.clip;
        v.stream       = c.play.stream;
        v.frames       = frames;
        v.samples_per_block = spb;
        v.cached_block = ~0u;
        // a stream's first ring block is the block of the frame it was opened at; a later offset skips ring blocks
        // (the instance waited for the ring to fill and catches up with the time it lost)
        uint64_t offset = c.play.offset;
        if (c.play.stream && offset < c.play.stream->start_frame) offset = c.play.stream->start_frame;
        v.cursor       = c.play.loop && frames ? offset % frames : offset;
        v.frac         = 0;
        v.step         = c.play.step ? c.play.step : 65536;
        v.delay        = c.play.delay;
        v.gain         = v.gain_target = c.play.gain;
        v.fade_from    = c.play.fade_in ? 0.0f : 1.0f;
        v.fade_to      = 1.0f;
        v.fade_len     = c.play.fade_in;
        v.fade_pos     = 0;
        v.stop_after_fade = false;
        v.loop         = c.play.loop;
        v.active       = true;
        v.instance     = c.play.instance;
        v.next         = m_instances[v.instance].head;
        m_instances[v.instance].head = c.target;
        break;
    }
    case audio_mixer_cmd_voice_stop: {
        if (c.target >= m_desc.max_voices) return;
        voice& v = m_voices[c.target];
        if (!v.active) return;
        if (c.stop.fade == 0 || v.delay > 0) { finish(c.target, audio_mixer_note_stopped); return; }   // not started yet: at once
        v.fade_from = fade_at(v, v.fade_pos);
        v.fade_to = 0.0f;
        v.fade_len = c.stop.fade;
        v.fade_pos = 0;
        v.stop_after_fade = true;
        break;
    }
    case audio_mixer_cmd_voice_gain:
        if (c.target < m_desc.max_voices) m_voices[c.target].gain_target = c.gain.gain;
        break;
    case audio_mixer_cmd_voice_pitch:
        if (c.target < m_desc.max_voices) m_voices[c.target].step = c.pitch.step ? c.pitch.step : 65536;
        break;
    case audio_mixer_cmd_instance_start: {
        if (c.target >= m_desc.max_instances) return;
        instance& in = m_instances[c.target];
        in.bus = c.instance.bus < m_desc.max_buses ? c.instance.bus : 0;
        in.spatial = audio_spatial_input{ { 0, 0, 0 }, 0, 1.0f, false };
        in.filtered = false; in.lowpass_hz = in.highpass_hz = 0.0f;                 // a reused slot: no filter, no history
        in.fc[0] = in.fc[1] = audio_mix::biquad(); in.fs[0] = in.fs[1] = audio_mix::biquad_state();
        for (int k = 0; k < 2; ++k) { in.send_bus[k] = ~0u; in.send_gain[k] = in.send_target[k] = 0.0f; }
        if (m_desc.plugin.source_reset) m_desc.plugin.source_reset(m_desc.plugin.user, c.target);
        break;
    }
    case audio_mixer_cmd_instance_spatial:
        if (c.target < m_desc.max_instances) m_instances[c.target].spatial = c.spatial;
        break;
    case audio_mixer_cmd_instance_send: {
        if (c.target >= m_desc.max_instances || c.send.slot >= 2) return;
        instance& in = m_instances[c.target];
        uint32_t bus = c.send.bus < m_desc.max_buses && m_buses[c.send.bus].effect ? c.send.bus : ~0u;
        if (bus != in.send_bus[c.send.slot]) in.send_gain[c.send.slot] = 0.0f;    // a new target ramps up from 0
        in.send_bus[c.send.slot] = bus;
        in.send_target[c.send.slot] = bus != ~0u ? c.send.gain : 0.0f;
        break;
    }
    case audio_mixer_cmd_bus_reverb: {
        if (c.target >= m_desc.max_buses) return;
        bus& b = m_buses[c.target];
        audio_reverb_preset p;
        p.decay = c.reverb.decay; p.predelay = c.reverb.predelay; p.damping = c.reverb.damping; p.size = c.reverb.size; p.level = c.reverb.level;
        b.effect = true;
        b.used = true;
        // the current FDN rings out with its own preset (line lengths can't change while it sounds); a fresh one takes the sends
        if (b.reverb >= 0) {
            if (b.tail >= 0) m_reverb_used[b.tail] = false;          // a third change before the tail died: drop the oldest
            b.tail = b.reverb;
            b.tail_left = uint32_t(b.decay * 1.2f * float(m_desc.sample_rate));
            b.reverb = -1;
        }
        int r = take_reverb();
        if (r < 0 && b.tail >= 0) { r = b.tail; b.tail = -1; }          // pool exhausted: reuse the tail (it is cut)
        if (r < 0) return;
        m_reverbs[r].configure(p, float(m_desc.sample_rate));
        b.reverb = r;
        b.decay = p.decay;
        break;
    }
    case audio_mixer_cmd_instance_filter: {
        if (c.target >= m_desc.max_instances) return;
        instance& in = m_instances[c.target];
        const float rate = float(m_desc.sample_rate);
        bool lp = c.filter.lowpass_hz > 0.0f && c.filter.lowpass_hz < 20000.0f, hp = c.filter.highpass_hz > 0.0f;
        bool was_lp = in.lowpass_hz > 0.0f, was_hp = in.highpass_hz > 0.0f;
        // a stage that turns on starts from rest; one that stays on keeps its state (the cutoff glides, no click)
        if (lp != was_lp) in.fs[0] = audio_mix::biquad_state();
        if (hp != was_hp) in.fs[1] = audio_mix::biquad_state();
        in.fc[0] = lp ? audio_mix::biquad_lowpass(c.filter.lowpass_hz, rate) : audio_mix::biquad();
        in.fc[1] = hp ? audio_mix::biquad_highpass(c.filter.highpass_hz, rate) : audio_mix::biquad();
        in.lowpass_hz = lp ? c.filter.lowpass_hz : 0.0f;
        in.highpass_hz = hp ? c.filter.highpass_hz : 0.0f;
        in.filtered = lp || hp;
        break;
    }
    case audio_mixer_cmd_bus_config: {
        if (c.target == 0 || c.target >= m_desc.max_buses || c.bus.parent >= c.target) {
            if (c.target == 0) { m_buses[0].volume = m_buses[0].volume_target = c.bus.volume; m_buses[0].mute = c.bus.mute; m_buses[0].pause = c.bus.pause; }
            return;
        }
        bus& b = m_buses[c.target];
        b.parent = c.bus.parent;
        b.volume = b.volume_target = c.bus.volume;
        b.mute = c.bus.mute;
        b.pause = c.bus.pause;
        b.used = true;
        break;
    }
    case audio_mixer_cmd_bus_volume:
        if (c.target < m_desc.max_buses) m_buses[c.target].volume_target = c.bus_volume.volume;
        break;
    }
}

// ---------------------------------------------------------------------------
// rendering (audio thread)
// ---------------------------------------------------------------------------

// Renders one period of a voice into dst (accumulating). The period is cut into segments at block end, clip end and
// fade end; each segment is one kernel call with a linear gain ramp (gain ramp x fade ramp, linear between the
// segment ends). Returns true if the voice finished.
// Decodes block b into the voice cache; cache[spb] = first sample of the following block (for interpolation).
// Streams deliver blocks strictly in play order, so b is always the next ring block.
bool audio_mixer::fetch_block(voice& v, uint32_t b)
{
    const double t0 = m_prof ? now_us() : 0.0;
    struct decode_time {                                    // every return path adds the decode time
        audio_mixer& m; double t0;
        ~decode_time() { if (m.m_prof) m.m_stage[audio_stage_decode] += now_us() - t0; }
    } dt{ *this, t0 };
    const uint32_t spb = v.samples_per_block;
    if (v.clip) {
        const audio_adpcm::clip& c = *v.clip;
        audio_adpcm::decode_block(c.blocks + size_t(b) * c.block_align, spb, v.cache);
        uint32_t nb = b + 1 < c.block_count ? b + 1 : (v.loop ? 0 : ~0u);
        v.cache[spb] = nb != ~0u ? audio_adpcm::block_first_sample(c.blocks + size_t(nb) * c.block_align) : 0;
    } else {
        audio_stream& s = *v.stream;
        uint32_t cns = s.consumed.load(std::memory_order_relaxed);
        uint32_t prd = s.produced.load(std::memory_order_acquire);
        // block id of the next ring slot; skip slots up to b (a play offset past the stream's start frame)
        const uint32_t start_block = uint32_t(s.start_frame / spb);
        for (;;) {
            uint32_t id = s.loop ? (start_block + cns) % s.block_count : start_block + cns;
            uint32_t ahead = s.loop ? (b + s.block_count - id) % s.block_count : (b > id ? b - id : 0);
            if (ahead == 0) break;
            if (prd == cns) { s.consumed.store(cns, std::memory_order_release); return false; }   // wait for more blocks
            ++cns;
        }
        s.consumed.store(cns, std::memory_order_release);
        if (prd == cns) return false;
        if (prd - cns < s.fill_min.load(std::memory_order_relaxed)) s.fill_min.store(prd - cns, std::memory_order_relaxed);
        const uint8_t* blk = s.ring + size_t(cns % s.ring_blocks) * s.block_align;
        // the following block is still owned by the mixer side (not consumed), the IO thread does not touch it
        const uint8_t* nxt = prd > cns + 1 ? s.ring + size_t((cns + 1) % s.ring_blocks) * s.block_align : nullptr;
        if (s.pcm) {                                        // Vorbis: decoded by the IO thread, mono int16
            memcpy(v.cache, blk, spb * sizeof(int16_t));
            v.cache[spb] = nxt ? ((const int16_t*)nxt)[0] : 0;
        } else {
            audio_adpcm::decode_block(blk, spb, v.cache);
            v.cache[spb] = nxt ? audio_adpcm::block_first_sample(nxt) : 0;
        }
        s.consumed.store(cns + 1, std::memory_order_release);
    }
    v.cached_block = b;
    return true;
}

bool audio_mixer::render_voice(voice& v, float* dst, bool& produced)
{
    const uint32_t n = m_desc.period_frames;
    if (v.delay >= n) { v.delay -= n; return false; }
    uint32_t i = v.delay;
    v.delay = 0;

    const uint64_t frames = v.frames;
    const uint32_t spb = v.samples_per_block;
    const float g0 = v.gain, dgp = (v.gain_target - v.gain) / float(n);
    auto total = [&](uint32_t k, uint32_t fpos) { return (g0 + float(k) * dgp) * fade_at(v, fpos); };
    bool finished = false;

    while (i < n) {
        if (v.fade_pos >= v.fade_len && v.stop_after_fade) { finished = true; break; }
        if (v.cursor >= frames) {
            if (!v.loop || frames == 0) { finished = true; break; }
            v.cursor %= frames;
        }
        uint32_t b = uint32_t(v.cursor / spb);
        uint32_t o = uint32_t(v.cursor % spb);
        if (b != v.cached_block && !fetch_block(v, b)) {
            // stream underrun: silence for the rest of the period, the position does not move (the stream pauses).
            // A voice that is fading out finishes at once - otherwise its fade would wait for data forever.
            audio_stream& s = *v.stream;
            if (v.stop_after_fade) { finished = true; break; }
            s.underrun_frames.fetch_add(n - i, std::memory_order_relaxed);
            if (!s.in_underrun) { s.in_underrun = true; s.underrun_events.fetch_add(1, std::memory_order_relaxed); m_underruns.fetch_add(1, std::memory_order_relaxed); }
            break;
        }
        if (v.stream) v.stream->in_underrun = false;
        uint64_t block_frames64 = frames - uint64_t(b) * spb;
        uint32_t block_end = block_frames64 < spb ? uint32_t(block_frames64) : spb;   // frames of the clip in this block

        uint32_t len = n - i;
        if (v.fade_pos < v.fade_len && v.fade_len - v.fade_pos < len) len = v.fade_len - v.fade_pos;

        if (v.step == 65536 && v.frac == 0) {
            if (block_end - o < len) len = block_end - o;
            float t0 = total(i, v.fade_pos), t1 = total(i + len, v.fade_pos + len);
            audio_mix::voice_s16(dst + i, v.cache + o, len, t0, (t1 - t0) / float(len));
            v.cursor += len;
        } else {
            uint32_t p = (o << 16) | v.frac, limit = block_end << 16;
            uint32_t kmax = (limit - 1 - p) / v.step + 1;
            if (kmax < len) len = kmax;
            float t0 = total(i, v.fade_pos), t1 = total(i + len, v.fade_pos + len);
            p = audio_mix::voice_s16_resample(dst + i, v.cache, len, p, v.step, t0, (t1 - t0) / float(len));
            v.cursor = uint64_t(b) * spb + (p >> 16);
            v.frac = p & 0xffff;
        }
        produced = true;
        if (v.fade_pos < v.fade_len) v.fade_pos += len;
        i += len;
    }
    v.gain = v.gain_target;
    if (!finished && v.fade_pos >= v.fade_len && v.stop_after_fade) finished = true;
    if (!finished && !v.loop && v.cursor >= frames) finished = true;
    return finished;
}

void audio_mixer::render_period(float* out)
{
    const uint32_t n = m_desc.period_frames;
    double t0 = now_us();
    if (m_prof) { for (double& s : m_stage) s = 0.0; m_stage[audio_stage_commands] = m_cmd_us; }
    m_cmd_us = 0;
    m_period_instances = m_period_voices = 0;

    for (uint32_t b = 0; b < m_desc.max_buses; ++b) {
        bus& bs = m_buses[b];
        if (!bs.used) continue;
        bs.paused_chain = bs.pause || (b > 0 && m_buses[bs.parent].paused_chain);   // parents have smaller indices
        memset(bs.l, 0, sizeof(float) * n);
        memset(bs.r, 0, sizeof(float) * n);
        if (bs.effect) memset(bs.in, 0, sizeof(float) * n);
    }

    // instances in groups of 4: voices -> own mono buffer; then filter (4 in SIMD lanes) and spatialise the group
    uint32_t group[4], count = 0;
    for (uint32_t ii = 0; ii < m_desc.max_instances; ++ii) {
        instance& in = m_instances[ii];
        if (in.head == ~0u || m_buses[in.bus].paused_chain) continue;
        float* mono = m_mono[count];
        memset(mono, 0, sizeof(float) * n);
        bool produced = false;
        const double tv = m_prof ? now_us() : 0.0, dec0 = m_stage[audio_stage_decode];
        for (uint32_t vi = in.head; vi != ~0u;) {
            uint32_t next = m_voices[vi].next;
            ++m_period_voices;
            if (render_voice(m_voices[vi], mono, produced)) finish(vi, m_voices[vi].stop_after_fade ? audio_mixer_note_stopped : audio_mixer_note_ended);
            vi = next;
        }
        if (m_prof) m_stage[audio_stage_voices] += (now_us() - tv) - (m_stage[audio_stage_decode] - dec0);   // decode counted apart
        if (!produced) continue;
        ++m_period_instances;                    // all its voices still wait for their delay: nothing to filter / pan
        group[count++] = ii;
        if (count == 4) { flush_group(group, count); count = 0; }
    }
    flush_group(group, count);

    const float inv = 1.0f / float(n);
    double tb = m_prof ? now_us() : 0.0;
    for (uint32_t b = m_desc.max_buses - 1; b > 0; --b) {
        bus& bs = m_buses[b];
        if (!bs.used) continue;
        float target = bs.mute ? 0.0f : bs.volume_target;
        if (bs.effect && !bs.paused_chain) {
            if (bs.reverb >= 0) audio_reverb_kernel::process(m_reverbs[bs.reverb], bs.in, bs.l, bs.r, n, 1.0f, 0.0f);
            if (bs.tail >= 0) {
                audio_reverb_kernel::process(m_reverbs[bs.tail], m_zero, bs.l, bs.r, n, 1.0f, 0.0f);
                if (bs.tail_left <= n) { m_reverb_used[bs.tail] = false; bs.tail = -1; bs.tail_left = 0; }
                else bs.tail_left -= n;
            }
        }
        if (!bs.paused_chain) {
            bus& parent = m_buses[bs.parent];
            audio_mix::mix_f32(parent.l, bs.l, n, bs.volume, (target - bs.volume) * inv);
            audio_mix::mix_f32(parent.r, bs.r, n, bs.volume, (target - bs.volume) * inv);
        }
        bs.volume = target;
    }
    if (m_prof) { const double t = now_us(); m_stage[audio_stage_buses] += t - tb; tb = t; }
    bus& master = m_buses[0];
    float target = master.mute ? 0.0f : master.volume_target;
    if (master.pause) memset(out, 0, sizeof(float) * n * 2);
    else audio_mix::output_stereo(out, master.l, master.r, n, master.volume, (target - master.volume) * inv);
    master.volume = target;
    if (m_prof) m_stage[audio_stage_output] += now_us() - tb;

    m_time.store(m_time.load(std::memory_order_relaxed) + n, std::memory_order_release);
    m_periods.fetch_add(1, std::memory_order_relaxed);
    double us = now_us() - t0;
    if (us > m_period_us_max.load(std::memory_order_relaxed)) m_period_us_max.store(us, std::memory_order_relaxed);
    m_period_us_sum.store(m_period_us_sum.load(std::memory_order_relaxed) + us, std::memory_order_relaxed);
    m_period_count_window.fetch_add(1, std::memory_order_relaxed);
    end_period(us);
}

void audio_mixer::end_period(double us)
{
    audio_mixer_profile& w = m_win;
    if (m_reset_all.exchange(false, std::memory_order_relaxed)) {
        w.period_max_all = 0;
        for (double& s : w.stage_max_all) s = 0;
    }
    const double budget = m_desc.period_frames * 1e6 / m_desc.sample_rate;
    ++w.periods;
    w.period_sum += us;
    if (us > w.period_max) w.period_max = us;
    if (us > w.period_max_all) w.period_max_all = us;
    w.over_50 += us > budget * 0.5; w.over_80 += us > budget * 0.8; w.over_100 += us > budget;
    if (m_period_instances > w.instances_max) w.instances_max = m_period_instances;
    if (m_period_voices > w.voices_max) w.voices_max = m_period_voices;
    if (m_prof) {
        w.stages_on = true;
        for (int s = 0; s < audio_stage_count; ++s) {
            w.stage_sum[s] += m_stage[s];
            if (m_stage[s] > w.stage_max[s]) w.stage_max[s] = m_stage[s];
            if (m_stage[s] > w.stage_max_all[s]) w.stage_max_all[s] = m_stage[s];
        }
    }
    // a window = one second of periods; publish it (never wait for the reader: a busy lock skips this window)
    if (w.periods * uint64_t(m_desc.period_frames) < m_desc.sample_rate) return;
    if (m_pub_lock.try_lock()) { m_pub = w; m_pub_valid = true; m_pub_lock.unlock(); }
    const double keep_max = w.period_max_all;
    double keep_stages[audio_stage_count];
    memcpy(keep_stages, w.stage_max_all, sizeof(keep_stages));
    w = audio_mixer_profile{};
    w.period_max_all = keep_max;
    memcpy(w.stage_max_all, keep_stages, sizeof(keep_stages));
}

bool audio_mixer::profile(audio_mixer_profile& out)
{
    std::lock_guard<std::mutex> lock(m_pub_lock);
    out = m_pub;
    return m_pub_valid;
}

void audio_mixer::flush_group(const uint32_t* group, uint32_t count)
{
    const uint32_t n = m_desc.period_frames;
    double t = m_prof && count ? now_us() : 0.0;
    auto lap = [&](audio_stage s) { if (m_prof && count) { const double now = now_us(); m_stage[s] += now - t; t = now; } };
    uint32_t filtered = 0;
    for (uint32_t k = 0; k < count; ++k) filtered += m_instances[group[k]].filtered;
    if (filtered >= 2) {
        // 4 lanes; instances without a filter and the empty lanes pass through identity stages (bit-exact)
        audio_mix::biquad c[4][2];
        audio_mix::biquad_state s[4][2];
        float* bufs[4];
        for (uint32_t k = 0; k < 4; ++k) {
            if (k < count) {
                const instance& in = m_instances[group[k]];
                bufs[k] = m_mono[k];
                if (in.filtered) { c[k][0] = in.fc[0]; c[k][1] = in.fc[1]; s[k][0] = in.fs[0]; s[k][1] = in.fs[1]; }
            } else {
                bufs[k] = m_silent;
            }
        }
        audio_mix::filter4(bufs, n, c, s);
        for (uint32_t k = 0; k < count; ++k) {
            instance& in = m_instances[group[k]];
            if (in.filtered) { in.fs[0] = s[k][0]; in.fs[1] = s[k][1]; }
        }
    } else if (filtered == 1) {
        for (uint32_t k = 0; k < count; ++k) {
            instance& in = m_instances[group[k]];
            if (in.filtered) audio_mix::filter(m_mono[k], n, in.fc, in.fs);
        }
    }
    lap(audio_stage_filters);
    const float inv = 1.0f / float(n);
    for (uint32_t k = 0; k < count; ++k) {
        instance& in = m_instances[group[k]];
        for (int s = 0; s < 2; ++s) {                       // mono, post filter, pre pan (the reverb makes the width)
            if (in.send_bus[s] == ~0u || (in.send_gain[s] <= 0.0f && in.send_target[s] <= 0.0f)) continue;
            audio_mix::mix_f32(m_buses[in.send_bus[s]].in, m_mono[k], n, in.send_gain[s], (in.send_target[s] - in.send_gain[s]) * inv);
            in.send_gain[s] = in.send_target[s];
        }
    }
    lap(audio_stage_sends);
    for (uint32_t k = 0; k < count; ++k) {
        const instance& in = m_instances[group[k]];
        m_desc.plugin.process(m_desc.plugin.user, group[k], in.spatial, m_mono[k], m_buses[in.bus].l, m_buses[in.bus].r, n);
    }
    lap(audio_stage_spatial);
}

int audio_mixer::take_reverb()
{
    for (uint32_t i = 0; i < m_desc.max_reverbs; ++i) if (!m_reverb_used[i]) { m_reverb_used[i] = true; return int(i); }
    return -1;
}

void audio_mixer::render(float* out, uint32_t frames)
{
#if AUDIO_MIX_SIMD && (defined(_M_X64) || defined(__SSE2__))
    // denormals flushed to zero while mixing: a decaying IIR state is ~100x slower in denormals on x86 (ARM flushes)
    const unsigned int csr = _mm_getcsr();
    _mm_setcsr(csr | 0x8040);                       // FTZ | DAZ
#endif
    const double tc = now_us();
    if (m_last_callback > 0) {
        const double gap = tc - m_last_callback;
        ++m_win.callbacks; m_win.callback_sum += gap;
        if (gap > m_win.callback_max) m_win.callback_max = gap;
    }
    m_last_callback = tc;
    m_prof = m_profiling.load(std::memory_order_relaxed);

    // all commands published so far: a game frame's batch is applied as a whole, before any frame of this call
    audio_mixer_cmd c;
    uint64_t applied = 0;
    while (m_cmds.pop(c)) { apply(c); ++applied; }
    if (applied) m_commands.fetch_add(applied, std::memory_order_relaxed);
    if (m_prof) m_cmd_us += now_us() - tc;

    const uint32_t n = m_desc.period_frames;
    while (frames) {
        if (m_fifo_count) {
            uint32_t k = m_fifo_count < frames ? m_fifo_count : frames;
            memcpy(out, m_fifo + m_fifo_pos * 2, sizeof(float) * k * 2);
            m_fifo_pos += k; m_fifo_count -= k; out += k * 2; frames -= k;
        } else if (frames >= n) {
            render_period(out);
            out += n * 2; frames -= n;
        } else {
            render_period(m_fifo);
            m_fifo_pos = 0; m_fifo_count = n;
        }
    }
    m_notes.commit();
#if AUDIO_MIX_SIMD && (defined(_M_X64) || defined(__SSE2__))
    _mm_setcsr(csr);
#endif
}
