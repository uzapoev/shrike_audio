#ifndef __audio_mixer_h__
#define __audio_mixer_h__

// audio_mixer.h - mixer (own graph, audio thread)
//
// Role:      render the graph voice -> event instance -> bus -> master in fixed periods (256 frames).
// Contents:
//   - commands (game -> audio)   voice play / stop / gain / pitch, instance start / spatial / filter / send,
//                                bus config / volume / reverb
//   - notes (audio -> game)      voice finished (ended / stopped)
//   - voice                      resident clip or stream ring, rendered in segments (block / clip / fade / period end),
//                                one kernel call per segment; delays are frame counters (a paused bus holds them)
//   - instance                   mono mix of its voices -> filter (low-pass + high-pass) -> spatial plugin -> its bus;
//                                rendered in groups of 4 (filter4 runs the group in SIMD lanes)
//   - bus                        volume ramp, mute, pause; index 0 is master, parents have smaller indices;
//                                an effect bus sums the instances' sends and runs a reverb (FDN, audio_reverb.h)
//   - device                     own miniaudio playback device; a FIFO keeps whole periods for any callback size
// Thread:    game thread: push() ... commit() once per frame, pop_note(), time(), stats().
//            audio thread: render() - device callback (start_device) or tests. Everything a voice / instance / bus
//            holds changes only on the audio thread while applying commands.
// Not here:  decisions which voice plays (audio_policy), file IO (audio_stream), spatial math (audio_plugins).
// Uses:      audio_mix.h (kernels), audio_adpcm.h, audio_queue.h, audio_plugins.h, audio_stream.h (rings), miniaudio (device only).
// Tests:     tests/audio_mixer_test.cpp.

#include <stdint.h>
#include <atomic>

#include "audio_adpcm.h"
#include "audio_queue.h"
#include "audio_plugins.h"
#include "audio_mix.h"          // biquad
#include "audio_reverb.h"
#include "audio.h"              // audio_stage

#include <mutex>

struct audio_stream;

enum audio_mixer_cmd_type : uint8_t {
    audio_mixer_cmd_voice_play,
    audio_mixer_cmd_voice_stop,
    audio_mixer_cmd_voice_gain,
    audio_mixer_cmd_voice_pitch,
    audio_mixer_cmd_instance_start,
    audio_mixer_cmd_instance_spatial,
    audio_mixer_cmd_instance_filter,
    audio_mixer_cmd_instance_send,
    audio_mixer_cmd_bus_config,
    audio_mixer_cmd_bus_volume,
    audio_mixer_cmd_bus_reverb,
};

struct audio_mixer_cmd {
    audio_mixer_cmd_type    type;
    uint32_t                target;             // voice, instance or bus index
    union {
        struct {
            const audio_adpcm::clip* clip;      // resident clip, or
            audio_stream*   stream;             // a stream in state ready (offset = its start_frame)
            uint32_t        instance;
            uint32_t        delay;              // frames from the moment the batch is applied
            uint64_t        offset;             // start position in the clip (virtual 'update' return)
            uint32_t        step;               // pitch, 16.16 fixed point (65536 = 1)
            float           gain;
            uint32_t        fade_in;            // frames
            bool            loop;
        } play;
        struct { uint32_t fade; }               stop;
        struct { float gain; }                  gain;       // ramped over one period
        struct { uint32_t step; }               pitch;
        struct { uint32_t bus; }                instance;
        audio_spatial_input                     spatial;
        struct { float lowpass_hz, highpass_hz; } filter;      // <= 0: stage off (lowpass >= 20 kHz: off too)
        struct { uint32_t slot, bus; float gain; } send;      // slot 0 / 1; bus ~0u: off; gain ramped over a period
        struct { float decay, predelay, damping, size, level; } reverb;   // audio_reverb_preset as POD
        struct { uint32_t parent; float volume; bool mute, pause; } bus;
        struct { float volume; }                bus_volume;
    };
};

enum audio_mixer_note_reason : uint8_t { audio_mixer_note_ended, audio_mixer_note_stopped };

struct audio_mixer_note {
    uint32_t                voice;
    audio_mixer_note_reason reason;
};

struct audio_mixer_desc {
    uint32_t                sample_rate         = 48000;
    uint32_t                period_frames       = 256;
    uint32_t                max_voices          = 256;
    uint32_t                max_instances       = 48;
    uint32_t                max_buses           = 16;
    uint32_t                max_samples_per_block = 2041;  // 1024-byte mono IMA blocks
    uint32_t                max_reverbs         = 4;        // FDN instances (2 per reverb bus: current + ringing out)
    audio_spatial_plugin    plugin              = {};       // empty -> built-in panning
};

struct audio_mixer_stats {
    uint64_t    periods;
    uint64_t    commands;
    uint64_t    play_on_busy_voice;     // game reused a voice before its notification: a bug on the game side
    uint64_t    notes_dropped;
    uint64_t    stream_underruns;
    double      period_us_max;          // since the previous stats() call
    double      period_us_avg;
    double      budget_us;
};

// One profiling window of the audio thread (about a second of periods), published for the game thread.
struct audio_mixer_profile {
    uint32_t    periods;
    double      period_sum, period_max;                             // microseconds
    double      stage_sum[audio_stage_count], stage_max[audio_stage_count];
    uint32_t    over_50, over_80, over_100;
    uint32_t    callbacks;
    double      callback_sum, callback_max;
    uint32_t    instances_max, voices_max;                          // rendered in one period
    double      period_max_all, stage_max_all[audio_stage_count];   // since init / reset_profile
    bool        stages_on;
};

class audio_mixer
{
public:
    bool                init(const audio_mixer_desc& desc);
    void                shutdown();

    bool                start_device();             // own miniaudio playback device; render() is then called by it
    void                stop_device();

    // game thread
    bool                push(const audio_mixer_cmd& cmd)    { ++m_batch; if (m_cmds.push(cmd)) return true; ++m_cmds_dropped; return false; }
    void                commit()                            { m_cmds.commit(); if (m_batch > m_batch_peak) m_batch_peak = m_batch; m_batch = 0; }
    uint32_t            batch_peak() const                  { return m_batch_peak; }       // commands of one commit, max
    uint32_t            batch_capacity() const              { return 4096; }
    uint64_t            commands_dropped() const            { return m_cmds_dropped; }
    void                set_profiling(bool on)              { m_profiling.store(on, std::memory_order_relaxed); }
    bool                profile(audio_mixer_profile& out);  // the last published window; false: none yet
    void                reset_profile()                     { m_batch_peak = 0; m_reset_all.store(true, std::memory_order_relaxed); }
    size_t              memory_bytes() const                { return m_memory_bytes; }     // the one block (voices ... reverbs)
    size_t              reverb_bytes() const                { return m_reverb_bytes; }
    size_t              queue_bytes() const                 { return sizeof(m_cmds) + sizeof(m_notes); }
    bool                pop_note(audio_mixer_note& note)    { return m_notes.pop(note); }
    uint64_t            time() const                        { return m_time.load(std::memory_order_acquire); }   // frames rendered
    audio_mixer_stats   stats();
    const audio_mixer_desc& desc() const                    { return m_desc; }

    // audio thread: interleaved stereo float, any frame count (an internal FIFO keeps whole periods)
    void                render(float* out, uint32_t frames);

    // tests only, from the thread that calls render()
    struct voice_info { bool active; uint64_t cursor; uint32_t delay; uint32_t instance; };
    voice_info          debug_voice(uint32_t i) const { const voice& v = m_voices[i]; return { v.active, v.cursor, v.delay, v.instance }; }

private:
    struct voice {
        const audio_adpcm::clip* clip   = nullptr;
        audio_stream*   stream          = nullptr;
        uint64_t        frames          = 0;            // length of the clip / stream
        uint32_t        samples_per_block = 0;
        int16_t*        cache           = nullptr;      // decoded block + first sample of the next block
        uint32_t        cached_block    = ~0u;
        uint64_t        cursor          = 0;
        uint32_t        frac            = 0;            // 16.16 fraction of the position
        uint32_t        step            = 65536;
        uint32_t        delay           = 0;
        float           gain            = 1.0f, gain_target = 1.0f;
        float           fade_from       = 1.0f, fade_to = 1.0f;
        uint32_t        fade_len        = 0, fade_pos = 0;
        bool            stop_after_fade = false;
        bool            loop            = false;
        bool            active          = false;
        uint32_t        instance        = ~0u;
        uint32_t        next            = ~0u;          // next voice of the same instance
    };
    struct instance {
        uint32_t            head        = ~0u;
        uint32_t            bus         = 0;
        audio_spatial_input spatial     = {};
        bool                filtered    = false;    // any filter stage on
        float               lowpass_hz  = 0, highpass_hz = 0;
        audio_mix::biquad   fc[2];                  // low-pass, high-pass (identity when off)
        audio_mix::biquad_state fs[2];
        uint32_t            send_bus[2]  = { ~0u, ~0u };            // aux buses (reverb), after the filter
        float               send_gain[2] = { 0, 0 }, send_target[2] = { 0, 0 };
    };
    struct bus {
        uint32_t        parent          = 0;
        float           volume          = 1.0f, volume_target = 1.0f;
        bool            mute = false, pause = false, paused_chain = false, used = false;
        float*          l               = nullptr;
        float*          r               = nullptr;
        float*          in              = nullptr;      // mono sum of the sends (effect buses)
        bool            effect          = false;        // a reverb bus
        int             reverb          = -1;           // current FDN (receives the sends)
        int             tail            = -1;           // previous FDN ringing out after a preset change
        uint32_t        tail_left       = 0;            // frames until the tail is below -70 dB
        float           decay           = 0;            // of the current preset
    };

    void                apply(const audio_mixer_cmd& c);
    void                render_period(float* out);
    void                flush_group(const uint32_t* group, uint32_t count);    // filter, sends, spatialise rendered instances
    int                 take_reverb();
    bool                render_voice(voice& v, float* dst, bool& produced);   // returns true if the voice finished
    void                end_period(double us);                                 // profiling window, publish
    bool                fetch_block(voice& v, uint32_t b);                     // false: stream underrun
    void                finish(uint32_t index, audio_mixer_note_reason reason);
    void                unlink(uint32_t index);
    float               fade_at(const voice& v, uint32_t pos) const;

    audio_mixer_desc    m_desc;
    voice*              m_voices        = nullptr;
    instance*           m_instances     = nullptr;
    bus*                m_buses         = nullptr;
    float*              m_mono[4]       = {};       // instances are rendered in groups of 4 (filter4 runs them in SIMD lanes)
    float*              m_silent        = nullptr;  // a dummy lane for an incomplete group
    float*              m_zero          = nullptr;  // always zero: input of a ringing-out reverb
    audio_reverb*       m_reverbs       = nullptr;
    bool*               m_reverb_used   = nullptr;
    float*              m_fifo          = nullptr;      // one period, interleaved
    uint32_t            m_fifo_pos      = 0, m_fifo_count = 0;
    void*               m_memory        = nullptr;
    void*               m_device        = nullptr;      // ma_device*

    audio_spsc_queue<audio_mixer_cmd, 4096>   m_cmds;
    audio_spsc_queue<audio_mixer_note, 4096>  m_notes;
    std::atomic<uint64_t> m_time{0};

    // stats, written by the audio thread
    std::atomic<uint64_t> m_periods{0}, m_commands{0}, m_play_on_busy{0}, m_notes_dropped{0}, m_underruns{0};
    std::atomic<double>   m_period_us_max{0}, m_period_us_sum{0};
    std::atomic<uint64_t> m_period_count_window{0};

    // profiling. Audio thread: the current period's stage times and the open window; the game thread takes the last
    // closed window under m_pub_lock (the audio thread only try_locks: it never waits, a busy lock skips one window)
    std::atomic<bool>     m_profiling{false}, m_reset_all{false};
    bool                  m_prof                = false;            // m_profiling, read once per render()
    double                m_stage[audio_stage_count] = {};          // this period
    double                m_cmd_us              = 0;                // commands of this render() call -> its first period
    double                m_last_callback       = 0;
    uint32_t              m_period_instances    = 0, m_period_voices = 0;
    audio_mixer_profile   m_win                 = {};
    audio_mixer_profile   m_pub                 = {};
    bool                  m_pub_valid           = false;
    std::mutex            m_pub_lock;
    size_t                m_memory_bytes        = 0, m_reverb_bytes = 0;
    // game thread
    uint32_t              m_batch               = 0, m_batch_peak = 0;
    uint64_t              m_cmds_dropped        = 0;
};

#endif
