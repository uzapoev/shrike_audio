#ifndef __audio_policy_h__
#define __audio_policy_h__

// audio_policy.h - event instances and voice management (game thread)
//
// Role:      both "Event Runtime" and "Voice Manager" of requrements.txt section 7 (kept together: the score needs the
//            layers' current gains, the start needs the resolved layers).
// Contents:
//   - event runtime (~1/3)   records of instances and their states (pending / starting / playing / virtual / fading);
//                            resolve at post: chance, switch case, random with avoid_repeat, random pitch, delay;
//                            every update: volume / pitch curves, filter cutoffs, send levels, distance / angle;
//                            pending until the bank's sounds are loaded; waiting for stream rings; silent loops start late
//   - voice manager (~2/3)   score: priority by distance + audibility (envelope + layer gain + attenuation, power sum);
//                            select: threshold -> max_instances -> bus limits -> global budget; hysteresis;
//                            steal with fade-out reserve; virtual kill / update / restart; pools of mixer voices /
//                            instances; commands to the mixer, notifications back; binaural: the max_binaural most
//                            audible instances of spatial="hrtf" events (hysteresis_db for the current ones) while it is on
// Thread:    game thread only.
// Not here:  mixing (audio_mixer), file IO (audio_stream), XML (audio_bank), parameter storage of objects - values come
//            through audio_param_source (object -> global -> default lookup lives in audio.cpp).
// Uses:      audio_bank.h (templates), audio_mixer.h, audio_stream.h.
// Tests:     tests/audio_policy_test.cpp. Design: requrements.txt section 9 "Вытеснение".

#include <stdint.h>
#include <vector>

#include "audio_bank.h"

class audio_mixer;
class audio_stream_system;
struct audio_stream;

// Values of parameters / switches and positions of objects (instance -> object -> global -> default lookup and object
// storage live in the caller). `instance` / `generation` identify the event instance (per-instance overrides).
// Any callback may be nullptr: defaults / the position given at post are used.
struct audio_param_source {
    void*       user;
    float       (*parameter)(void* user, uint32_t object, uint32_t instance, uint32_t generation, uint32_t id, float def);
    uint32_t    (*switch_value)(void* user, uint32_t object, uint32_t id, uint32_t def);
    bool        (*position)(void* user, uint32_t object, float out[3]);     // false: object gone, keep the last position
};

struct audio_instance_handle {
    uint32_t index = ~0u, generation = 0;
    bool valid() const { return index != ~0u; }
};

enum audio_instance_state : uint8_t {
    audio_instance_none,        // finished, killed, stale handle
    audio_instance_pending,     // a sound of its bank is not loaded yet (waits up to pending_timeout)
    audio_instance_starting,    // got mixer resources, waits for its streams to fill
    audio_instance_playing,
    audio_instance_virtual,
    audio_instance_fading,      // stolen or stopped, waits for the mixer to report its voices finished
};

struct audio_policy_desc {
    uint32_t    max_voices          = 64;       // audible voices (selection budget)
    uint32_t    voice_reserve       = 16;       // extra mixer voices for fade-outs; mixer must have max_voices + voice_reserve
    uint32_t    max_instances       = 48;       // mixer instances
    uint32_t    max_records         = 512;      // playing + virtual + pending + starting
    float       threshold_db        = -60.0f;
    float       hysteresis_db       = 3.0f;
    uint32_t    hysteresis_priority = 0;        // > 0 blocks equal-priority stealing by loudness
    uint32_t    steal_fade          = 480;      // frames
    uint32_t    return_fade         = 480;
    uint32_t    pending_timeout     = 7200;     // frames (150 ms) for one-shots without their own; loops wait forever
    uint32_t    seed                = 1;        // chance, random, random pitch
    uint32_t    max_binaural        = 16;       // instances marked binaural (audio_spatial_input::binaural) at once
};

// One record as the debug view sees it (debug_records).
struct audio_policy_record_info {
    uint32_t                index, generation;
    const audio_event_desc* desc;
    uint32_t                object;
    audio_instance_state    state;
    float                   rel[3];                 // listener space
    float                   distance, audibility_db;
    int                     priority;
    uint32_t                voices;
    bool                    binaural;
    float                   lowpass_hz, highpass_hz, send[2];   // as sent to the mixer (<= 0: off, -1: none)
};

struct audio_policy_stats {
    uint32_t    records, pending, playing, starting, virtual_, fading, voices_used, voices_audible, binaural, instances_used;
    uint64_t    started, returns, steals, culls, kills, ended, record_steals, post_failed, waits_for_voices, pending_timeouts, late_layers;
};

class audio_policy
{
public:
    bool                    init(const audio_policy_desc& desc, audio_mixer* mixer, audio_stream_system* streams,
                                 const audio_buses* buses = nullptr, const audio_param_source* params = nullptr);
    void                    shutdown();

    // listener: world position and an orthonormal basis (+x right, +y up, +z forward in listener space)
    void                    set_listener(const float pos[3], const float right[3], const float up[3], const float forward[3]);

    audio_instance_handle   post(const audio_event_desc* desc, uint32_t object, const float pos[3]);   // decided in the next update()
    void                    stop(audio_instance_handle h, uint32_t fade_frames);
    void                    set_position(audio_instance_handle h, const float pos[3]);
    audio_instance_state    state(audio_instance_handle h) const;

    // stop instances of an object (~0u: any) and of an event (nullptr: any); loops_only: one-shots play on
    void                    stop_matching(uint32_t object, const audio_event_desc* desc, uint32_t fade_frames, bool loops_only);
    // bank unload: stop everything of the bank, then wait until no instance of it is left (the mixer no longer reads it)
    void                    stop_bank(const audio_bank_data* bank, uint32_t fade_frames);
    uint32_t                count_bank(const audio_bank_data* bank) const;

    void                    update();          // notifications -> pending -> score -> select -> commands -> commit
    const double*           phase_us() const { return m_phase_us; }      // the last update: [0] notes + pending, [1] score,
                                                                         // [2] select + apply, [3] live + commit
    uint32_t                records_capacity() const { return uint32_t(m_records.size()); }
    size_t                  memory_bytes() const;
    void                    debug_records(void (*fn)(void* user, const audio_policy_record_info& r), void* user) const;
    void                    bus_activity(uint32_t* counts, uint32_t bus_count) const;     // playing / starting instances per bus
    void                    set_binaural(bool on) { m_binaural = on; }     // headphones mode: the plugin's expensive path
    bool                    binaural(audio_instance_handle h) const;       // the instance is marked binaural now
    audio_policy_stats      stats() const;

    // scoring, exposed for tests
    float                   audibility_db(audio_instance_handle h) const;
    int                     priority(audio_instance_handle h) const;
    const audio_sound*      layer_sound(audio_instance_handle h, uint32_t layer) const;   // resolved at post (nullptr: layer off)
    uint32_t                layer_voice(audio_instance_handle h, uint32_t layer) const;   // mixer voice or ~0u
    static float            attenuation(float distance, float min_distance, float max_distance);   // inverse, clamped

private:
    struct layer_state {
        const audio_sound*  sound       = nullptr;  // nullptr: chance failed / no switch case
        float               pitch       = 1.0f;     // random part
        uint32_t            delay       = 0;
        uint32_t            voice       = ~0u;
        audio_stream*       stream      = nullptr;
        float               gain        = 0.0f;     // current (curves), linear
        float               pitch_now   = 1.0f;     // current (random x curve)
        float               sent_gain   = -1.0f, sent_pitch = -1.0f;
    };
    struct record {
        const audio_event_desc* desc    = nullptr;
        audio_instance_state state      = audio_instance_none;
        uint32_t        generation      = 0;
        uint32_t        object          = ~0u;
        float           pos[3]          = { 0, 0, 0 };
        uint64_t        start           = 0;        // mixer frame of the time reference
        uint64_t        posted          = 0;
        bool            ever_played     = false;
        bool            stop_requested  = false;
        uint32_t        instance        = ~0u;
        uint32_t        alive           = 0;        // launched voices not yet reported finished
        layer_state     layers[16];
        // scoring
        int             prio            = 0;
        float           aud             = -1000.0f;
        uint32_t        need            = 0;
        float           distance        = 0.0f, angle = 0.0f;
        float           sent_lowpass    = -1.0f, sent_highpass = -1.0f;   // last filter sent to the mixer
        bool            send_on[2]      = { false, false };              // switch condition, resolved at post
        float           sent_send[2]    = { -1.0f, -1.0f };
        float           rel[3]          = { 0, 0, 0 };
        bool            binaural        = false;    // marked for the plugin's binaural path (sent with the spatial input)
    };

    void                    resolve(record& r);
    float                   param(const record& r, uint32_t id) const;
    void                    eval_layers(record& r);
    bool                    score(record& r, uint64_t now);
    bool                    start(uint32_t index, uint64_t now);
    bool                    launch_layer(uint32_t index, uint32_t l, uint64_t now, bool late);
    void                    launch_voices(uint32_t index, uint64_t now);
    void                    stop_voices(record& r, uint32_t fade);
    void                    release(record& r);
    void                    free_reserved(record& r);
    void                    send_spatial(const record& r);
    void                    send_filter(record& r, bool force);           // event's <lowpass> / <highpass>, curves live; air absorption
    void                    send_sends(record& r, bool force);            // event's <send>: level / curve x attenuation
    void                    send_curves(record& r);
    void                    pick_binaural();                              // the max_binaural most audible spatial instances
    float                   rnd();

    audio_policy_desc       m_desc;
    audio_mixer*            m_mixer         = nullptr;
    audio_stream_system*    m_streams       = nullptr;
    const audio_buses*      m_buses         = nullptr;
    audio_param_source      m_params        = {};
    std::vector<record>     m_records;
    std::vector<uint32_t>   m_free_voices, m_free_instances, m_voice_owner;
    std::vector<uint32_t>   m_cand, m_bus_used;
    std::vector<uint8_t>    m_selected;
    float                   m_listener[4][3] = { { 0, 0, 0 }, { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 } };   // pos, right, up, forward
    uint32_t                m_rng           = 1;
    bool                    m_binaural      = false;
    double                  m_phase_us[4]   = {};
    std::vector<std::pair<float, uint32_t>> m_bin;                        // (score, record) candidates of pick_binaural
    audio_policy_stats      m_stats         = {};
};

#endif
