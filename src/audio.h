#ifndef __audio_h__
#define __audio_h__

// audio.h - public API of the Shrike audio module
//
// Role:      event-driven audio: game code posts events and sets parameters / switches, the bank describes how it
//            sounds. Independent of the rest of the engine (own file IO interface, own threads).
// Contents:  ids (audio_id), handles, audio_desc_t, stats, geometry descs, class audio: banks, mixer buses, objects,
//            events, parameters / switches, binaural (HRTF: headphones mode), geometry (Steam Audio, stub), debug:
//            stats (counters, pools), profile (time per stage of each thread), memory, live streams / instances.
// Thread:    every call from one (game) thread; update() once per frame.
// Not here:  everything internal - audio.cpp glues: audio_policy (instances, voice management), audio_mixer (audio
//            thread), audio_stream (IO thread), audio_bank (XML), audio_plugins (spatial), audio_geometry (handles).
// Tests:     tests/audio_api_test.cpp (whole module through this API, bank files on disk).
// Design:    src/audio/requrements.txt; prototype: tests/audio_proto_test.cpp.

#include <stdint.h>
#include <stddef.h>
#include "../mathlib.h"

// IDs: 32-bit FNV-1a of the name, computed when the bank metadata is loaded.
typedef uint32_t audio_bank_id;
typedef uint32_t audio_event_id;
typedef uint32_t audio_parameter_id;
typedef uint32_t audio_switch_id;
typedef uint32_t audio_switch_value;
typedef uint32_t audio_bus_id;

constexpr uint32_t audio_id(const char* name)
{
    uint32_t h = 2166136261u;
    while (*name) { h ^= uint8_t(*name++); h *= 16777619u; }
    return h;
}

// Handles: index + generation. A stale handle makes the call a no-op (warning in dev builds).
struct audio_bank_t              { uint32_t index = ~0u; uint32_t generation = 0; bool valid() const { return index != ~0u; } };
struct audio_object_t            { uint32_t index = ~0u; uint32_t generation = 0; bool valid() const { return index != ~0u; } };
struct audio_playing_t           { uint32_t index = ~0u; uint32_t generation = 0; bool valid() const { return index != ~0u; } };
struct audio_geometry_t          { uint32_t index = ~0u; uint32_t generation = 0; bool valid() const { return index != ~0u; } };
struct audio_geometry_instance_t { uint32_t index = ~0u; uint32_t generation = 0; bool valid() const { return index != ~0u; } };

enum audio_bank_state {
    audio_bank_state_registered,        // metadata loaded, no sound data
    audio_bank_state_loading,
    audio_bank_state_ready,
    audio_bank_state_error,
};

enum audio_playing_state {
    audio_playing_state_none,           // finished, killed or stale handle
    audio_playing_state_pending,        // waits for its bank or for the stream ring to fill
    audio_playing_state_playing,
    audio_playing_state_virtual,        // stolen or inaudible; time runs (update) or waits (restart)
};

struct audio_spatial_plugin;            // audio_plugins.h

// File IO of the module. Not set -> stdio. Called from the module's own threads: register_bank / init (game thread),
// the bank loader and the stream IO thread - calls on different files may run at the same time.
struct audio_file_io {
    void*   user;
    void*   (*open)(void* user, const char* path);
    size_t  (*read_at)(void* user, void* file, uint64_t offset, void* dst, size_t size);
    void    (*close)(void* user, void* file);
    uint64_t (*size)(void* user, void* file);   // optional (nullptr: found by reads); Vorbis streams need the length
};

struct audio_desc_t {
    uint32_t        sample_rate         = 48000;
    uint32_t        period_frames       = 256;      // fixed mixer block (Steam Audio)
    uint32_t        max_voices          = 64;       // audible voices
    uint32_t        voice_reserve       = 16;       // voices for fade-outs of stolen events
    uint32_t        max_instances       = 48;       // event instances sounding at once (mixer nodes)
    uint32_t        max_virtual         = 512;      // records of virtual and pending events
    uint32_t        stream_ring_blocks  = 8;        // 8 x 1024 bytes ~ 340 ms per stream
    uint32_t        max_streams         = 16;
    float           stream_seconds      = 10.0f;    // sounds longer than this stream unless the bank says otherwise
    uint32_t        max_vorbis_streams  = 4;        // Ogg Vorbis streams open at once (each has its own decoder)
    uint32_t        vorbis_memory       = 256 * 1024;   // per decoder (codebooks + work); a file needing more fails
    bool            right_handed        = true;     // listener forward: -Z (right-handed) or +Z (left-handed); engine: is_right_hand()
    bool            start_device        = true;     // false: no playback device, render() is called by the owner (tests, tools)
    float           threshold_db        = -60.0f;   // below -> virtual
    float           hysteresis_db       = 3.0f;     // bonus to playing events
    uint32_t        hysteresis_priority = 0;        // bonus to playing events; > 0 blocks equal-priority stealing by
                                                    // loudness (rapid fire: old tails would outlive every new shot)
    const char*     root_path           = "audio/";
    const char*     init_bank           = "init";   // buses, global mixer settings
    const char*     hrtf_file           = "hrtf.bin";   // next to init.xml: the built-in binaural plugin (tools/hrtf_tool.py);
                                                    // missing -> panning only; nullptr -> not looked for
    uint32_t        max_binaural        = 16;       // binaural instances at once (the most audible spatial ones), the rest panned
    const audio_spatial_plugin* plugin  = nullptr;  // own spatial plugin (audio_plugins.h) instead of the built-in ones;
                                                    // copied at init, its shutdown() is called by the module
    audio_file_io*  io                  = nullptr;
};

struct audio_stats_t {
    uint32_t    voices_audible;
    uint32_t    voices_fading;
    uint32_t    instances_playing;
    uint32_t    instances_binaural;     // of them through HRTF (headphones mode)
    uint32_t    events_virtual;
    uint32_t    events_pending;
    uint32_t    steals;                 // last update: pushed out by a limit
    uint32_t    culls;                  // last update: became inaudible
    uint32_t    returns;                // last update: virtual -> playing
    uint32_t    kills;                  // last update
    uint32_t    post_failed;            // total: no record even after stealing the weakest virtual event
    uint32_t    stream_underruns;       // total
    float       mixer_load_max;         // fraction of a period, max over the last full second (get_profile: per stage)
    uint32_t    command_queue_max;      // = commands.peak
    uint32_t    memory_bytes;           // sound data of the loaded banks (everything: get_memory)

    // pools: in use now, the peak since init / reset_stats(), the limit from audio_desc_t - size them by the peaks
    struct pool { uint32_t used, peak, capacity; };
    pool        voices;                 // mixer voices (audible + fading)
    pool        instances;              // mixer instances
    pool        records;                // playing + virtual + pending events (max_virtual)
    pool        streams;                // open streams (max_streams)
    pool        commands;               // commands of one update (one batch) against the queue size
    uint32_t    commands_dropped;       // total: the command queue was full (a bug: the batch is too big)
    uint32_t    notes_dropped;          // total: the notification queue was full
    uint32_t    waits_for_voices;       // total: an event could not start for lack of free voices / instances
    uint32_t    pending_timeouts;       // total: a one-shot waited too long for its bank
    uint32_t    late_layers;            // total: a silent loop layer started late (became audible)
};

// Stages of the audio thread, in the order of a period.
enum audio_stage : uint8_t {
    audio_stage_commands,               // the game's command batch applied
    audio_stage_decode,                 // IMA ADPCM blocks decoded (clips and stream rings), Vorbis blocks copied
    audio_stage_voices,                 // voices mixed into their instances (gain ramps, resampling)
    audio_stage_filters,                // instance low-pass / high-pass (air absorption included)
    audio_stage_sends,                  // instance -> aux bus sends
    audio_stage_spatial,                // the spatial plugin (panning / HRTF)
    audio_stage_buses,                  // bus effects (reverb) and bus mixing
    audio_stage_output,                 // master: volume, soft clip, interleave
    audio_stage_count
};
const char* audio_stage_name(audio_stage s);

struct audio_timing_t { float avg, max; };

// Time per stage. Values cover the last full second (`window_seconds`) and are refreshed once a second; *_all are the
// maxima since init / reset_stats(). Microseconds unless the name says otherwise.
struct audio_profile_t {
    bool            stages_on;                          // set_profiling(true): the per-stage timers run
    float           window_seconds;
    // audio thread (per mixer period)
    float           budget_us;                          // the period's length: above it the device starves
    audio_timing_t  period;                             // a whole period (always measured)
    audio_timing_t  stage[audio_stage_count];           // per period; zeros while stages_on is false
    float           period_max_all;
    float           stage_max_all[audio_stage_count];
    uint32_t        periods;                            // in the window
    uint32_t        over_50, over_80, over_100;         // periods that took more than 50 / 80 / 100 % of the budget
    audio_timing_t  callback_interval;                  // between device callbacks (jitter, a stalled device)
    uint32_t        instances_max, voices_max;          // rendered in one period (max in the window)
    // game thread
    audio_timing_t  update;                             // audio::update() as a whole
    audio_timing_t  policy[4];                          // its parts: [0] notifications + pending, [1] score,
                                                        // [2] select + start / stop, [3] live instances + commit
    // IO threads
    audio_timing_t  io_read;                            // one file read (ADPCM block, 4 KB of a Vorbis file)
    audio_timing_t  io_decode;                          // one Vorbis block (1024 frames) decoded, reads excluded
    float           io_bytes_per_second;
    audio_timing_t  stream_start_ms;                    // stream open -> ring filled (the start waits that long)
    float           bank_load_ms_last, bank_load_ms_max;
};

// Memory of the module (bytes). Everything except `sounds` / `envelopes` / `metadata` is allocated once at init.
struct audio_memory_t {
    uint32_t        sounds;                             // resident ADPCM data of the loaded banks
    uint32_t        envelopes;                          // their loudness envelopes (voice policy)
    uint32_t        metadata;                           // events, curves, sound tables of the registered banks (approx.)
    uint32_t        mixer;                              // voices + block caches, instances, buses, buffers
    uint32_t        reverbs;                            // FDN delay lines (2 per reverb bus)
    uint32_t        queues;                             // command / notification queues
    uint32_t        streams;                            // stream rings, Vorbis decoders (memory + rings)
    uint32_t        spatial;                            // the spatial plugin (HRTF data + per-source state)
    uint32_t        policy;                             // event records
    uint32_t        total, peak;                        // peak since init / reset_stats()
};

// A live stream (debug_streams).
struct audio_stream_info_t {
    const char*     path;
    const char*     state;                              // "opening" / "ready" / "error" / "closing"
    bool            loop;
    uint32_t        ring_fill, ring_size, ring_fill_min;    // blocks; the minimum since it opened (0: it starved)
    uint32_t        underruns;
    uint64_t        bytes_read;
    float           seconds;                            // position (blocks consumed)
    bool            vorbis;                             // Ogg Vorbis (decoded on the IO thread), else IMA ADPCM
    const char*     error;                              // state "error": why ("" otherwise)
};

// An event instance (debug_instances): what the voice policy sees and decided.
struct audio_instance_info_t {
    audio_playing_t handle;
    const char*     event;                              // event name
    const char*     object;                             // object debug name ("" none, "global" for 2D posts)
    const char*     state;                              // "pending" / "starting" / "playing" / "virtual" / "fading"
    const char*     bus;
    float           position[3];                        // listener space (+x right, +y up, +z forward)
    float           distance;
    float           audibility_db;                      // the score: envelope + layer gains + attenuation
    int             priority;
    uint32_t        voices;                             // mixer voices it holds
    bool            binaural;
    float           lowpass_hz, highpass_hz;            // sent to the mixer (0: off; air absorption included)
    float           send[2];                            // send levels sent to the mixer (-1: none)
};

// Acoustic geometry (Steam Audio). Stubs for now.
struct audio_material_t {
    float       absorption[3];          // low / mid / high
    float       scattering;
    float       transmission[3];
};

struct audio_geometry_desc_t {
    const vec3*             vertices;
    uint32_t                vertex_count;
    const uint32_t*         indices;
    uint32_t                index_count;
    const uint8_t*          material_indices;   // per triangle; null -> materials[0]
    const audio_material_t* materials;
    uint32_t                material_count;
};

// All calls from one (game) thread.
// update() once per frame: mixer notifications -> voice policy -> publish the command batch.
// Fade times are in seconds.
class audio
{
public:
    bool                        init(const audio_desc_t& desc);
    void                        shutdown();
    void                        update(float dt);

    // banks
    audio_bank_t                register_bank(const char* name);        // synchronous: metadata (XML)
    void                        unregister_bank(audio_bank_t bank);
    void                        load_bank(audio_bank_t bank);           // asynchronous, refcount++
    void                        unload_bank(audio_bank_t bank);         // refcount--, unloaded after its voices are silent
    audio_bank_state            get_bank_state(audio_bank_t bank) const;

    // mixer
    void                        set_bus_volume(audio_bus_id bus, float volume, float fade = 0.0f);
    void                        set_bus_mute(audio_bus_id bus, bool mute);
    void                        set_bus_pause(audio_bus_id bus, bool pause);
    void                        set_bus_reverb(audio_bus_id bus, uint32_t preset);    // audio_id(<reverb name>); the old tail rings out
    // ducking is declared in init.xml (<bus> <duck bus="music" gain="0.3" attack="0.2" release="0.8"/> </bus>);
    // the bus's current ducking gain (1 = none), multiplied with its volume
    float                       get_bus_duck(audio_bus_id bus) const;

    // binaural (headphones mode): the most audible spatial instances go through HRTF, the rest stay panned; switches with a
    // one-period crossfade. Off by default (on speakers panning sounds better). Needs hrtf.bin (or an own plugin).
    void                        set_binaural(bool on);
    bool                        binaural_available() const;             // the HRTF data loaded (or an own plugin given)

    // objects
    audio_object_t              create_object(const char* debug_name = nullptr);
    void                        destroy_object(audio_object_t object);  // loops stop with fade, one-shots finish in place
    void                        set_object_transform(audio_object_t object, const vec3& position, const quat& rotation);
    void                        set_object_velocity(audio_object_t object, const vec3& velocity);  // doppler; not derived from positions
    void                        set_listener(audio_object_t object);

    // events
    // When all records are taken, the weakest virtual event is stolen; if there is none, the handle is invalid.
    audio_playing_t             post_event(audio_object_t object, audio_event_id event);
    audio_playing_t             post_event(audio_event_id event);                       // 2D / UI
    void                        stop(audio_playing_t playing, float fade = 0.0f);
    void                        stop_event(audio_object_t object, audio_event_id event, float fade = 0.0f);
    void                        stop_all(audio_object_t object, float fade = 0.0f);
    audio_playing_state         get_state(audio_playing_t playing) const;

    // parameters / switches. Lookup: playing -> object -> global -> default of the event.
    void                        set_parameter(audio_object_t object, audio_parameter_id id, float value);
    void                        set_switch(audio_object_t object, audio_switch_id id, audio_switch_value value);
    void                        set_parameter(audio_playing_t playing, audio_parameter_id id, float value);
    void                        set_global_parameter(audio_parameter_id id, float value);
    void                        set_global_switch(audio_switch_id id, audio_switch_value value);

    // geometry (Steam Audio, stub)
    audio_geometry_t            create_geometry(const audio_geometry_desc_t& desc);
    void                        destroy_geometry(audio_geometry_t geometry);
    audio_geometry_instance_t   create_geometry_instance(audio_geometry_t geometry, const vec3& position, const quat& rotation);
    void                        destroy_geometry_instance(audio_geometry_instance_t instance);
    void                        set_geometry_instance_transform(audio_geometry_instance_t instance, const vec3& position, const quat& rotation);
    void                        set_geometry_instance_enabled(audio_geometry_instance_t instance, bool enabled);  // doors instead of portals

    // debug / profiling (an editor profiler reads these; they cost nothing until called, except set_profiling)
    audio_stats_t               get_stats() const;
    audio_profile_t             get_profile() const;
    audio_memory_t              get_memory() const;
    void                        set_profiling(bool on);                 // per-stage timers of the audio thread (~2 us / period)
    void                        reset_stats();                          // peaks and *_all maxima start over
    // the callbacks run inside the call; the strings live until the next update()
    void                        debug_streams(void (*fn)(void* user, const audio_stream_info_t& s), void* user) const;
    void                        debug_instances(void (*fn)(void* user, const audio_instance_info_t& i), void* user) const;
    const char*                 last_error() const;                     // the last failed call (bank parse, missing file...)

    // without a device (desc.start_device = false): interleaved stereo float, any frame count
    void                        render(float* out, uint32_t frames);

    struct impl;                                                        // defined in audio.cpp only
private:
    impl*                       m_impl = nullptr;
};

#endif
