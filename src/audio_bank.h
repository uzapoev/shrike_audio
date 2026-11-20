#ifndef __audio_bank_h__
#define __audio_bank_h__

// audio_bank.h - bank data and its loader
//
// Role:      what a bank describes: sounds, curves, layer / event templates, buses; built from the JSON files
//            (audio_format.h structs) with all the checks.
// Contents:
//   - audio_sound            name, file, format, length, envelope (sounds.json), resident data after load (attach)
//   - audio_curve            piecewise linear, up to 16 points, parameter id (declared or built-in distance / angle)
//   - audio_layer_desc       sound pool / switch cases / default, chance, delay, volume, pitch, loop, curves
//   - audio_event_desc       layers, parameters, switches, spatial mode, priorities, attenuation, filters, virtual,
//                            limits, bus
//   - audio_bank_data        sounds + curves + events of one bank; audio_buses from init.json (buses, reverb presets)
//   - audio_build_init / audio_build_bank (from audio_format.h structs; the editor validates with them),
//     audio_parse_init / audio_parse_bank (JSON text -> structs -> build), audio_sound_attach   (audio_bank.cpp)
// Thread:    game thread. A loaded bank does not move in memory: pointers into it stay valid until unload.
// Not here:  reading files (the caller passes buffers), instances (audio_policy), the bank tool (encoding, envelopes).
// Format:    audio_format.h (init.json, <bank>/meta.json, <bank>/sounds/sounds.json).
// Tests:     tests/audio_policy_test.cpp (events written in XML, parse errors).

#include <stdint.h>
#include <vector>
#include <string>

#include "audio_adpcm.h"
#include "audio_vorbis.h"   // k_block_frames
#include "audio_reverb.h"   // audio_reverb_preset
#include "audio.h"          // audio_id

// built-in parameters of curves
static constexpr uint32_t k_audio_param_distance = audio_id("distance");   // metres to the listener
static constexpr uint32_t k_audio_param_angle    = audio_id("angle");      // degrees 0..180 from the listener's forward

enum audio_virtual_behavior : uint8_t { audio_virtual_kill, audio_virtual_update, audio_virtual_restart };
enum audio_spatial_mode : uint8_t { audio_spatial_2d, audio_spatial_pan, audio_spatial_hrtf };

struct audio_sound {
    uint32_t            id                  = 0;
    std::string         name, file;          // file relative to the bank's sound folder
    std::string         path;                // full path (streams open it)
    bool                stream              = false;
    bool                vorbis              = false;  // Ogg Vorbis (always streamed), else IMA ADPCM
    uint64_t            frames              = 0;
    uint32_t            sample_rate         = 0;
    uint32_t            block_align         = 0;
    uint32_t            samples_per_block   = 0;
    std::vector<uint8_t> envelope;           // 1 byte per block: -dB * 2.55 (0 = 0 dB, 255 = silence)
    // resident sounds after load_bank: the WAV in memory and a view into it
    std::vector<uint8_t> data;
    audio_adpcm::clip   clip;
    bool                loaded              = false;
    bool available() const { return stream || loaded; }
};

struct audio_curve {                          // piecewise linear, x ascending, clamped outside
    uint32_t            parameter           = 0;
    uint32_t            count               = 0;
    float               x[16], y[16];
    float               eval(float v) const;
};

struct audio_choice {                         // one switch case (or the plain / default pool)
    uint32_t            value               = 0;        // switch value id (cases only)
    std::vector<const audio_sound*> sounds;             // > 1: random
};

struct audio_layer_desc {
    std::string         name;
    audio_choice        pool;                           // no switch: the layer's sound(s)
    uint32_t            switch_id           = 0;        // 0: no switch
    std::vector<audio_choice> cases;
    audio_choice        fallback;                       // <default> of the switch
    uint32_t            avoid_repeat        = 0;
    float               chance              = 1.0f;
    uint32_t            delay               = 0;        // frames
    float               volume_db           = 0.0f;
    float               pitch_min           = 1.0f, pitch_max = 1.0f;
    bool                loop                = false;
    const audio_curve*  volume_curve        = nullptr;  // linear multiplier
    const audio_curve*  pitch_curve         = nullptr;  // pitch factor
    const audio_curve*  delay_curve         = nullptr;  // seconds, evaluated at post
    mutable uint32_t    history[8]          = { ~0u, ~0u, ~0u, ~0u, ~0u, ~0u, ~0u, ~0u };   // last random picks
};

struct audio_parameter_desc { uint32_t id; float min, max, def; };

// a send to an effect (reverb) bus: level or curve (linear gain, taken after the distance attenuation),
// optionally only for one value of a switch (resolved at post)
struct audio_send_desc { uint32_t bus = 0; float level = 0; const audio_curve* curve = nullptr; uint32_t switch_id = 0, switch_value = 0; };
struct audio_switch_desc    { uint32_t id; uint32_t def; std::vector<uint32_t> values; };

struct audio_bank_data;

struct audio_event_desc {
    uint32_t            id                  = 0;
    std::string         name;
    const audio_bank_data* bank             = nullptr;
    std::vector<audio_layer_desc> layers;
    std::vector<audio_parameter_desc> parameters;
    std::vector<audio_switch_desc> switches;
    bool                loop                = false;    // any layer loops
    audio_spatial_mode  spatial             = audio_spatial_pan;
    uint8_t             priority_far        = 0, priority_near = 0;
    float               min_distance        = 1.0f, max_distance = 50.0f;
    audio_virtual_behavior virtual_behavior = audio_virtual_kill;
    uint32_t            max_instances       = 8;
    uint32_t            bus                 = 0;        // index in audio_buses
    uint32_t            pending_timeout     = 0;        // frames; 0 = from the default
    // instance filter: a fixed cutoff (Hz, 0 = off) or a curve (y in Hz); low-pass >= 20 kHz is off
    float               lowpass_hz          = 0.0f, highpass_hz = 0.0f;
    const audio_curve*  lowpass_curve       = nullptr;
    const audio_curve*  highpass_curve      = nullptr;
    float               air                 = 1.0f;     // air absorption: distance multiplier (0 = none; 2D: none)
    std::vector<audio_send_desc> sends;                 // up to 2
};

struct audio_bank_data {
    uint32_t            id                  = 0;
    std::string         name;
    std::string         sound_folder;       // where the sound files are (<bank>/sounds); sounds get path = sound_folder/file
    std::vector<audio_sound> sounds;
    std::vector<audio_curve> curves;
    std::vector<audio_event_desc> events;
    const audio_sound*  find_sound(uint32_t id) const;
    const audio_event_desc* find_event(uint32_t id) const;
};

struct audio_bus_desc { uint32_t id; std::string name; uint32_t parent; float volume; uint32_t voices; bool reverb; uint32_t preset; };   // voices 0: no limit

// <bus name="voice"> <duck bus="music" gain="0.3" attack="0.2" release="0.8"/> </bus>: while the source bus (or a
// bus under it) has a playing instance and is not paused, the target bus is turned down to `gain` (linear, multiplied
// with its volume); the fade is linear in dB - down over `attack`, back up over `release` (seconds). Several sources
// on one target: the lowest gain wins.
struct audio_duck_desc { uint32_t source, target; float gain, attack, release; };

// Ducking state of one bus (as a target): its gain now, and the depth / release of the last source that ducked it.
struct audio_duck_state { float gain = 1.0f, depth_db = 0.0f, release = 0.0f; };

struct audio_buses {                          // from init.json; index 0 is master, parents have smaller indices
    std::vector<audio_bus_desc> buses;
    std::vector<std::pair<uint32_t, audio_reverb_preset>> reverbs;     // <reverb name=...> presets
    // <air_absorption cutoff_100m min_cutoff>: low-pass of every spatial instance by distance,
    // cutoff = cutoff_100m * sqrt(100 / (distance * event air)), not below min_cutoff; >= 20 kHz: off; cutoff_100m 0: none
    float               air_cutoff_100m     = 4000.0f;  // ~ real air (absorption ~ f^2): 13 kHz at 10 m, 4 kHz at 100 m
    float               air_min_cutoff      = 800.0f;
    float               air_cutoff(float distance, float event_air) const;    // 0: no filter
    std::vector<audio_duck_desc> ducks;
    uint32_t            find(uint32_t id) const;   // ~0u if unknown
    const audio_reverb_preset* find_reverb(uint32_t id) const;
};

// Parsers (game thread). Texts are not null-terminated; errors go to `error` (empty on success).
struct audio_fmt_init;
struct audio_fmt_bank;
struct audio_fmt_sounds;
bool audio_build_init(const audio_fmt_init& in, audio_buses& out, std::string& error);
bool audio_build_bank(const audio_fmt_bank& bank, const audio_fmt_sounds& sounds, const audio_buses& buses, uint32_t sample_rate,
                      float stream_seconds, audio_bank_data& out, std::string& error);
bool audio_parse_init(const char* xml, size_t size, audio_buses& out, std::string& error);
bool audio_parse_bank(const char* bank_xml, size_t bank_size, const char* sounds_xml, size_t sounds_size,
                      const audio_buses& buses, uint32_t sample_rate, float stream_seconds,
                      audio_bank_data& out, std::string& error);

// One step of ducking: active[i] - bus i (or a bus under it) sounds and is not paused; state per bus (as a target).
// Returns true if any gain changed.
bool audio_duck_update(const audio_buses& b, const uint8_t* active, float dt, audio_duck_state* state);

// Resident sound data loaded: takes the WAV bytes, checks them against sounds.json.
bool audio_sound_attach(audio_sound& s, std::vector<uint8_t>&& wav, std::string& error);

#endif
