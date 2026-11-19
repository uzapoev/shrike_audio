#ifndef __audio_plugins_h__
#define __audio_plugins_h__

// audio_plugins.h - spatial plugin interface (built-in panning now, Steam Audio later)
//
// Role:      turn the mono mix of an event instance into stereo (panning, HRTF, occlusion filters...).
// Contents:
//   - audio_spatial_input / listener / source   what the core passes in
//   - audio_spatial_plugin                      callbacks: mixer side (source_reset, process), game side (simulate,
//                                               occlusion_db), acoustic geometry (create / instance / transform / enable)
//   - audio_plugin_builtin_pan()                equal-power panning with a gain ramp per period (audio_plugins.cpp)
//   - audio_pan_gains                           its pan law, shared with the HRTF plugin (audio_hrtf.h) for panned sources
// Thread:    process / source_reset - audio thread; simulate / occlusion / geometry - game thread.
// Not here:  distance attenuation - the core applies the event's attenuation curve itself and passes the gain in, so
//            the voice score and what is heard use the same curve (Steam Audio must have its own disabled).
// Rules:     one plugin source per event instance slot; source_reset when a slot is reused (no history carried over).
//            Listener space: +x right, +y up, +z forward.
// Used by:   audio_mixer.cpp (process), audio_geometry.cpp (geometry).

#include <stdint.h>
#include <stddef.h>

struct audio_geometry_desc_t;

struct audio_spatial_input {            // per instance per mixer period, prepared by the core
    float       position[3];            // listener space
    float       distance;
    float       attenuation;            // linear gain from the event's attenuation curve
    bool        spatial;                // false: 2D, mono to both channels at `attenuation`
    bool        binaural;               // the core asks for the expensive path (HRTF): one of the N most audible spatial
                                        // instances while headphones mode is on; a plugin may ignore it
};

struct audio_spatial_listener {         // world space, game thread
    float       position[3];
    float       forward[3];
    float       up[3];
};

struct audio_spatial_source {           // world space, game thread
    float       position[3];
    bool        active;
};

struct audio_spatial_plugin {
    const char* name;
    void*       user;

    bool        (*init)(void* user, uint32_t sample_rate, uint32_t period_frames, uint32_t max_sources);
    void        (*shutdown)(void* user);

    // Mixer thread.
    void        (*source_reset)(void* user, uint32_t source);      // slot reused by another instance: forget all history
    void        (*process)(void* user, uint32_t source, const audio_spatial_input& in,
                           const float* mono, float* l, float* r, uint32_t frames);   // accumulates into l / r

    // Game thread (the plugin may run its own simulation thread). Optional (nullptr).
    void        (*simulate)(void* user, const audio_spatial_listener& listener, const audio_spatial_source* sources, uint32_t count, float dt);
    float       (*occlusion_db)(void* user, uint32_t source);      // goes into the voice score; nullptr -> 0 dB
    size_t      (*memory_bytes)(void* user);                       // profiling (audio_memory_t::spatial); nullptr -> 0

    // Acoustic geometry, game thread. Optional (nullptr): geometry calls are no-ops.
    uint32_t    (*geometry_create)(void* user, const audio_geometry_desc_t& desc);
    void        (*geometry_destroy)(void* user, uint32_t geometry);
    uint32_t    (*geometry_instance_create)(void* user, uint32_t geometry, const float position[3], const float rotation[4]);
    void        (*geometry_instance_destroy)(void* user, uint32_t instance);
    void        (*geometry_instance_transform)(void* user, uint32_t instance, const float position[3], const float rotation[4]);
    void        (*geometry_instance_enable)(void* user, uint32_t instance, bool enabled);
};

// Equal-power panning, gain ramp over each period, no geometry. Every call creates its own state, released by shutdown().
audio_spatial_plugin audio_plugin_builtin_pan();

// The pan law of the built-in plugins (attenuation included; 2D: both at attenuation; inside 0.25 m folds to the centre).
void audio_pan_gains(const audio_spatial_input& in, float& gl, float& gr);

#endif
