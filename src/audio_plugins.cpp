// audio_plugins.cpp - built-in spatial plugin: equal-power panning (summary in audio_plugins.h).
// Every audio_plugin_builtin_pan() call creates its own state, released by shutdown(): mixers don't share it.

#include "audio_plugins.h"
#include "audio_mix.h"

#include <math.h>
#include <stdlib.h>

namespace {

struct pan_source { float gl, gr; bool valid; };

struct pan_state {
    pan_source* sources = nullptr;
    uint32_t    count   = 0;
};

bool pan_init(void* user, uint32_t, uint32_t, uint32_t max_sources)
{
    pan_state& s = *(pan_state*)user;
    s.sources = (pan_source*)calloc(max_sources, sizeof(pan_source));
    s.count = max_sources;
    return s.sources != nullptr;
}

void pan_shutdown(void* user)       // the state was created by audio_plugin_builtin_pan(): one per mixer
{
    pan_state* s = (pan_state*)user;
    free(s->sources);
    delete s;
}

size_t pan_memory(void* user) { pan_state& s = *(pan_state*)user; return sizeof(pan_state) + sizeof(pan_source) * s.count; }

void pan_reset(void* user, uint32_t source)
{
    pan_state& s = *(pan_state*)user;
    if (source < s.count) s.sources[source].valid = false;
}

void pan_process(void* user, uint32_t source, const audio_spatial_input& in, const float* mono, float* l, float* r, uint32_t n)
{
    pan_state& s = *(pan_state*)user;
    if (source >= s.count || n == 0) return;
    float gl, gr;
    audio_pan_gains(in, gl, gr);
    pan_source& p = s.sources[source];
    if (!p.valid) { p.gl = gl; p.gr = gr; p.valid = true; }   // first period of an instance: no ramp from a stale value
    float inv = 1.0f / float(n);
    audio_mix::pan_mono_stereo(l, r, mono, n, p.gl, (gl - p.gl) * inv, p.gr, (gr - p.gr) * inv);
    p.gl = gl; p.gr = gr;
}

} // namespace

void audio_pan_gains(const audio_spatial_input& in, float& gl, float& gr)
{
    if (!in.spatial) { gl = gr = in.attenuation; return; }
    float d = in.distance > 1e-4f ? in.distance : 1e-4f;
    float px = in.position[0] / d;
    if (d < 0.25f) px *= d / 0.25f;                 // fold to the centre inside the head: no left/right flip
    px = px < -1.0f ? -1.0f : px > 1.0f ? 1.0f : px;
    float a = (px + 1.0f) * 0.78539816f;            // 0 .. pi/2
    gl = cosf(a) * in.attenuation;
    gr = sinf(a) * in.attenuation;
}

audio_spatial_plugin audio_plugin_builtin_pan()
{
    audio_spatial_plugin p = {};
    p.name          = "builtin_pan";
    p.user          = new pan_state();      // own state: several mixers (editor + game, tests) don't share it
    p.init          = pan_init;
    p.shutdown      = pan_shutdown;
    p.source_reset  = pan_reset;
    p.process       = pan_process;
    p.memory_bytes  = pan_memory;
    return p;
}
