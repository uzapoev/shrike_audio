// audio_hrtf.cpp - built-in binaural spatial plugin (summary in audio_hrtf.h).
// Parts: hrtf.bin parse, direction interpolation, the plugin (state per source, process).

#include "audio_hrtf.h"
#include "audio_mix.h"

#include <math.h>
#include <string.h>

// ---------------------------------------------------------------------------------------------------------------
// data

namespace {

template <class T> bool take(const uint8_t*& p, const uint8_t* end, T& v)
{
    if (size_t(end - p) < sizeof(T)) return false;
    memcpy(&v, p, sizeof(T));
    p += sizeof(T);
    return true;
}

} // namespace

bool audio_hrtf_data::parse(const void* data, size_t size, std::string& error)
{
    const uint8_t* p = (const uint8_t*)data;
    const uint8_t* end = p + size;
    char magic[4];
    uint32_t version = 0;
    if (size < 4 || memcmp(p, "HRTF", 4) != 0) { error = "hrtf: not an HRTF file"; return false; }
    memcpy(magic, p, 4); p += 4;
    if (!take(p, end, version) || version != 1) { error = "hrtf: unknown version"; return false; }
    if (!take(p, end, sample_rate) || !take(p, end, taps) || !take(p, end, count) || !take(p, end, scale)) { error = "hrtf: truncated header"; return false; }
    if (taps == 0 || taps > k_max_taps || count == 0 || count > 100000 || !(scale > 0)) { error = "hrtf: bad header values"; return false; }
    if (size_t(end - p) != size_t(count) * 16 + size_t(count) * taps * 4) { error = "hrtf: size does not match the header"; return false; }

    dir.resize(size_t(count) * 3);
    delay.resize(size_t(count) * 2);
    for (uint32_t i = 0; i < count; ++i) {
        float az, el, dl, dr;
        take(p, end, az); take(p, end, el); take(p, end, dl); take(p, end, dr);
        const float a = az * 0.017453293f, e = el * 0.017453293f;
        dir[i * 3 + 0] = cosf(e) * sinf(a);     // 90 degrees: right
        dir[i * 3 + 1] = sinf(e);
        dir[i * 3 + 2] = cosf(e) * cosf(a);     // 0: forward
        delay[i * 2 + 0] = dl < 0 ? 0 : dl > float(k_max_delay) ? float(k_max_delay) : dl;
        delay[i * 2 + 1] = dr < 0 ? 0 : dr > float(k_max_delay) ? float(k_max_delay) : dr;
    }
    h.resize(size_t(count) * taps * 2);
    memcpy(h.data(), p, h.size() * sizeof(int16_t));
    return true;
}

void audio_hrtf_data::interpolate(const float d[3], float* hl_rev, float* hr_rev, float& delay_l, float& delay_r) const
{
    // the 3 nearest measured directions (largest dot products)
    int best[3] = { -1, -1, -1 };
    float dot[3] = { -2, -2, -2 };
    for (uint32_t i = 0; i < count; ++i) {
        const float* v = &dir[i * 3];
        const float c = v[0] * d[0] + v[1] * d[1] + v[2] * d[2];
        if (c <= dot[2]) continue;
        int k = 2;
        while (k > 0 && c > dot[k - 1]) { dot[k] = dot[k - 1]; best[k] = best[k - 1]; --k; }
        dot[k] = c; best[k] = int(i);
    }
    // weights by inverse angle; on a measured direction it alone
    float w[3] = { 0, 0, 0 }, sum = 0;
    for (int k = 0; k < 3; ++k) {
        if (best[k] < 0) continue;
        const float c = dot[k] > 1.0f ? 1.0f : dot[k] < -1.0f ? -1.0f : dot[k];
        const float angle = acosf(c);
        if (angle < 0.002f) { w[0] = w[1] = w[2] = 0; w[k] = 1; sum = 1; break; }
        w[k] = 1.0f / angle; sum += w[k];
    }
    delay_l = delay_r = 0;
    for (uint32_t t = 0; t < taps; ++t) hl_rev[t] = hr_rev[t] = 0;
    for (int k = 0; k < 3; ++k) {
        if (w[k] == 0) continue;
        const float g = w[k] / sum, gs = g * scale;
        const int16_t* hl = &h[size_t(best[k]) * taps * 2];
        const int16_t* hr = hl + taps;
        for (uint32_t t = 0; t < taps; ++t) { hl_rev[taps - 1 - t] += gs * float(hl[t]); hr_rev[taps - 1 - t] += gs * float(hr[t]); }
        delay_l += g * delay[size_t(best[k]) * 2 + 0];
        delay_r += g * delay[size_t(best[k]) * 2 + 1];
    }
}

// ---------------------------------------------------------------------------------------------------------------
// plugin

namespace {

const float k_turn_cos = 0.99984770f;           // cos(1 degree): a smaller turn keeps the current pair
const float k_near = 0.25f;                     // closer than this: panned (the direction folds to the centre)
const uint32_t k_dl = audio_hrtf_data::k_max_delay + 1;    // delay line history (+ period + 1: the interpolation reads one ahead)

struct hrtf_source {
    float*  x;                  // taps-1 history + period
    float*  hl; float* hr;      // current pair, reversed
    float*  yl; float* yr;      // delay lines: k_dl history + period
    float   d[3];               // direction of the current pair
    float   dl, dr;             // current delays
    float   gl, gr;             // pan gains (attenuation included)
    float   att;                // binaural gain
    float   bw;                 // binaural weight 0 .. 1
    bool    valid, has_pair;
};

struct hrtf_state {
    audio_hrtf_data         data;
    uint32_t                period = 0, count = 0;
    std::vector<float>      memory;
    std::vector<hrtf_source> sources;
    std::vector<float>      nl, nr, tl, tr, ul, ur;     // scratch: new pair, FIR outputs (current, new)
};

bool hrtf_init(void* user, uint32_t, uint32_t period, uint32_t max_sources)
{
    hrtf_state& s = *(hrtf_state*)user;
    const uint32_t taps = s.data.taps;
    s.period = period;
    s.count = max_sources;
    const size_t per = size_t(taps - 1 + period) + 2 * taps + 2 * size_t(k_dl + period + 1);
    s.memory.assign(per * max_sources, 0.0f);
    s.sources.assign(max_sources, hrtf_source{});
    for (uint32_t i = 0; i < max_sources; ++i) {
        float* m = s.memory.data() + per * i;
        hrtf_source& src = s.sources[i];
        src.x = m; m += taps - 1 + period;
        src.hl = m; m += taps;
        src.hr = m; m += taps;
        src.yl = m; m += k_dl + period + 1;
        src.yr = m;
    }
    s.nl.assign(taps, 0); s.nr.assign(taps, 0);
    s.tl.assign(period, 0); s.tr.assign(period, 0); s.ul.assign(period, 0); s.ur.assign(period, 0);
    return true;
}

void hrtf_shutdown(void* user) { delete (hrtf_state*)user; }

size_t hrtf_memory(void* user)
{
    const hrtf_state& s = *(const hrtf_state*)user;
    return sizeof(hrtf_state) + s.memory.capacity() * sizeof(float) + s.sources.capacity() * sizeof(hrtf_source)
         + (s.nl.capacity() + s.nr.capacity() + s.tl.capacity() + s.tr.capacity() + s.ul.capacity() + s.ur.capacity()) * sizeof(float)
         + s.data.dir.capacity() * sizeof(float) + s.data.delay.capacity() * sizeof(float) + s.data.h.capacity() * sizeof(int16_t);
}

void hrtf_reset(void* user, uint32_t source)
{
    hrtf_state& s = *(hrtf_state*)user;
    if (source >= s.count) return;
    hrtf_source& src = s.sources[source];
    memset(src.x, 0, sizeof(float) * (s.data.taps - 1 + s.period));
    memset(src.yl, 0, sizeof(float) * (k_dl + s.period + 1));
    memset(src.yr, 0, sizeof(float) * (k_dl + s.period + 1));
    src.valid = src.has_pair = false;
}

// fractional delay ramped d0 -> d1 over the period, x gain ramp, accumulated into out; line = k_dl history + n new
void delay_out(float* line, const float* in, float* out, uint32_t n, float d0, float d1, float g0, float g1)
{
    memcpy(line + k_dl, in, sizeof(float) * n);
    const float inv = 1.0f / float(n), dd = (d1 - d0) * inv, dg = (g1 - g0) * inv;
    for (uint32_t i = 0; i < n; ++i) {
        const float pos = float(k_dl + i) - (d0 + dd * float(i));     // >= 1: the delay is at most k_dl - 1
        const uint32_t j = uint32_t(pos);
        const float f = pos - float(j);
        out[i] += (line[j] + (line[j + 1] - line[j]) * f) * (g0 + dg * float(i));
    }
    memmove(line, line + n, sizeof(float) * k_dl);
}

void hrtf_process(void* user, uint32_t source, const audio_spatial_input& in, const float* mono, float* l, float* r, uint32_t n)
{
    hrtf_state& s = *(hrtf_state*)user;
    if (source >= s.count || n == 0 || n > s.period) return;
    hrtf_source& src = s.sources[source];
    const uint32_t taps = s.data.taps;

    float gl, gr;
    audio_pan_gains(in, gl, gr);
    const bool want = in.spatial && in.binaural && in.distance >= k_near;
    const float bw1 = want ? 1.0f : 0.0f;
    if (!src.valid) { src.gl = gl; src.gr = gr; src.att = in.attenuation; src.bw = bw1; src.valid = true; }
    const float bw0 = src.bw;

    memcpy(src.x + taps - 1, mono, sizeof(float) * n);
    const float inv = 1.0f / float(n);

    // panned part
    if (bw0 < 1.0f || bw1 < 1.0f) {
        const float a0 = 1.0f - bw0, a1 = 1.0f - bw1;
        audio_mix::pan_mono_stereo(l, r, mono, n, src.gl * a0, (gl * a1 - src.gl * a0) * inv, src.gr * a0, (gr * a1 - src.gr * a0) * inv);
    }

    // binaural part
    if (bw0 > 0.0f || bw1 > 0.0f) {
        const float id = 1.0f / in.distance;
        float d[3] = { in.position[0] * id, in.position[1] * id, in.position[2] * id };
        if (!want) { d[0] = src.d[0]; d[1] = src.d[1]; d[2] = src.d[2]; }       // fading out: keep the pair
        if (!src.has_pair) {                                                     // fading in: start clean
            s.data.interpolate(d, src.hl, src.hr, src.dl, src.dr);
            memcpy(src.d, d, sizeof(d));
            memset(src.yl, 0, sizeof(float) * k_dl);
            memset(src.yr, 0, sizeof(float) * k_dl);
            src.has_pair = true;
        }
        audio_mix::fir_stereo_store(s.tl.data(), s.tr.data(), src.x + taps - 1, src.hl, src.hr, taps, n);
        float dl1 = src.dl, dr1 = src.dr;
        if (d[0] * src.d[0] + d[1] * src.d[1] + d[2] * src.d[2] < k_turn_cos) {
            s.data.interpolate(d, s.nl.data(), s.nr.data(), dl1, dr1);
            audio_mix::fir_stereo_store(s.ul.data(), s.ur.data(), src.x + taps - 1, s.nl.data(), s.nr.data(), taps, n);
            for (uint32_t i = 0; i < n; ++i) {
                const float t = float(i) * inv;
                s.tl[i] += (s.ul[i] - s.tl[i]) * t;
                s.tr[i] += (s.ur[i] - s.tr[i]) * t;
            }
            memcpy(src.hl, s.nl.data(), sizeof(float) * taps);
            memcpy(src.hr, s.nr.data(), sizeof(float) * taps);
            memcpy(src.d, d, sizeof(d));
        }
        delay_out(src.yl, s.tl.data(), l, n, src.dl, dl1, src.att * bw0, in.attenuation * bw1);
        delay_out(src.yr, s.tr.data(), r, n, src.dr, dr1, src.att * bw0, in.attenuation * bw1);
        src.dl = dl1; src.dr = dr1;
        if (bw1 == 0.0f) src.has_pair = false;                                   // faded out: next fade-in starts clean
    }

    memmove(src.x, src.x + n, sizeof(float) * (taps - 1));
    src.gl = gl; src.gr = gr; src.att = in.attenuation; src.bw = bw1;
}

} // namespace

audio_spatial_plugin audio_plugin_builtin_hrtf(const audio_hrtf_data& data)
{
    hrtf_state* s = new hrtf_state();
    s->data = data;
    audio_spatial_plugin p = {};
    p.name          = "builtin_hrtf";
    p.user          = s;
    p.init          = hrtf_init;
    p.shutdown      = hrtf_shutdown;
    p.source_reset  = hrtf_reset;
    p.process       = hrtf_process;
    p.memory_bytes  = hrtf_memory;
    return p;
}
