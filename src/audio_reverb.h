#ifndef __audio_reverb_h__
#define __audio_reverb_h__

// audio_reverb.h - algorithmic reverb (FDN) for aux buses
//
// Role:      the tail of a room: a mono send in -> a wide stereo tail out (wet only), accumulated into a bus.
// Contents:
//   - audio_reverb_preset   decay (T60, s), predelay (s), damping (Hz: the tail gets darker above), size (0..1: line
//                           lengths), level (output gain)
//   - audio_reverb          8-line feedback delay network: predelay -> lines of mutually different lengths -> one-pole
//                           low-pass per line in the feedback -> Hadamard 8x8 mix -> per-line gain from T60;
//                           L / R taken from the lines with different sign patterns
//   - process_scalar / process_simd   one period; the SIMD version runs the 8 lines in 2 x 4 lanes (blocks of 4 samples
//                           transposed: every line is at least 4 samples long, so a block reads before it writes)
// Thread:    audio thread. Memory is given by the mixer (no allocation here).
// Not here:  sends and the aux bus (audio_mixer), presets from init.xml (audio_bank), early reflections, modulation.
// Rules:     both versions use the same operation order: bit-exact. Line lengths change only while the instance is
//            silent (a preset change takes a fresh instance; the old one rings out).
// Tests:     tests/audio_mix_test.cpp (bit-exact, decay time, stability, speed).

#include <stdint.h>
#include <string.h>
#include <math.h>

#include "audio_mix.h"      // AUDIO_MIX_SIMD, intrinsics

struct audio_reverb_preset {
    float   decay       = 1.2f;     // seconds to -60 dB
    float   predelay    = 0.02f;    // seconds (<= 0.1)
    float   damping     = 6000.0f;  // Hz
    float   size        = 0.6f;     // 0.1 .. 1
    float   level       = 1.0f;
};

struct audio_reverb {
    static const uint32_t k_lines = 8;
    static const uint32_t k_max_line = 4096;                // samples, size 1 at 48 kHz fits
    static const uint32_t k_max_predelay = 4800;            // 100 ms at 48 kHz
    static size_t memory_floats() { return size_t(k_lines) * k_max_line + k_max_predelay; }

    float*      line[k_lines]   = {};
    uint32_t    len[k_lines]    = {};
    uint32_t    pos[k_lines]    = {};
    float       gain[k_lines]   = {};                       // feedback gain per line (T60)
    float       lp[k_lines]     = {};                       // one-pole state per line
    float       damp            = 0.5f;                     // one-pole coefficient
    float*      pre             = nullptr;
    uint32_t    pre_len         = 1, pre_pos = 0;
    float       level           = 1.0f;

    void attach(float* memory)
    {
        for (uint32_t i = 0; i < k_lines; ++i) line[i] = memory + size_t(i) * k_max_line;
        pre = memory + size_t(k_lines) * k_max_line;
    }

    // Fresh state for a preset (lengths may change: only on a silent instance).
    void configure(const audio_reverb_preset& p, float rate)
    {
        static const float base[k_lines] = { 1427, 1601, 1783, 1951, 2113, 2297, 2477, 2659 };   // ~30..55 ms at 48 kHz
        const float size = p.size < 0.1f ? 0.1f : p.size > 1.0f ? 1.0f : p.size;
        const float decay = p.decay < 0.05f ? 0.05f : p.decay;
        for (uint32_t i = 0; i < k_lines; ++i) {
            uint32_t l = uint32_t(base[i] * size * rate / 48000.0f) | 1u;
            len[i] = l < 8 ? 8 : l > k_max_line ? k_max_line : l;
            pos[i] = 0;
            gain[i] = powf(10.0f, -3.0f * float(len[i]) / (decay * rate));
            lp[i] = 0.0f;
            memset(line[i], 0, sizeof(float) * len[i]);
        }
        // decay is the T60 of the lows / mids; above `damping` the tail dies faster. >= 18 kHz: no damping at all
        // (a one-pole near Nyquist still eats highs on every pass and shortens the broadband decay).
        float d = p.damping < 100.0f ? 100.0f : p.damping;
        damp = d >= 18000.0f ? 1.0f : 1.0f - expf(-6.2831853f * d / rate);
        uint32_t pl = uint32_t(p.predelay * rate);
        pre_len = pl < 1 ? 1 : pl > k_max_predelay ? k_max_predelay : pl;
        pre_pos = 0;
        memset(pre, 0, sizeof(float) * pre_len);
        level = p.level;
    }
};

namespace audio_reverb_kernel {

// input scaling and output sign patterns (orthogonal: L and R decorrelated)
static const float k_in[8] = { 0.35f, -0.35f, 0.35f, -0.35f, 0.35f, -0.35f, 0.35f, -0.35f };
static const float k_l[8]  = { 0.5f, 0.0f, 0.5f, 0.0f, -0.5f, 0.0f, 0.5f, 0.0f };
static const float k_r[8]  = { 0.0f, 0.5f, 0.0f, -0.5f, 0.0f, 0.5f, 0.0f, 0.5f };
static const float k_h = 0.35355339f;                       // 1 / sqrt(8): Hadamard 8 stays orthogonal

// Hadamard 8 as butterflies: (a, b) -> (a + b, a - b) between the halves, then H4 inside each half.
inline void hadamard8(float* f)
{
    float a[8];
    for (int i = 0; i < 4; ++i) { a[i] = f[i] + f[i + 4]; a[i + 4] = f[i] - f[i + 4]; }
    for (int h = 0; h < 8; h += 4) {
        float s0 = a[h] + a[h + 1], d0 = a[h] - a[h + 1], s1 = a[h + 2] + a[h + 3], d1 = a[h + 2] - a[h + 3];
        f[h] = s0 + s1; f[h + 1] = d0 + d1; f[h + 2] = s0 - s1; f[h + 3] = d0 - d1;
    }
}

// one sample: returns L, R (before level)
inline void step_scalar(audio_reverb& r, float x, float& out_l, float& out_r)
{
    float in = r.pre[r.pre_pos];                            // predelay
    r.pre[r.pre_pos] = x;
    if (++r.pre_pos == r.pre_len) r.pre_pos = 0;

    float f[8], o[8];
    for (int i = 0; i < 8; ++i) {
        o[i] = r.line[i][r.pos[i]];
        r.lp[i] = r.lp[i] + r.damp * (o[i] - r.lp[i]);
        f[i] = r.lp[i];
    }
    hadamard8(f);
    for (int i = 0; i < 8; ++i) {
        r.line[i][r.pos[i]] = in * k_in[i] + r.gain[i] * (f[i] * k_h);
        if (++r.pos[i] == r.len[i]) r.pos[i] = 0;
    }
    // outputs: ((l0 + l4) + (l1 + l5)) + ((l2 + l6) + (l3 + l7)) - the order the SIMD lanes sum in
    float l4[4], r4[4];
    for (int i = 0; i < 4; ++i) { l4[i] = o[i] * k_l[i] + o[i + 4] * k_l[i + 4]; r4[i] = o[i] * k_r[i] + o[i + 4] * k_r[i + 4]; }
    out_l = ((l4[0] + l4[1]) + l4[2]) + l4[3];
    out_r = ((r4[0] + r4[1]) + r4[2]) + r4[3];
}

// One period: in (mono send sum) -> accumulate into l / r with a level ramp g + i * dg.
inline void process_scalar(audio_reverb& r, const float* in, float* l, float* rr, uint32_t n, float g, float dg)
{
    for (uint32_t i = 0; i < n; ++i) {
        float ol, orr;
        step_scalar(r, in[i], ol, orr);
        float k = (g + float(i) * dg) * r.level;
        l[i] += ol * k;
        rr[i] += orr * k;
    }
}

#if AUDIO_MIX_SIMD
// H4 inside one vector, same order as hadamard8's second stage
inline __m128 hadamard4(__m128 a)
{
    // s0 = a0 + a1, d0 = a0 - a1, s1 = a2 + a3, d1 = a2 - a3  ->  [s0 d0 s1 d1]  (x + (-y) == x - y exactly)
    __m128 lo = _mm_shuffle_ps(a, a, _MM_SHUFFLE(2, 2, 0, 0));                 // a0 a0 a2 a2
    __m128 hi = _mm_shuffle_ps(a, a, _MM_SHUFFLE(3, 3, 1, 1));                 // a1 a1 a3 a3
    __m128 sdv = _mm_add_ps(lo, _mm_xor_ps(hi, _mm_setr_ps(0.0f, -0.0f, 0.0f, -0.0f)));   // s0 d0 s1 d1
    // [s0 + s1, d0 + d1, s0 - s1, d0 - d1]
    __m128 p = _mm_shuffle_ps(sdv, sdv, _MM_SHUFFLE(1, 0, 1, 0));              // s0 d0 s0 d0
    __m128 q = _mm_shuffle_ps(sdv, sdv, _MM_SHUFFLE(3, 2, 3, 2));              // s1 d1 s1 d1
    return _mm_add_ps(p, _mm_xor_ps(q, _mm_setr_ps(0.0f, 0.0f, -0.0f, -0.0f)));
}

// 8 lines in 2 x 4 lanes, 4 samples per block. Every line is >= 8 samples long, so within a block all reads come
// before all writes. A block where some line wraps around its end is done scalar (rare).
inline void process_simd(audio_reverb& r, const float* in, float* l, float* rr, uint32_t n, float g, float dg)
{
    const __m128 damp = _mm_set1_ps(r.damp), h = _mm_set1_ps(k_h);
    const __m128 in0 = _mm_loadu_ps(k_in), in1 = _mm_loadu_ps(k_in + 4);
    const __m128 kl0 = _mm_loadu_ps(k_l), kl1 = _mm_loadu_ps(k_l + 4), kr0 = _mm_loadu_ps(k_r), kr1 = _mm_loadu_ps(k_r + 4);
    const __m128 g0 = _mm_loadu_ps(r.gain), g1 = _mm_loadu_ps(r.gain + 4);
    __m128 lp0 = _mm_loadu_ps(r.lp), lp1 = _mm_loadu_ps(r.lp + 4);
    uint32_t i = 0;
    while (i < n) {
        bool fits = i + 4 <= n && r.pre_pos + 4 <= r.pre_len;
        for (int k = 0; k < 8 && fits; ++k) if (r.pos[k] + 4 > r.len[k]) fits = false;
        if (!fits) {                                                            // wrap or tail: scalar step
            _mm_storeu_ps(r.lp, lp0); _mm_storeu_ps(r.lp + 4, lp1);
            float ol, orr;
            step_scalar(r, in[i], ol, orr);
            float k = (g + float(i) * dg) * r.level;
            l[i] += ol * k; rr[i] += orr * k;
            lp0 = _mm_loadu_ps(r.lp); lp1 = _mm_loadu_ps(r.lp + 4);
            ++i;
            continue;
        }
        // predelay: 4 samples out, 4 in
        float pin[4];
        for (int t = 0; t < 4; ++t) { pin[t] = r.pre[r.pre_pos + t]; r.pre[r.pre_pos + t] = in[i + t]; }
        r.pre_pos += 4; if (r.pre_pos == r.pre_len) r.pre_pos = 0;

        // read 4 time steps of the 8 lines, transpose -> o[t] = lines 0..3 / 4..7 at time t
        __m128 a0 = _mm_loadu_ps(r.line[0] + r.pos[0]), a1 = _mm_loadu_ps(r.line[1] + r.pos[1]);
        __m128 a2 = _mm_loadu_ps(r.line[2] + r.pos[2]), a3 = _mm_loadu_ps(r.line[3] + r.pos[3]);
        __m128 b0 = _mm_loadu_ps(r.line[4] + r.pos[4]), b1 = _mm_loadu_ps(r.line[5] + r.pos[5]);
        __m128 b2 = _mm_loadu_ps(r.line[6] + r.pos[6]), b3 = _mm_loadu_ps(r.line[7] + r.pos[7]);
        _MM_TRANSPOSE4_PS(a0, a1, a2, a3);
        _MM_TRANSPOSE4_PS(b0, b1, b2, b3);
        __m128 oa[4] = { a0, a1, a2, a3 }, ob[4] = { b0, b1, b2, b3 };
        __m128 wa[4], wb[4], ol[4], orv[4];
        for (int t = 0; t < 4; ++t) {
            lp0 = _mm_add_ps(lp0, _mm_mul_ps(damp, _mm_sub_ps(oa[t], lp0)));
            lp1 = _mm_add_ps(lp1, _mm_mul_ps(damp, _mm_sub_ps(ob[t], lp1)));
            // Hadamard 8: halves, then H4
            __m128 fa = hadamard4(_mm_add_ps(lp0, lp1)), fb = hadamard4(_mm_sub_ps(lp0, lp1));
            __m128 x = _mm_set1_ps(pin[t]);
            wa[t] = _mm_add_ps(_mm_mul_ps(x, in0), _mm_mul_ps(g0, _mm_mul_ps(fa, h)));
            wb[t] = _mm_add_ps(_mm_mul_ps(x, in1), _mm_mul_ps(g1, _mm_mul_ps(fb, h)));
            ol[t] = _mm_add_ps(_mm_mul_ps(oa[t], kl0), _mm_mul_ps(ob[t], kl1));
            orv[t] = _mm_add_ps(_mm_mul_ps(oa[t], kr0), _mm_mul_ps(ob[t], kr1));
        }
        _MM_TRANSPOSE4_PS(wa[0], wa[1], wa[2], wa[3]);                           // back to per line, 4 time steps
        _MM_TRANSPOSE4_PS(wb[0], wb[1], wb[2], wb[3]);
        _mm_storeu_ps(r.line[0] + r.pos[0], wa[0]); _mm_storeu_ps(r.line[1] + r.pos[1], wa[1]);
        _mm_storeu_ps(r.line[2] + r.pos[2], wa[2]); _mm_storeu_ps(r.line[3] + r.pos[3], wa[3]);
        _mm_storeu_ps(r.line[4] + r.pos[4], wb[0]); _mm_storeu_ps(r.line[5] + r.pos[5], wb[1]);
        _mm_storeu_ps(r.line[6] + r.pos[6], wb[2]); _mm_storeu_ps(r.line[7] + r.pos[7], wb[3]);
        for (int k = 0; k < 8; ++k) { r.pos[k] += 4; if (r.pos[k] == r.len[k]) r.pos[k] = 0; }

        // outputs: lane sums of ol[t] -> transpose -> ((lane0 + lane1) + lane2) + lane3 for the 4 time steps at once
        _MM_TRANSPOSE4_PS(ol[0], ol[1], ol[2], ol[3]);
        _MM_TRANSPOSE4_PS(orv[0], orv[1], orv[2], orv[3]);
        __m128 L = _mm_add_ps(_mm_add_ps(_mm_add_ps(ol[0], ol[1]), ol[2]), ol[3]);
        __m128 R = _mm_add_ps(_mm_add_ps(_mm_add_ps(orv[0], orv[1]), orv[2]), orv[3]);
        __m128 k = _mm_mul_ps(audio_mix::ramp4(g, dg, _mm_setr_ps(float(i), float(i + 1), float(i + 2), float(i + 3))), _mm_set1_ps(r.level));
        _mm_storeu_ps(l + i,  _mm_add_ps(_mm_loadu_ps(l + i),  _mm_mul_ps(L, k)));
        _mm_storeu_ps(rr + i, _mm_add_ps(_mm_loadu_ps(rr + i), _mm_mul_ps(R, k)));
        i += 4;
    }
    _mm_storeu_ps(r.lp, lp0); _mm_storeu_ps(r.lp + 4, lp1);
}
#endif

inline void process(audio_reverb& r, const float* in, float* l, float* rr, uint32_t n, float g, float dg)
{
#if AUDIO_MIX_SIMD
    process_simd(r, in, l, rr, n, g, dg);
#else
    process_scalar(r, in, l, rr, n, g, dg);
#endif
}

} // namespace audio_reverb_kernel

#endif
