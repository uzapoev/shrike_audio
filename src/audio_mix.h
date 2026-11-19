#ifndef __audio_mix_h__
#define __audio_mix_h__

// audio_mix.h - mixer kernels (hot path of the audio thread)
//
// Role:      whole-algorithm kernels the mixer calls per voice / instance / bus / master per period.
// Contents:
//   - voice_s16           voice, pitch 1: int16 block cache -> x gain ramp -> accumulate into the instance mono buffer
//   - voice_s16_resample  voice, pitch != 1: own linear interpolation (16.16 position) + gain ramp + accumulate
//   - pan_mono_stereo     instance: mono -> planar stereo with two gain ramps, accumulate into the bus
//   - mix_f32             bus: accumulate into the parent with a volume ramp
//   - output_stereo       master: planar -> interleaved, volume ramp, soft clip (transparent below 0.75)
//   - filter / filter4    instance: low-pass + high-pass biquads in place; filter4 = 4 instances in the SIMD lanes
//   - fir_stereo_store    HRTF plugin: mono -> L / R through two FIRs (head-related impulse responses), stores (=)
//   - mono_s16_store      Vorbis streams: float (1 or 2 channels) -> mono int16 with rounding and clamping, stores (=)
// Thread:    audio thread (pure functions, no state); mono_s16_store runs on the IO thread.
// Not here:  ADPCM decoding (audio_adpcm.h), the graph itself (audio_mixer.cpp).
// Rules:     every kernel has a scalar reference (_scalar) and a SIMD version (_simd, SSE2; ARM through sse2neon),
//            bit-exact with each other; SIMD is kept only if it beats the scalar one. audio_mix::<kernel> is what the
//            mixer calls. Planar float buffers, gains as a linear ramp over the call: gain(i) = g + i * dg.
//            Kernels accumulate (+=) into dst unless the name says otherwise.
// Tests:     tests/audio_mix_test.cpp (correctness + speed, scalar vs SIMD).

#include <stdint.h>
#include <math.h>

#if defined(_M_X64) || defined(__SSE2__) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
    #define AUDIO_MIX_SIMD 1
    #include <emmintrin.h>
#elif defined(__ARM_NEON) || defined(__ARM_NEON__) || defined(_M_ARM64)
    #define AUDIO_MIX_SIMD 1
    #include "sse2neon/sse2neon.h"
#else
    #define AUDIO_MIX_SIMD 0
#endif

namespace audio_mix {

static const float k_s16_to_f32 = 1.0f / 32768.0f;

// ---------------------------------------------------------------------------
// 1. Voice, pitch 1: dst[i] += src[i] / 32768 * (g + i * dg)
// ---------------------------------------------------------------------------
inline void voice_s16_scalar(float* dst, const int16_t* src, uint32_t n, float g, float dg)
{
    for (uint32_t i = 0; i < n; ++i) dst[i] += float(src[i]) * k_s16_to_f32 * (g + float(i) * dg);
}

// ---------------------------------------------------------------------------
// 2. Voice, pitch != 1: linear interpolation, position in 16.16 fixed point relative to src[0].
//    src must hold samples up to index ((pos + (n - 1) * step) >> 16) + 1. Returns the position after n outputs.
// ---------------------------------------------------------------------------
inline uint32_t voice_s16_resample_scalar(float* dst, const int16_t* src, uint32_t n, uint32_t pos, uint32_t step, float g, float dg)
{
    for (uint32_t i = 0; i < n; ++i) {
        uint32_t idx = pos >> 16;
        float f = float(pos & 0xffff) * (1.0f / 65536.0f);
        float s0 = float(src[idx]), s1 = float(src[idx + 1]);
        dst[i] += (s0 + (s1 - s0) * f) * k_s16_to_f32 * (g + float(i) * dg);
        pos += step;
    }
    return pos;
}

// ---------------------------------------------------------------------------
// 3. Instance, mono -> stereo planar: L[i] += in[i] * (gl + i * dgl), R[i] += in[i] * (gr + i * dgr)
// ---------------------------------------------------------------------------
inline void pan_mono_stereo_scalar(float* l, float* r, const float* in, uint32_t n, float gl, float dgl, float gr, float dgr)
{
    for (uint32_t i = 0; i < n; ++i) { float x = in[i]; l[i] += x * (gl + float(i) * dgl); r[i] += x * (gr + float(i) * dgr); }
}

// ---------------------------------------------------------------------------
// 4. Bus: dst[i] += src[i] * (g + i * dg)
// ---------------------------------------------------------------------------
inline void mix_f32_scalar(float* dst, const float* src, uint32_t n, float g, float dg)
{
    for (uint32_t i = 0; i < n; ++i) dst[i] += src[i] * (g + float(i) * dg);
}

// ---------------------------------------------------------------------------
// 5. Master -> device: interleaved stereo, gain ramp, soft clip. Writes, not accumulates.
//    Soft clip is transparent below the knee (|x| <= 0.75), above it y = t + (1 - t) * u / (1 + u), u = (|x| - t) / (1 - t):
//    continuous value and slope at the knee, approaches 1 and never reaches it.
// ---------------------------------------------------------------------------
static const float k_clip_knee = 0.75f;

inline float soft_clip(float x)
{
    float a = x < 0.0f ? -x : x;
    if (a <= k_clip_knee) return x;
    float u = (a - k_clip_knee) * (1.0f / (1.0f - k_clip_knee));
    float y = k_clip_knee + (1.0f - k_clip_knee) * (u / (1.0f + u));
    return x < 0.0f ? -y : y;
}

inline void output_stereo_scalar(float* out, const float* l, const float* r, uint32_t n, float g, float dg)
{
    for (uint32_t i = 0; i < n; ++i) { float k = g + float(i) * dg; out[i * 2] = soft_clip(l[i] * k); out[i * 2 + 1] = soft_clip(r[i] * k); }
}

// ---------------------------------------------------------------------------
// 6. Instance filter: low-pass then high-pass, each a 2nd order Butterworth biquad (12 dB / octave), in place on the
//    instance's mono mix. Transposed direct form II. An IIR is serial inside one signal, so the SIMD version runs
//    4 instances in the 4 lanes (blocks of 4 samples transposed), not 4 samples of one instance.
//    A stage that is off has identity coefficients (b0 = 1): it passes the signal bit-exactly.
// ---------------------------------------------------------------------------
struct biquad       { float b0 = 1, b1 = 0, b2 = 0, a1 = 0, a2 = 0; };     // normalised (a0 = 1)
struct biquad_state { float z1 = 0, z2 = 0; };

// RBJ cookbook, Q = 1/sqrt(2). hz is clamped below 0.45 of the rate (the formula breaks down near Nyquist).
inline biquad biquad_lowpass(float hz, float rate)
{
    float w = 6.2831853f * (hz < rate * 0.45f ? hz : rate * 0.45f) / rate, cw = cosf(w), alpha = sinf(w) * 0.70710678f;
    float inv = 1.0f / (1.0f + alpha);
    biquad c; c.b0 = (1.0f - cw) * 0.5f * inv; c.b1 = (1.0f - cw) * inv; c.b2 = c.b0; c.a1 = -2.0f * cw * inv; c.a2 = (1.0f - alpha) * inv;
    return c;
}

inline biquad biquad_highpass(float hz, float rate)
{
    float w = 6.2831853f * (hz < rate * 0.45f ? hz : rate * 0.45f) / rate, cw = cosf(w), alpha = sinf(w) * 0.70710678f;
    float inv = 1.0f / (1.0f + alpha);
    biquad c; c.b0 = (1.0f + cw) * 0.5f * inv; c.b1 = -(1.0f + cw) * inv; c.b2 = c.b0; c.a1 = -2.0f * cw * inv; c.a2 = (1.0f - alpha) * inv;
    return c;
}

inline float biquad_step(const biquad& c, biquad_state& s, float x)
{
    float y = c.b0 * x + s.z1;
    s.z1 = c.b1 * x - c.a1 * y + s.z2;
    s.z2 = c.b2 * x - c.a2 * y;
    return y;
}

// one instance: stages c[0] (low-pass), c[1] (high-pass)
inline void filter_scalar(float* buf, uint32_t n, const biquad c[2], biquad_state s[2])
{
    for (uint32_t i = 0; i < n; ++i) buf[i] = biquad_step(c[1], s[1], biquad_step(c[0], s[0], buf[i]));
}

// four instances (the reference of filter4_simd)
inline void filter4_scalar(float* const buf[4], uint32_t n, const biquad c[4][2], biquad_state s[4][2])
{
    for (int k = 0; k < 4; ++k) filter_scalar(buf[k], n, c[k], s[k]);
}

// ---------------------------------------------------------------------------
// 7. Two FIRs on one input (HRTF): l[i] = sum_k hl[k] * x[i - (taps-1) + k], r the same with hr. The responses are
//    stored reversed (hl[taps-1] is the tap of the newest sample); x[-(taps-1) .. -1] is the history kept by the caller.
// ---------------------------------------------------------------------------
inline void fir_stereo_store_scalar(float* l, float* r, const float* x, const float* hl, const float* hr, uint32_t taps, uint32_t n)
{
    const float* base = x - (taps - 1);
    for (uint32_t i = 0; i < n; ++i) {
        float al = 0, ar = 0;
        for (uint32_t k = 0; k < taps; ++k) { al += hl[k] * base[i + k]; ar += hr[k] * base[i + k]; }
        l[i] = al; r[i] = ar;
    }
}

// ---------------------------------------------------------------------------
// 8. Decoded float -> mono int16 (Vorbis streams): v = r ? (l + r) * 0.5 : l; x = v * 32768 clamped to
//    [-32768, 32767] (NaN -> 32767), rounded to nearest even. Writes, not accumulates.
// ---------------------------------------------------------------------------
inline int16_t s16_from_f32(float v)
{
    float x = v * 32768.0f;
    x = x < 32767.0f ? x : 32767.0f;                                    // the order of minps / maxps (NaN -> 32767)
    x = x > -32768.0f ? x : -32768.0f;
    return int16_t(lrintf(x));
}

inline void mono_s16_store_scalar(int16_t* dst, const float* l, const float* r, uint32_t n)
{
    if (r) for (uint32_t i = 0; i < n; ++i) dst[i] = s16_from_f32((l[i] + r[i]) * 0.5f);
    else   for (uint32_t i = 0; i < n; ++i) dst[i] = s16_from_f32(l[i]);
}

#if AUDIO_MIX_SIMD
// ---------------------------------------------------------------------------
// SIMD versions. Same arithmetic order as the scalar ones (no FMA in SSE2) -> results are bit-exact.
// ---------------------------------------------------------------------------
inline __m128 ramp4(float g, float dg, __m128 idx) { return _mm_add_ps(_mm_set1_ps(g), _mm_mul_ps(idx, _mm_set1_ps(dg))); }

inline void voice_s16_simd(float* dst, const int16_t* src, uint32_t n, float g, float dg)
{
    const __m128 k = _mm_set1_ps(k_s16_to_f32), eight = _mm_set1_ps(8.0f);
    __m128 idx0 = _mm_setr_ps(0, 1, 2, 3), idx1 = _mm_setr_ps(4, 5, 6, 7);
    uint32_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m128i s = _mm_loadu_si128((const __m128i*)(src + i));
        __m128 lo = _mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpacklo_epi16(s, s), 16));
        __m128 hi = _mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpackhi_epi16(s, s), 16));
        __m128 d0 = _mm_loadu_ps(dst + i), d1 = _mm_loadu_ps(dst + i + 4);
        d0 = _mm_add_ps(d0, _mm_mul_ps(_mm_mul_ps(lo, k), ramp4(g, dg, idx0)));
        d1 = _mm_add_ps(d1, _mm_mul_ps(_mm_mul_ps(hi, k), ramp4(g, dg, idx1)));
        _mm_storeu_ps(dst + i, d0); _mm_storeu_ps(dst + i + 4, d1);
        idx0 = _mm_add_ps(idx0, eight); idx1 = _mm_add_ps(idx1, eight);
    }
    for (; i < n; ++i) dst[i] += float(src[i]) * k_s16_to_f32 * (g + float(i) * dg);
}

inline uint32_t voice_s16_resample_simd(float* dst, const int16_t* src, uint32_t n, uint32_t pos, uint32_t step, float g, float dg)
{
    const __m128 k = _mm_set1_ps(k_s16_to_f32), inv = _mm_set1_ps(1.0f / 65536.0f), four = _mm_set1_ps(4.0f);
    const __m128i mask = _mm_set1_epi32(0xffff), step4 = _mm_set1_epi32(int(step * 4));
    __m128i p = _mm_setr_epi32(int(pos), int(pos + step), int(pos + step * 2), int(pos + step * 3));
    __m128 idx = _mm_setr_ps(0, 1, 2, 3);
    alignas(16) int32_t ix[4];
    uint32_t i = 0;
    for (; i + 4 <= n; i += 4) {
        _mm_store_si128((__m128i*)ix, _mm_srli_epi32(p, 16));
        __m128 s0 = _mm_setr_ps(float(src[ix[0]]),     float(src[ix[1]]),     float(src[ix[2]]),     float(src[ix[3]]));       // gather by hand
        __m128 s1 = _mm_setr_ps(float(src[ix[0] + 1]), float(src[ix[1] + 1]), float(src[ix[2] + 1]), float(src[ix[3] + 1]));
        __m128 f = _mm_mul_ps(_mm_cvtepi32_ps(_mm_and_si128(p, mask)), inv);
        __m128 v = _mm_add_ps(s0, _mm_mul_ps(_mm_sub_ps(s1, s0), f));
        __m128 d = _mm_loadu_ps(dst + i);
        _mm_storeu_ps(dst + i, _mm_add_ps(d, _mm_mul_ps(_mm_mul_ps(v, k), ramp4(g, dg, idx))));
        p = _mm_add_epi32(p, step4);
        idx = _mm_add_ps(idx, four);
    }
    pos += step * i;
    for (; i < n; ++i) {
        uint32_t ii = pos >> 16;
        float f = float(pos & 0xffff) * (1.0f / 65536.0f);
        float s0 = float(src[ii]), s1 = float(src[ii + 1]);
        dst[i] += (s0 + (s1 - s0) * f) * k_s16_to_f32 * (g + float(i) * dg);
        pos += step;
    }
    return pos;
}

inline void pan_mono_stereo_simd(float* l, float* r, const float* in, uint32_t n, float gl, float dgl, float gr, float dgr)
{
    const __m128 four = _mm_set1_ps(4.0f);
    __m128 idx = _mm_setr_ps(0, 1, 2, 3);
    uint32_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m128 x = _mm_loadu_ps(in + i);
        _mm_storeu_ps(l + i, _mm_add_ps(_mm_loadu_ps(l + i), _mm_mul_ps(x, ramp4(gl, dgl, idx))));
        _mm_storeu_ps(r + i, _mm_add_ps(_mm_loadu_ps(r + i), _mm_mul_ps(x, ramp4(gr, dgr, idx))));
        idx = _mm_add_ps(idx, four);
    }
    for (; i < n; ++i) { float x = in[i]; l[i] += x * (gl + float(i) * dgl); r[i] += x * (gr + float(i) * dgr); }
}

inline void mix_f32_simd(float* dst, const float* src, uint32_t n, float g, float dg)
{
    const __m128 four = _mm_set1_ps(4.0f);
    __m128 idx = _mm_setr_ps(0, 1, 2, 3);
    uint32_t i = 0;
    for (; i + 4 <= n; i += 4) {
        _mm_storeu_ps(dst + i, _mm_add_ps(_mm_loadu_ps(dst + i), _mm_mul_ps(_mm_loadu_ps(src + i), ramp4(g, dg, idx))));
        idx = _mm_add_ps(idx, four);
    }
    for (; i < n; ++i) dst[i] += src[i] * (g + float(i) * dg);
}

inline __m128 soft_clip4(__m128 x)
{
    const __m128 sign = _mm_set1_ps(-0.0f), knee = _mm_set1_ps(k_clip_knee), one = _mm_set1_ps(1.0f);
    __m128 s = _mm_and_ps(x, sign);
    __m128 a = _mm_andnot_ps(sign, x);
    __m128 u = _mm_mul_ps(_mm_sub_ps(a, knee), _mm_set1_ps(1.0f / (1.0f - k_clip_knee)));
    __m128 y = _mm_add_ps(knee, _mm_mul_ps(_mm_set1_ps(1.0f - k_clip_knee), _mm_div_ps(u, _mm_add_ps(one, u))));
    __m128 above = _mm_cmpgt_ps(a, knee);
    __m128 r = _mm_or_ps(_mm_and_ps(above, y), _mm_andnot_ps(above, a));
    return _mm_or_ps(r, s);
}

inline void output_stereo_simd(float* out, const float* l, const float* r, uint32_t n, float g, float dg)
{
    const __m128 four = _mm_set1_ps(4.0f);
    __m128 idx = _mm_setr_ps(0, 1, 2, 3);
    uint32_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m128 k = ramp4(g, dg, idx);
        __m128 a = soft_clip4(_mm_mul_ps(_mm_loadu_ps(l + i), k));
        __m128 b = soft_clip4(_mm_mul_ps(_mm_loadu_ps(r + i), k));
        _mm_storeu_ps(out + i * 2,     _mm_unpacklo_ps(a, b));
        _mm_storeu_ps(out + i * 2 + 4, _mm_unpackhi_ps(a, b));
        idx = _mm_add_ps(idx, four);
    }
    for (; i < n; ++i) { float k = g + float(i) * dg; out[i * 2] = soft_clip(l[i] * k); out[i * 2 + 1] = soft_clip(r[i] * k); }
}

// 4 instances in the lanes: load 4 samples of each buffer, transpose (-> 4 time steps x 4 instances), run both
// stages step by step, transpose back. Same operation order as biquad_step: bit-exact with filter4_scalar.
inline void filter4_simd(float* const buf[4], uint32_t n, const biquad c[4][2], biquad_state s[4][2])
{
    __m128 B0[2], B1[2], B2[2], A1[2], A2[2], Z1[2], Z2[2];
    for (int st = 0; st < 2; ++st) {
        B0[st] = _mm_setr_ps(c[0][st].b0, c[1][st].b0, c[2][st].b0, c[3][st].b0);
        B1[st] = _mm_setr_ps(c[0][st].b1, c[1][st].b1, c[2][st].b1, c[3][st].b1);
        B2[st] = _mm_setr_ps(c[0][st].b2, c[1][st].b2, c[2][st].b2, c[3][st].b2);
        A1[st] = _mm_setr_ps(c[0][st].a1, c[1][st].a1, c[2][st].a1, c[3][st].a1);
        A2[st] = _mm_setr_ps(c[0][st].a2, c[1][st].a2, c[2][st].a2, c[3][st].a2);
        Z1[st] = _mm_setr_ps(s[0][st].z1, s[1][st].z1, s[2][st].z1, s[3][st].z1);
        Z2[st] = _mm_setr_ps(s[0][st].z2, s[1][st].z2, s[2][st].z2, s[3][st].z2);
    }
    uint32_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m128 x[4] = { _mm_loadu_ps(buf[0] + i), _mm_loadu_ps(buf[1] + i), _mm_loadu_ps(buf[2] + i), _mm_loadu_ps(buf[3] + i) };
        _MM_TRANSPOSE4_PS(x[0], x[1], x[2], x[3]);                      // x[t] = sample i + t of the 4 instances
        for (int t = 0; t < 4; ++t) {
            __m128 v = x[t];
            for (int st = 0; st < 2; ++st) {
                __m128 y = _mm_add_ps(_mm_mul_ps(B0[st], v), Z1[st]);
                Z1[st] = _mm_add_ps(_mm_sub_ps(_mm_mul_ps(B1[st], v), _mm_mul_ps(A1[st], y)), Z2[st]);
                Z2[st] = _mm_sub_ps(_mm_mul_ps(B2[st], v), _mm_mul_ps(A2[st], y));
                v = y;
            }
            x[t] = v;
        }
        _MM_TRANSPOSE4_PS(x[0], x[1], x[2], x[3]);
        _mm_storeu_ps(buf[0] + i, x[0]); _mm_storeu_ps(buf[1] + i, x[1]); _mm_storeu_ps(buf[2] + i, x[2]); _mm_storeu_ps(buf[3] + i, x[3]);
    }
    alignas(16) float z[2][2][4];
    for (int st = 0; st < 2; ++st) { _mm_store_ps(z[st][0], Z1[st]); _mm_store_ps(z[st][1], Z2[st]); }
    for (int k = 0; k < 4; ++k) {
        for (int st = 0; st < 2; ++st) { s[k][st].z1 = z[st][0][k]; s[k][st].z2 = z[st][1][k]; }
        if (i < n) filter_scalar(buf[k] + i, n - i, c[k], s[k]);       // tail
    }
}

// 8 outputs per step in the lanes (each lane sums its taps in the scalar order -> bit-exact), the input load is shared
// by the ears; the rest (n % 8) goes the scalar way.
inline void fir_stereo_store_simd(float* l, float* r, const float* x, const float* hl, const float* hr, uint32_t taps, uint32_t n)
{
    const float* base = x - (taps - 1);
    const uint32_t n8 = n & ~7u;
    for (uint32_t i = 0; i < n8; i += 8) {
        __m128 l0 = _mm_setzero_ps(), l1 = _mm_setzero_ps(), r0 = _mm_setzero_ps(), r1 = _mm_setzero_ps();
        for (uint32_t k = 0; k < taps; ++k) {
            const __m128 cl = _mm_set1_ps(hl[k]), cr = _mm_set1_ps(hr[k]);
            const __m128 x0 = _mm_loadu_ps(base + i + k), x1 = _mm_loadu_ps(base + i + k + 4);
            l0 = _mm_add_ps(l0, _mm_mul_ps(cl, x0)); l1 = _mm_add_ps(l1, _mm_mul_ps(cl, x1));
            r0 = _mm_add_ps(r0, _mm_mul_ps(cr, x0)); r1 = _mm_add_ps(r1, _mm_mul_ps(cr, x1));
        }
        _mm_storeu_ps(l + i, l0); _mm_storeu_ps(l + i + 4, l1);
        _mm_storeu_ps(r + i, r0); _mm_storeu_ps(r + i + 4, r1);
    }
    if (n8 < n) fir_stereo_store_scalar(l + n8, r + n8, x + n8, hl, hr, taps, n - n8);
}

// cvtps2dq rounds to nearest even like lrintf; min / max first, so packs never sees an out-of-range value
inline void mono_s16_store_simd(int16_t* dst, const float* l, const float* r, uint32_t n)
{
    const __m128 k = _mm_set1_ps(32768.0f), hi = _mm_set1_ps(32767.0f), lo = _mm_set1_ps(-32768.0f), half = _mm_set1_ps(0.5f);
    uint32_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m128 a = _mm_loadu_ps(l + i), b = _mm_loadu_ps(l + i + 4);
        if (r) { a = _mm_mul_ps(_mm_add_ps(a, _mm_loadu_ps(r + i)), half); b = _mm_mul_ps(_mm_add_ps(b, _mm_loadu_ps(r + i + 4)), half); }
        a = _mm_max_ps(_mm_min_ps(_mm_mul_ps(a, k), hi), lo);
        b = _mm_max_ps(_mm_min_ps(_mm_mul_ps(b, k), hi), lo);
        _mm_storeu_si128((__m128i*)(dst + i), _mm_packs_epi32(_mm_cvtps_epi32(a), _mm_cvtps_epi32(b)));
    }
    if (i < n) mono_s16_store_scalar(dst + i, l + i, r ? r + i : nullptr, n - i);
}
#endif

// ---------------------------------------------------------------------------
// What the mixer calls. Chosen by tests/audio_mix_test.cpp (keep SIMD only where it is faster).
// ---------------------------------------------------------------------------
#if AUDIO_MIX_SIMD
    #define AUDIO_MIX_PICK(name) name##_simd
#else
    #define AUDIO_MIX_PICK(name) name##_scalar
#endif

inline void     voice_s16(float* dst, const int16_t* src, uint32_t n, float g, float dg)                                     { AUDIO_MIX_PICK(voice_s16)(dst, src, n, g, dg); }
inline uint32_t voice_s16_resample(float* dst, const int16_t* src, uint32_t n, uint32_t pos, uint32_t step, float g, float dg) { return AUDIO_MIX_PICK(voice_s16_resample)(dst, src, n, pos, step, g, dg); }
inline void     pan_mono_stereo(float* l, float* r, const float* in, uint32_t n, float gl, float dgl, float gr, float dgr)    { AUDIO_MIX_PICK(pan_mono_stereo)(l, r, in, n, gl, dgl, gr, dgr); }
inline void     mix_f32(float* dst, const float* src, uint32_t n, float g, float dg)                                         { AUDIO_MIX_PICK(mix_f32)(dst, src, n, g, dg); }
inline void     output_stereo(float* out, const float* l, const float* r, uint32_t n, float g, float dg)                     { AUDIO_MIX_PICK(output_stereo)(out, l, r, n, g, dg); }
inline void     filter4(float* const buf[4], uint32_t n, const biquad c[4][2], biquad_state s[4][2])                        { AUDIO_MIX_PICK(filter4)(buf, n, c, s); }
inline void     filter(float* buf, uint32_t n, const biquad c[2], biquad_state s[2])                                        { filter_scalar(buf, n, c, s); }
inline void     fir_stereo_store(float* l, float* r, const float* x, const float* hl, const float* hr, uint32_t taps, uint32_t n) { AUDIO_MIX_PICK(fir_stereo_store)(l, r, x, hl, hr, taps, n); }
inline void     mono_s16_store(int16_t* dst, const float* l, const float* r, uint32_t n)                                    { AUDIO_MIX_PICK(mono_s16_store)(dst, l, r, n); }

} // namespace audio_mix

#endif
