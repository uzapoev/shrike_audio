# stb_vorbis_patch.py - makes external/stb/stb_vorbis.c (patched copy) from upstream stb_vorbis.c
#
# Role:      the only way external/stb/stb_vorbis.c is produced: every engine change to stb_vorbis lives here as an exact
#            text substitution (each fails loudly if upstream changed around it).
# Contents:  SIMD (SSE2 / sse2neon) for the IMDCT steps 1, 2, 3 (+ ld654), 7, 8, the overlap-add and the inverse
#            coupling, with the scalar code kept behind the run-time switch stb_vorbis_simd; IO callbacks
#            (STB_VORBIS_IO_CALLBACKS: stb_vorbis_open_io, read_at + length, 4 KB buffer); a header note.
# Not here:  how the engine uses the decoder (src/audio/audio_vorbis.h).
# Usage:     python stb_vorbis_patch.py <upstream stb_vorbis.c> <out: external/stb/stb_vorbis.c>
#            upstream: https://raw.githubusercontent.com/nothings/stb/master/stb_vorbis.c (v1.25,
#            sha1 9a29ad19251f29dd4adae5428912278d9679b815 is the one this script was written against).
# Tests:     tests/audio_vorbis_test.cpp (SIMD == scalar bit for bit, open_io == open_memory, seeking).

import sys
src, dst = sys.argv[1], sys.argv[2]
s = open(src, encoding="utf-8").read()

def sub(a, b):
    global s
    assert s.count(a) == 1, a[:80]
    s = s.replace(a, b, 1)

SIMD_HDR = r'''
// [engine] SIMD (SSE2; ARM through sse2neon). Same arithmetic order as the scalar code, no FMA -> bit-exact.
//          stb_vorbis_simd = 0 runs the original scalar code (tests compare both).
#ifndef STB_VORBIS_NO_SIMD
   #if defined(_M_X64) || defined(__SSE2__) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
      #include <emmintrin.h>
      #define STB_VORBIS_SIMD 1
   #elif defined(__ARM_NEON) || defined(__ARM_NEON__) || defined(_M_ARM64)
      #include "sse2neon/sse2neon.h"
      #define STB_VORBIS_SIMD 1
   #endif
#endif
#ifndef STB_VORBIS_SIMD
   #define STB_VORBIS_SIMD 0
#endif
int stb_vorbis_simd = STB_VORBIS_SIMD;
'''

# right before the step3 helpers
sub("// the following were split out into separate functions while optimizing;",
    SIMD_HDR + r'''
#if STB_VORBIS_SIMD
// 4 complex butterflies at e0[0..-7] / e2[0..-7]: e0 += e2, e2 = (e0 - e2) rotated by pair p's twiddle (a0[p], a1[p]).
// Lanes in memory order: hi = [-3,-2,-1,0] holds pairs 1 (lanes 0,1) and 0 (lanes 2,3), lo = [-7..-4] pairs 3 and 2.
//   even slot (k00): k00*A0 - k01*A1   odd slot (k01): k01*A0 + k00*A1
//   = D*W0 + swap(D)*W1, W0 = [A0q,A0q,A0p,A0p], W1 = [A1q,-A1q,A1p,-A1p]   (x + (-y) == x - y exactly)
static __forceinline void imdct_bfly8_sse(float *e0, float *e2, __m128 w0h, __m128 w1h, __m128 w0l, __m128 w1l)
{
   __m128 a = _mm_loadu_ps(e0 - 3), b = _mm_loadu_ps(e2 - 3);
   __m128 c = _mm_loadu_ps(e0 - 7), d = _mm_loadu_ps(e2 - 7);
   __m128 dh = _mm_sub_ps(a, b), dl = _mm_sub_ps(c, d);
   _mm_storeu_ps(e0 - 3, _mm_add_ps(a, b));
   _mm_storeu_ps(e0 - 7, _mm_add_ps(c, d));
   __m128 sh = _mm_shuffle_ps(dh, dh, _MM_SHUFFLE(2,3,0,1)), sl = _mm_shuffle_ps(dl, dl, _MM_SHUFFLE(2,3,0,1));
   _mm_storeu_ps(e2 - 3, _mm_add_ps(_mm_mul_ps(dh, w0h), _mm_mul_ps(sh, w1h)));
   _mm_storeu_ps(e2 - 7, _mm_add_ps(_mm_mul_ps(dl, w0l), _mm_mul_ps(sl, w1l)));
}
// twiddle pairs p = (p[0], p[1]) and q: [q0,q1,p0,p1] by two 64-bit loads -> W0 = [q0,q0,p0,p0], W1 = [q1,-q1,p1,-p1]
static __forceinline __m128 imdct_pair2(const float *p, const float *q)
{
   return _mm_loadh_pi(_mm_castpd_ps(_mm_load_sd((const double *)q)), (const __m64 *)p);
}
#define IMDCT_NEG13 _mm_castsi128_ps(_mm_set_epi32((int)0x80000000, 0, (int)0x80000000, 0))
#define IMDCT_W0(P) _mm_shuffle_ps(P, P, _MM_SHUFFLE(2,2,0,0))
#define IMDCT_W1(P) _mm_xor_ps(_mm_shuffle_ps(P, P, _MM_SHUFFLE(3,3,1,1)), IMDCT_NEG13)

static void imdct_step3_r_loop_sse(int lim, float *e0, float *e2, float *A, int k1)
{
   int i;
   for (i = lim >> 2; i > 0; --i) {
      __m128 ph = imdct_pair2(A, A + k1), pl = imdct_pair2(A + 2*k1, A + 3*k1);
      imdct_bfly8_sse(e0, e2, IMDCT_W0(ph), IMDCT_W1(ph), IMDCT_W0(pl), IMDCT_W1(pl));
      A += 4*k1; e0 -= 8; e2 -= 8;
   }
}

#define SSE_REV(x)      _mm_shuffle_ps(x, x, _MM_SHUFFLE(0,1,2,3))
#define SSE_EVEN(a, b)  _mm_shuffle_ps(a, b, _MM_SHUFFLE(2,0,2,0))
#define SSE_ODD(a, b)   _mm_shuffle_ps(a, b, _MM_SHUFFLE(3,1,3,1))
#define SSE_SIGN        _mm_castsi128_ps(_mm_set1_epi32((int)0x80000000))
#endif

// the following were split out into separate functions while optimizing;''')

sub('''   float *ee0 = e + i_off;
   float *ee2 = ee0 + k_off;
   int i;

   assert((n & 3) == 0);
''', '''   float *ee0 = e + i_off;
   float *ee2 = ee0 + k_off;
   int i;

   assert((n & 3) == 0);
   #if STB_VORBIS_SIMD
   if (stb_vorbis_simd) { imdct_step3_r_loop_sse(n, ee0, ee2, A, 8); return; }
   #endif
''')

sub('''   float *e0 = e + d0;
   float *e2 = e0 + k_off;

   for (i=lim >> 2; i > 0; --i) {''', '''   float *e0 = e + d0;
   float *e2 = e0 + k_off;

   #if STB_VORBIS_SIMD
   if (stb_vorbis_simd) { imdct_step3_r_loop_sse(lim, e0, e2, A, k1); return; }
   #endif
   for (i=lim >> 2; i > 0; --i) {''')

sub('''   float *ee0 = e  +i_off;
   float *ee2 = ee0+k_off;

   for (i=n; i > 0; --i) {''', '''   float *ee0 = e  +i_off;
   float *ee2 = ee0+k_off;

   #if STB_VORBIS_SIMD
   if (stb_vorbis_simd) {
      float *P0 = A, *P1 = A + a_off, *P2 = A + a_off*2, *P3 = A + a_off*3;
      __m128 ph = imdct_pair2(P0, P1), pl = imdct_pair2(P2, P3);
      __m128 w0h = IMDCT_W0(ph), w1h = IMDCT_W1(ph), w0l = IMDCT_W0(pl), w1l = IMDCT_W1(pl);
      for (i=n; i > 0; --i) { imdct_bfly8_sse(ee0, ee2, w0h, w1h, w0l, w1l); ee0 -= k0; ee2 -= k0; }
      (void)A0; (void)A1; (void)A2; (void)A3; (void)A4; (void)A5; (void)A6; (void)A7;
      return;
   }
   #endif
   for (i=n; i > 0; --i) {''')

# step 2
sub('''      d0 = &u[n4];
      d1 = &u[0];

      while (AA >= A) {''', '''      d0 = &u[n4];
      d1 = &u[0];

      #if STB_VORBIS_SIMD
      // pair (0,1) by AA[4],AA[5], pair (2,3) by AA[0],AA[1]: out = D*[A4,A4,A0,A0] + swap(D)*[A5,-A5,A1,-A1]
      if (stb_vorbis_simd) {
         while (AA >= A) {
            __m128 x = _mm_loadu_ps(e0), y = _mm_loadu_ps(e1), dd = _mm_sub_ps(x, y);
            _mm_storeu_ps(d0, _mm_add_ps(x, y));
            __m128 sw = _mm_shuffle_ps(dd, dd, _MM_SHUFFLE(2,3,0,1));
            __m128 w0 = _mm_set_ps(AA[0], AA[0], AA[4], AA[4]), w1 = _mm_set_ps(-AA[1], AA[1], -AA[5], AA[5]);
            _mm_storeu_ps(d1, _mm_add_ps(_mm_mul_ps(dd, w0), _mm_mul_ps(sw, w1)));
            AA -= 8; d0 += 4; d1 += 4; e0 += 4; e1 += 4;
         }
      }
      #endif
      while (AA >= A) {''')

# step 1 (copy and reflect + step 0): 4 scalar iterations per SIMD one
sub('''      d = &buf2[n2-2];
      AA = A;
      e = &buffer[0];
      e_stop = &buffer[n2];
      while (e != e_stop) {''', '''      d = &buf2[n2-2];
      AA = A;
      e = &buffer[0];
      e_stop = &buffer[n2];
      #if STB_VORBIS_SIMD
      // iteration k: d[1] = X*A0 - Y*A1, d[0] = X*A1 + Y*A0 with X = e[4k], Y = e[4k+2]; d goes down by 2 per k
      if (stb_vorbis_simd) {
         while (e != e_stop) {
            __m128 t0 = SSE_EVEN(_mm_loadu_ps(e), _mm_loadu_ps(e + 4)), t1 = SSE_EVEN(_mm_loadu_ps(e + 8), _mm_loadu_ps(e + 12));
            __m128 X = SSE_EVEN(t0, t1), Y = SSE_ODD(t0, t1);
            __m128 q0 = _mm_loadu_ps(AA), q1 = _mm_loadu_ps(AA + 4);
            __m128 A0 = SSE_EVEN(q0, q1), A1 = SSE_ODD(q0, q1);
            __m128 o1 = SSE_REV(_mm_sub_ps(_mm_mul_ps(X, A0), _mm_mul_ps(Y, A1)));
            __m128 o0 = SSE_REV(_mm_add_ps(_mm_mul_ps(X, A1), _mm_mul_ps(Y, A0)));
            _mm_storeu_ps(d - 6, _mm_unpacklo_ps(o0, o1));
            _mm_storeu_ps(d - 2, _mm_unpackhi_ps(o0, o1));
            d -= 8; AA += 8; e += 16;
         }
         // second half: X = -e[2], Y = -e[0], e going down by 4 from n2-3
         e = &buffer[n2-3];
         while (d >= buf2) {
            __m128 t0 = SSE_ODD(_mm_loadu_ps(e - 13), _mm_loadu_ps(e - 9)), t1 = SSE_ODD(_mm_loadu_ps(e - 5), _mm_loadu_ps(e - 1));
            __m128 Y = _mm_xor_ps(SSE_REV(SSE_EVEN(t0, t1)), SSE_SIGN), X = _mm_xor_ps(SSE_REV(SSE_ODD(t0, t1)), SSE_SIGN);
            __m128 q0 = _mm_loadu_ps(AA), q1 = _mm_loadu_ps(AA + 4);
            __m128 A0 = SSE_EVEN(q0, q1), A1 = SSE_ODD(q0, q1);
            __m128 o1 = SSE_REV(_mm_sub_ps(_mm_mul_ps(X, A0), _mm_mul_ps(Y, A1)));
            __m128 o0 = SSE_REV(_mm_add_ps(_mm_mul_ps(X, A1), _mm_mul_ps(Y, A0)));
            _mm_storeu_ps(d - 6, _mm_unpacklo_ps(o0, o1));
            _mm_storeu_ps(d - 2, _mm_unpackhi_ps(o0, o1));
            d -= 8; AA += 8; e -= 16;
         }
         e = e_stop;      // both halves done: the scalar loops below see nothing left
      }
      #endif
      while (e != e_stop) {''')

# step 7: 2 scalar iterations (4 pairs) per SIMD one
sub('''      d = v;
      e = v + n2 - 4;

      while (d < e) {''', '''      d = v;
      e = v + n2 - 4;

      #if STB_VORBIS_SIMD
      // pair j of d pairs with e pair (2-2j, 3-2j), twiddle C[2j], C[2j+1]
      if (stb_vorbis_simd) {
         while (d < e) {
            __m128 d0 = _mm_loadu_ps(d), d1 = _mm_loadu_ps(d + 4);
            __m128 e0 = _mm_loadu_ps(e), e1 = _mm_loadu_ps(e - 4);         // pairs (e2,e3),(e0,e1) then (e-2,e-1),(e-4,e-3)
            __m128 er0 = _mm_shuffle_ps(e0, e0, _MM_SHUFFLE(1,0,3,2)), er1 = _mm_shuffle_ps(e1, e1, _MM_SHUFFLE(1,0,3,2));
            __m128 x = SSE_EVEN(d0, d1), y = SSE_ODD(d0, d1), ex = SSE_EVEN(er0, er1), ey = SSE_ODD(er0, er1);
            __m128 c0 = _mm_loadu_ps(C), c1 = _mm_loadu_ps(C + 4);
            __m128 C0 = SSE_EVEN(c0, c1), C1 = SSE_ODD(c0, c1);
            __m128 a02 = _mm_sub_ps(x, ex), a11 = _mm_add_ps(y, ey);
            __m128 b0 = _mm_add_ps(_mm_mul_ps(C1, a02), _mm_mul_ps(C0, a11));
            __m128 b1 = _mm_sub_ps(_mm_mul_ps(C1, a11), _mm_mul_ps(C0, a02));
            __m128 b2 = _mm_add_ps(x, ex), b3 = _mm_sub_ps(y, ey);
            __m128 dx = _mm_add_ps(b2, b0), dy = _mm_add_ps(b3, b1), nx = _mm_sub_ps(b2, b0), ny = _mm_sub_ps(b1, b3);
            _mm_storeu_ps(d,     _mm_unpacklo_ps(dx, dy));
            _mm_storeu_ps(d + 4, _mm_unpackhi_ps(dx, dy));
            __m128 n0 = _mm_unpacklo_ps(nx, ny), n1 = _mm_unpackhi_ps(nx, ny);   // pairs for e (2,3),(0,1) / (-2,-1),(-4,-3)
            _mm_storeu_ps(e,     _mm_shuffle_ps(n0, n0, _MM_SHUFFLE(1,0,3,2)));
            _mm_storeu_ps(e - 4, _mm_shuffle_ps(n1, n1, _MM_SHUFFLE(1,0,3,2)));
            C += 8; d += 8; e -= 8;
         }
      }
      #endif
      while (d < e) {''')

# step 8: one SIMD iteration per scalar one
sub('''      d3 = &buffer[n-4];
      while (e >= v) {''', '''      d3 = &buffer[n-4];
      #if STB_VORBIS_SIMD
      // pair k = (e[2k], e[2k+1]): pa = e0*B1 - e1*B0, pb = -e0*B0 - e1*B1; d0 = rev(pa), d1 = -pa, d2 = rev(pb), d3 = pb
      if (stb_vorbis_simd) {
         while (e >= v) {
            __m128 ea = _mm_loadu_ps(e), eb = _mm_loadu_ps(e + 4), ba = _mm_loadu_ps(B), bb = _mm_loadu_ps(B + 4);
            __m128 ex = SSE_EVEN(ea, eb), ey = SSE_ODD(ea, eb), bx = SSE_EVEN(ba, bb), by = SSE_ODD(ba, bb);
            __m128 pa = _mm_sub_ps(_mm_mul_ps(ex, by), _mm_mul_ps(ey, bx));
            __m128 pb = _mm_sub_ps(_mm_xor_ps(_mm_mul_ps(ex, bx), SSE_SIGN), _mm_mul_ps(ey, by));
            _mm_storeu_ps(d0, SSE_REV(pa));
            _mm_storeu_ps(d1, _mm_xor_ps(pa, SSE_SIGN));
            _mm_storeu_ps(d2, SSE_REV(pb));
            _mm_storeu_ps(d3, pb);
            B -= 8; e -= 8; d0 += 4; d2 += 4; d1 -= 4; d3 -= 4;
         }
      }
      #endif
      while (e >= v) {''')

# steps ld-6..-4 + iter_54: 16 floats per iteration, kept in registers
sub('''   float *z = e + i_off;
   float *base = z - 16 * n;

   while (z > base) {''', '''   float *z = e + i_off;
   float *base = z - 16 * n;

   #if STB_VORBIS_SIMD
   if (stb_vorbis_simd) {
      const __m128 sign = SSE_SIGN, zero = _mm_setzero_ps();
      const __m128 lo2 = _mm_castsi128_ps(_mm_set_epi32(0, 0, -1, -1));                      // lanes 0,1
      const __m128 s1  = _mm_castsi128_ps(_mm_set_epi32(0, 0, 0, (int)0x80000000));          // lane 0
      const __m128 s2  = _mm_castsi128_ps(_mm_set_epi32(0, 0, (int)0x80000000, 0));          // lane 1
      const __m128 s4  = _mm_castsi128_ps(_mm_set_epi32(0, (int)0x80000000, 0, 0));          // lane 2
      const __m128 s12 = _mm_castsi128_ps(_mm_set_epi32(0, (int)0x80000000, (int)0x80000000, 0));   // lanes 1,2
      const __m128 a2a = _mm_set_ps(1.0f, 1.0f, A2, A2), a2b = _mm_set_ps(1.0f, 1.0f, A2, -A2);
      (void)zero; (void)sign;
      while (z > base) {
         __m128 h1 = _mm_loadu_ps(z - 3), l1 = _mm_loadu_ps(z - 11), h2 = _mm_loadu_ps(z - 7), l2 = _mm_loadu_ps(z - 15);
         // z[0..-3] += z[-8..-11]; D = [l11, l00, k11, k00]; z[-11..-8] = [(l11-l00)*A2, (l00+l11)*A2, k11, k00]
         __m128 d1 = _mm_sub_ps(h1, l1), x1 = _mm_shuffle_ps(d1, d1, _MM_SHUFFLE(2,3,0,1));
         __m128 t1 = _mm_mul_ps(_mm_add_ps(d1, _mm_xor_ps(x1, s1)), a2a);
         __m128 n1 = _mm_or_ps(_mm_and_ps(lo2, t1), _mm_andnot_ps(lo2, d1));
         // z[-4..-7] += z[-12..-15]; z[-15..-12] = [(l00+l11)*-A2, (l11-l00)*A2, -k00, k11]
         __m128 d2 = _mm_sub_ps(h2, l2), x2 = _mm_shuffle_ps(d2, d2, _MM_SHUFFLE(2,3,0,1));
         __m128 t2 = _mm_mul_ps(_mm_add_ps(x2, _mm_xor_ps(d2, s2)), a2b);
         __m128 n2 = _mm_or_ps(_mm_and_ps(lo2, t2), _mm_andnot_ps(lo2, _mm_xor_ps(x2, s4)));
         h1 = _mm_add_ps(h1, l1); h2 = _mm_add_ps(h2, l2);
         // iter_54 on (h1, h2) and on (n1, n2): S = H + L = [y3,y2,y1,y0], D = H - L = [k33,k22,k11,k00]
         {
            __m128 S = _mm_add_ps(h1, h2), D = _mm_sub_ps(h1, h2), Sx = _mm_shuffle_ps(S, S, _MM_SHUFFLE(1,0,3,2));
            __m128 P = _mm_shuffle_ps(D, D, _MM_SHUFFLE(3,2,3,2)), Q = _mm_shuffle_ps(D, D, _MM_SHUFFLE(0,1,0,1));
            _mm_storeu_ps(z - 3, _mm_shuffle_ps(_mm_sub_ps(Sx, S), _mm_add_ps(S, Sx), _MM_SHUFFLE(3,2,1,0)));
            _mm_storeu_ps(z - 7, _mm_add_ps(P, _mm_xor_ps(Q, s12)));
         }
         {
            __m128 S = _mm_add_ps(n1, n2), D = _mm_sub_ps(n1, n2), Sx = _mm_shuffle_ps(S, S, _MM_SHUFFLE(1,0,3,2));
            __m128 P = _mm_shuffle_ps(D, D, _MM_SHUFFLE(3,2,3,2)), Q = _mm_shuffle_ps(D, D, _MM_SHUFFLE(0,1,0,1));
            _mm_storeu_ps(z - 11, _mm_shuffle_ps(_mm_sub_ps(Sx, S), _mm_add_ps(S, Sx), _MM_SHUFFLE(3,2,1,0)));
            _mm_storeu_ps(z - 15, _mm_add_ps(P, _mm_xor_ps(Q, s12)));
         }
         z -= 16;
      }
      return;
   }
   #endif
   while (z > base) {''')

# overlap-add with the previous window
sub('''      for (i=0; i < f->channels; ++i) {
         for (j=0; j < n; ++j)
            f->channel_buffers[i][left+j] =''', '''      for (i=0; i < f->channels; ++i) {
         j = 0;
         #if STB_VORBIS_SIMD
         if (stb_vorbis_simd) {
            float *cb = f->channel_buffers[i] + left, *pw = f->previous_window[i];
            for (; j + 4 <= n; j += 4)
               _mm_storeu_ps(cb + j, _mm_add_ps(_mm_mul_ps(_mm_loadu_ps(cb + j), _mm_loadu_ps(w + j)),
                                                _mm_mul_ps(_mm_loadu_ps(pw + j), SSE_REV(_mm_loadu_ps(w + n - 4 - j)))));
         }
         #endif
         for (; j < n; ++j)
            f->channel_buffers[i][left+j] =''')

# inverse coupling: r = m - a if (m > 0) == (a > 0) else m + a; a > 0 -> (m, r) else (r, m)
sub('''      float *a1 = f->channel_buffers[map->chan[i].angle    ];
      for (j=0; j < n2; ++j) {''', '''      float *a1 = f->channel_buffers[map->chan[i].angle    ];
      j = 0;
      #if STB_VORBIS_SIMD
      if (stb_vorbis_simd) {
         __m128 z = _mm_setzero_ps();
         for (; j + 4 <= n2; j += 4) {
            __m128 m = _mm_loadu_ps(m1 + j), a = _mm_loadu_ps(a1 + j);
            __m128 mp = _mm_cmpgt_ps(m, z), ap = _mm_cmpgt_ps(a, z);
            __m128 same = _mm_xor_ps(_mm_xor_ps(mp, ap), _mm_castsi128_ps(_mm_set1_epi32(-1)));
            __m128 r = _mm_add_ps(m, _mm_xor_ps(a, _mm_and_ps(same, SSE_SIGN)));      // m + (-a) == m - a
            _mm_storeu_ps(m1 + j, _mm_or_ps(_mm_and_ps(ap, m), _mm_andnot_ps(ap, r)));
            _mm_storeu_ps(a1 + j, _mm_or_ps(_mm_and_ps(ap, r), _mm_andnot_ps(ap, m)));
         }
      }
      #endif
      for (; j < n2; ++j) {''')

# ---------------------------------------------------------------------------------------------------------------------
# IO callbacks: read_at + file length instead of FILE* (packages, Android assets); 4 KB buffer inside the decoder.
# ---------------------------------------------------------------------------------------------------------------------
sub('''extern stb_vorbis * stb_vorbis_open_memory(const unsigned char *data, int len,
                                  int *error, const stb_vorbis_alloc *alloc_buffer);''',
'''extern stb_vorbis * stb_vorbis_open_memory(const unsigned char *data, int len,
                                  int *error, const stb_vorbis_alloc *alloc_buffer);

// [engine] a file read through a callback: read_at(user, offset, dst, size) returns the bytes read (0: end / error).
// Seeking works (the decoder only needs random access + the length). Files up to 4 GB.
#ifdef STB_VORBIS_IO_CALLBACKS
typedef struct
{
   void *user;
   size_t (*read_at)(void *user, stb_vorbis_uint64 offset, void *dst, size_t size);
   stb_vorbis_uint64 length;
} stb_vorbis_io;

extern stb_vorbis * stb_vorbis_open_io(const stb_vorbis_io *io, int *error, const stb_vorbis_alloc *alloc_buffer);
#endif

// [engine] 1: SIMD kernels (default where SSE2 / NEON exist), 0: the original scalar code. Same output either way.
extern int stb_vorbis_simd;''')

sub('''#ifndef STB_VORBIS_NO_STDIO
#include <stdio.h>
#endif

#ifdef __cplusplus''', '''#ifndef STB_VORBIS_NO_STDIO
#include <stdio.h>
#endif
#include <stddef.h>

#ifdef __cplusplus''')

sub('''   uint32 stream_len;

   uint8  push_mode;''', '''   uint32 stream_len;

#ifdef STB_VORBIS_IO_CALLBACKS
   // [engine] callback source (io.read_at == NULL: memory / stdio)
   stb_vorbis_io io;
   stb_vorbis_uint64 io_pos, io_buf_pos;      // next byte to read; file offset of io_buf[0]
   uint32 io_buf_len;
   uint8 io_buf[4096];
#endif

   uint8  push_mode;''')

sub('''#if defined(STB_VORBIS_NO_STDIO)
   #define USE_MEMORY(z)    TRUE
#else''', '''#if defined(STB_VORBIS_NO_STDIO) && !defined(STB_VORBIS_IO_CALLBACKS)
   #define USE_MEMORY(z)    TRUE
#else''')

sub('''static uint8 get8(vorb *z)
{
   if (USE_MEMORY(z)) {
      if (z->stream >= z->stream_end) { z->eof = TRUE; return 0; }
      return *z->stream++;
   }
''', '''#ifdef STB_VORBIS_IO_CALLBACKS
// [engine] refill the buffer at io_pos; 0 at the end of the file or on a read error
static int io_fill(vorb *z)
{
   size_t n;
   if (z->io_pos >= z->io.length) return 0;
   n = z->io.length - z->io_pos < sizeof(z->io_buf) ? (size_t) (z->io.length - z->io_pos) : sizeof(z->io_buf);
   z->io_buf_pos = z->io_pos;
   z->io_buf_len = (uint32) z->io.read_at(z->io.user, z->io_pos, z->io_buf, n);
   return z->io_buf_len > 0;
}
#endif

static uint8 get8(vorb *z)
{
   if (USE_MEMORY(z)) {
      if (z->stream >= z->stream_end) { z->eof = TRUE; return 0; }
      return *z->stream++;
   }
   #ifdef STB_VORBIS_IO_CALLBACKS
   if (z->io.read_at) {
      if (z->io_pos - z->io_buf_pos >= z->io_buf_len && !io_fill(z)) { z->eof = TRUE; return 0; }
      return z->io_buf[z->io_pos++ - z->io_buf_pos];
   }
   #ifdef STB_VORBIS_NO_STDIO
   z->eof = TRUE; return 0;               // no source
   #endif
   #endif
''')

sub('''      memcpy(data, z->stream, n);
      z->stream += n;
      return 1;
   }
''', '''      memcpy(data, z->stream, n);
      z->stream += n;
      return 1;
   }
   #ifdef STB_VORBIS_IO_CALLBACKS
   if (z->io.read_at) {
      while (n > 0) {
         stb_vorbis_uint64 at = z->io_pos - z->io_buf_pos;
         if (at < z->io_buf_len) {
            int k = (int) (z->io_buf_len - at) < n ? (int) (z->io_buf_len - at) : n;
            memcpy(data, z->io_buf + at, k);
            data += k; n -= k; z->io_pos += k;
         } else if (n >= (int) sizeof(z->io_buf)) {
            if (z->io.read_at(z->io.user, z->io_pos, data, n) != (size_t) n) { z->eof = 1; return 0; }
            z->io_pos += n;
            return 1;
         } else if (!io_fill(z)) {
            z->eof = 1; return 0;
         }
      }
      return 1;
   }
   #ifdef STB_VORBIS_NO_STDIO
   z->eof = 1; return 0;                  // no source
   #endif
   #endif
''')

sub('''      z->stream += n;
      if (z->stream >= z->stream_end) z->eof = 1;
      return;
   }
''', '''      z->stream += n;
      if (z->stream >= z->stream_end) z->eof = 1;
      return;
   }
   #ifdef STB_VORBIS_IO_CALLBACKS
   if (z->io.read_at) {
      z->io_pos += n;
      if (z->io_pos >= z->io.length) z->eof = 1;
      return;
   }
   #endif
''')

sub('''         f->stream = f->stream_start + loc;
         return 1;
      }
   }
''', '''         f->stream = f->stream_start + loc;
         return 1;
      }
   }
   #ifdef STB_VORBIS_IO_CALLBACKS
   if (f->io.read_at) {
      if (loc < 0 || (stb_vorbis_uint64) loc >= f->io.length) { f->io_pos = f->io.length; f->eof = 1; return 0; }
      f->io_pos = (stb_vorbis_uint64) loc;
      return 1;
   }
   #ifdef STB_VORBIS_NO_STDIO
   f->eof = 1; return 0;                  // no source
   #endif
   #endif
''')

sub('''   if (USE_MEMORY(f)) return (unsigned int) (f->stream - f->stream_start);
''', '''   if (USE_MEMORY(f)) return (unsigned int) (f->stream - f->stream_start);
   #ifdef STB_VORBIS_IO_CALLBACKS
   if (f->io.read_at) return (unsigned int) f->io_pos;
   #ifdef STB_VORBIS_NO_STDIO
   return 0;                              // no source
   #endif
   #endif
''')

sub('''// returns number of samples decoded, so int not int64''', '''#ifdef STB_VORBIS_IO_CALLBACKS
stb_vorbis * stb_vorbis_open_io(const stb_vorbis_io *io, int *error, const stb_vorbis_alloc *alloc)
{
   stb_vorbis *f, p;
   if (!io || !io->read_at || io->length == 0 || io->length > 0xffffffffu) {
      if (error) *error = VORBIS_unexpected_eof;
      return NULL;
   }
   vorbis_init(&p, alloc);
   p.io = *io;
   p.stream_len = (uint32) io->length;
   p.push_mode = FALSE;
   if (start_decoder(&p)) {
      f = vorbis_alloc(&p);
      if (f) {
         *f = p;
         vorbis_pump_first_frame(f);
         if (error) *error = VORBIS__no_error;
         return f;
      }
   }
   if (error) *error = p.error;
   vorbis_deinit(&p);
   return NULL;
}
#endif

// returns number of samples decoded, so int not int64''')

# what this copy is
sub('''// Ogg Vorbis audio decoder - v1.25 - public domain
''', '''// Ogg Vorbis audio decoder - v1.25 - public domain
//
// [engine] PATCHED COPY. Generated by src/audio/tools/stb_vorbis_patch.py from upstream stb_vorbis.c v1.25
// (github.com/nothings/stb); regenerate instead of editing by hand. Every change is marked "[engine]":
//   - SIMD (SSE2; ARM through sse2neon) for the IMDCT (steps 1, 2, 3, 7, 8), the overlap-add with the previous
//     window and the inverse channel coupling. Same arithmetic order as the scalar code (no FMA), so the output is
//     bit-exact with it; stb_vorbis_simd = 0 switches back to the scalar code at run time (tests compare both).
//     STB_VORBIS_NO_SIMD compiles it out.
//   - STB_VORBIS_IO_CALLBACKS: stb_vorbis_open_io() reads through read_at(offset) with a 4 KB buffer in the decoder
//     (seeking works), for file systems that are not stdio.
''')

open(dst, "w", encoding="utf-8").write(s)
print("patched")
