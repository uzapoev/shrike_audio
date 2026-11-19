#ifndef __audio_adpcm_h__
#define __audio_adpcm_h__

// audio_adpcm.h - IMA ADPCM format of the sounds
//
// Role:      decode IMA ADPCM blocks and parse the WAV header (format 0x11, mono, fixed block size).
// Contents:
//   - expand / decode_block    one block -> int16 samples (every block starts with its own predictor state)
//   - block_first_sample       the first sample of a block, stored as is in its header (interpolation across blocks)
//   - clip / parse_wav         a view into a WAV in memory (or into a header-only prefix for streams)
//   - encode_block             int16 samples -> one block (the bank tool; the header stores the first sample as is)
// Thread:    any (pure functions).
// Not here:  file conversion and envelope baking (tools/audio_bank_tool.cpp).
// Used by:   audio_mixer.cpp (resident clips and stream rings), audio_stream.cpp (header), audio_bank.cpp (attach).
// Tests:     bit-exact with dr_wav (tests/audio_proto_test.cpp), through the mixer (tests/audio_mixer_test.cpp).

#include <stdint.h>
#include <string.h>

namespace audio_adpcm {

static const int k_index_table[16] = { -1, -1, -1, -1, 2, 4, 6, 8, -1, -1, -1, -1, 2, 4, 6, 8 };
static const int k_step_table[89] = {
    7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 19, 21, 23, 25, 28, 31, 34, 37, 41, 45, 50, 55, 60, 66, 73, 80, 88, 97, 107, 118,
    130, 143, 157, 173, 190, 209, 230, 253, 279, 307, 337, 371, 408, 449, 494, 544, 598, 658, 724, 796, 876, 963, 1060,
    1166, 1282, 1411, 1552, 1707, 1878, 2066, 2272, 2499, 2749, 3024, 3327, 3660, 4026, 4428, 4871, 5358, 5894, 6484,
    7132, 7845, 8630, 9493, 10442, 11487, 12635, 13899, 15289, 16818, 18500, 20350, 22385, 24623, 27086, 29794, 32767 };

inline int expand(int nibble, int& pred, int& index)
{
    int step = k_step_table[index];
    int diff = step >> 3;
    if (nibble & 1) diff += step >> 2;
    if (nibble & 2) diff += step >> 1;
    if (nibble & 4) diff += step;
    if (nibble & 8) diff = -diff;
    pred += diff;
    if (pred > 32767) pred = 32767; else if (pred < -32768) pred = -32768;
    index += k_index_table[nibble];
    if (index < 0) index = 0; else if (index > 88) index = 88;
    return pred;
}

// The first sample of a block is stored as is in its header: the mixer uses it as the "next sample" for interpolation.
inline int16_t block_first_sample(const uint8_t* block) { return int16_t(block[0] | (block[1] << 8)); }

inline void decode_block(const uint8_t* p, uint32_t samples_per_block, int16_t* out)
{
    int pred = int16_t(p[0] | (p[1] << 8)), index = p[2] > 88 ? 88 : p[2];
    out[0] = int16_t(pred);
    for (uint32_t i = 1, n = 4; i < samples_per_block; i += 2, ++n) {
        out[i] = int16_t(expand(p[n] & 15, pred, index));
        if (i + 1 < samples_per_block) out[i + 1] = int16_t(expand(p[n] >> 4, pred, index));
    }
}

inline int quantize(int sample, int pred, int index)
{
    int step = k_step_table[index], diff = sample - pred, nibble = 0;
    if (diff < 0) { nibble = 8; diff = -diff; }
    if (diff >= step) { nibble |= 4; diff -= step; } step >>= 1;
    if (diff >= step) { nibble |= 2; diff -= step; } step >>= 1;
    if (diff >= step) { nibble |= 1; }
    return nibble;
}

// Encodes `count` samples (<= samples_per_block, the rest is silence) into one block of (samples_per_block - 1) / 2 + 4
// bytes. `index` carries the step index from block to block (the decoder reads it from the header).
inline void encode_block(const int16_t* pcm, uint32_t count, uint32_t samples_per_block, int& index, uint8_t* out)
{
    auto at = [&](uint32_t i) { return i < count ? int(pcm[i]) : 0; };
    int pred = at(0);
    out[0] = uint8_t(pred & 0xff); out[1] = uint8_t((pred >> 8) & 0xff); out[2] = uint8_t(index); out[3] = 0;
    for (uint32_t i = 1, n = 4; i < samples_per_block; i += 2, ++n) {
        int n0 = quantize(at(i), pred, index);     expand(n0, pred, index);
        int n1 = quantize(at(i + 1), pred, index); expand(n1, pred, index);
        out[n] = uint8_t(n0 | (n1 << 4));
    }
}

// A view into a WAV in memory; nothing is unpacked.
struct clip
{
    const uint8_t*  blocks              = nullptr;
    uint32_t        block_count         = 0;
    uint32_t        block_align         = 0;
    uint32_t        samples_per_block   = 0;
    uint32_t        sample_rate         = 0;
    uint64_t        frames              = 0;
};

// Parses the header. Works on a header-only prefix too (then `blocks` points past the buffer and only
// the offset blocks - data is meaningful; block_count comes from the data chunk size).
inline bool parse_wav(const uint8_t* w, size_t size, clip& c)
{
    auto rd16 = [&](size_t o) { return uint32_t(w[o] | (w[o + 1] << 8)); };
    auto rd32 = [&](size_t o) { return rd16(o) | (rd16(o + 2) << 16); };
    if (size < 12 || memcmp(w, "RIFF", 4) || memcmp(w + 8, "WAVE", 4)) return false;
    c = clip();
    for (size_t o = 12; o + 8 <= size;) {
        uint32_t sz = rd32(o + 4);
        if (!memcmp(w + o, "fmt ", 4) && o + 28 <= size) {
            if (rd16(o + 8) != 0x11 || rd16(o + 10) != 1) return false;      // IMA ADPCM, mono
            c.sample_rate       = rd32(o + 12);
            c.block_align       = rd16(o + 20);
            c.samples_per_block = rd16(o + 26);
        } else if (!memcmp(w + o, "fact", 4) && o + 12 <= size) {
            c.frames = rd32(o + 8);
        } else if (!memcmp(w + o, "data", 4)) {
            c.blocks = w + o + 8;
            c.block_count = c.block_align ? sz / c.block_align : 0;
            break;
        }
        o += 8 + sz + (sz & 1);
    }
    if (!c.blocks || !c.block_align || !c.samples_per_block) return false;
    if (!c.frames) c.frames = uint64_t(c.block_count) * c.samples_per_block;
    return true;
}

} // namespace audio_adpcm

#endif
