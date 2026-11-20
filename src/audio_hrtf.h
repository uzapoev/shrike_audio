#ifndef __audio_hrtf_h__
#define __audio_hrtf_h__

// audio_hrtf.h - built-in binaural spatial plugin (HRTF)
//
// Role:      the default spatial plugin when hrtf.bin is next to init.json: the instances the core marks binaural
//            (audio_spatial_input::binaural - the N most audible spatial ones, headphones mode on) go through a pair of
//            head-related impulse responses; the rest are panned (equal power, as the built-in pan plugin).
// Contents:
//   - audio_hrtf_data              hrtf.bin parsed: directions (listener space), per-ear delays (ITD), int16 taps;
//                                  interpolate(): the 3 nearest measured directions, weights by angle -> one HRIR pair
//   - audio_plugin_builtin_hrtf()  the plugin. Per source: input history (FIR), current HRIR pair + direction,
//                                  per-ear delay lines (fractional ITD, ramped over a period), pan gains, binaural weight.
//                                  A direction change (> 1 degree) runs the old and the new pair and crossfades over the
//                                  period; binaural <-> pan crossfade over one period as well.
// Thread:    data: parse on the game thread at init; plugin process / source_reset: audio thread.
// Not here:  the FIR kernel (audio_mix.h fir_stereo_store), which instances are binaural (audio_policy: rank by
//            audibility), the conversion of HRIR sets (tools/hrtf_tool.py, format there).
// Uses:      audio_plugins.h, audio_mix.h.
// Tests:     tests/audio_hrtf_test.cpp; cost: tests/audio_hrtf_bench.cpp.
// Data:      MIT KEMAR (Gardner, Martin, MIT Media Lab 1994) - free to use, the authors must be cited.

#include <stdint.h>
#include <stddef.h>
#include <string>
#include <vector>

#include "audio_plugins.h"

struct audio_hrtf_data {
    static const uint32_t k_max_taps = 256;
    static const uint32_t k_max_delay = 63;     // samples of ITD (KEMAR: 33 at 48 kHz)

    uint32_t                sample_rate = 0;
    uint32_t                taps = 0;
    uint32_t                count = 0;
    float                   scale = 0;          // int16 * scale = tap
    std::vector<float>      dir;                // count x 3: unit vector, listener space (+x right, +y up, +z forward)
    std::vector<float>      delay;              // count x 2: left, right (samples)
    std::vector<int16_t>    h;                  // count x (left[taps], right[taps]), forward order

    bool parse(const void* data, size_t size, std::string& error);

    // the HRIR pair for a unit direction (listener space): reversed (as fir_stereo_store wants) + the delays
    void interpolate(const float d[3], float* hl_rev, float* hr_rev, float& delay_l, float& delay_r) const;
};

// The plugin keeps its own copy of `data`; shutdown() releases it.
audio_spatial_plugin audio_plugin_builtin_hrtf(const audio_hrtf_data& data);

#endif
