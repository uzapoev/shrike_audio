# Audio module

An event-driven audio engine on top of miniaudio. The game posts **events** and sets **parameters / switches**. The
**bank** (XML written by a sound designer) decides how this sounds: which sounds play, how loud, with what filters,
where in space. The module is independent of the rest of the engine. It has its own file IO interface, its own threads
and its own mixing graph. miniaudio is used only for the playback device.

What it does:

- events with layers, random / switch containers, parameter curves (volume, pitch, filter cutoff, send level), delays
- voice management: priorities, audibility, virtual events, stealing with fade-out, limits per event / bus / global
- IMA ADPCM sounds (resident or streamed from disk), Ogg Vorbis sounds (streamed, decoded on the IO thread),
  resampling, gain ramps
- per-instance low-pass / high-pass filters, air absorption by distance
- buses with volume / mute / pause / fades, ducking (a bus turns others down while it sounds), reverb (FDN) as a bus
  effect with sends
- spatial: equal-power panning or binaural HRTF (headphones mode); replaceable by an own plugin
- profiling: time per stage of every thread, memory, pool usage, live streams and instances

Platforms: Windows / macOS / Linux / iOS / Android (main), web / Nintendo (experimental). SIMD: SSE2, ARM NEON through
sse2neon; every SIMD kernel has a scalar reference.

---

## Quick start

```cmake
add_subdirectory(src/audio audio_core)
target_link_libraries(my_game PRIVATE audio_core)
```

```cpp
#include "audio/audio.h"

audio a;
audio_desc_t d;
d.root_path = "assets/audio/";                 // init.xml, hrtf.bin and the bank folders are here
if (!a.init(d)) log("audio: %s", a.last_error());

audio_bank_t weapons = a.register_bank("weapons");    // reads weapons/meta.xml + weapons/sounds/sounds.xml
a.load_bank(weapons);                                  // asynchronous; events posted earlier wait (pending)

audio_object_t listener = a.create_object("listener");
a.set_listener(listener);
audio_object_t gun = a.create_object("gun");

// every frame
a.set_object_transform(listener, camera_pos, camera_rot);
a.set_object_transform(gun, gun_pos, gun_rot);
a.set_global_parameter(audio_id("inside"), inside);    // 0 .. 1, any parameter the bank declares
if (fired) a.post_event(gun, audio_id("rifle_fire"));
a.update(dt);                                          // once per frame: decisions + one command batch to the mixer
```

All calls are made from one thread (the game thread). `update()` is called once per frame.

---

## Files on disk

```text
assets/audio/
    init.xml                    buses, reverb presets, air absorption           (written by hand)
    hrtf.bin                    optional: HRIR set of the binaural plugin       (tools/hrtf_tool.py)
    weapons/                    a bank = a folder
        meta.xml                events                                          (written by hand)
        sounds/                 everything here is generated, do not edit
            sounds.xml          format, length, loudness envelope per sound     (tools/audio_bank_tool)
            rifle_shot_01.wav   IMA ADPCM, mono, 48 kHz, 1024-byte blocks       (tools/audio_bank_tool)
            theme.ogg           Ogg Vorbis, 48 kHz, 1 or 2 channels (copied)    (tools/audio_bank_tool)
```

- **audio_bank_tool** `<source dir> <bank dir> [--stream-seconds 10]` converts wav / flac / mp3 to IMA ADPCM. It also
  computes each sound's loudness envelope (the voice policy needs it before the data is loaded). Sounds longer than
  `--stream-seconds` are marked as streamed.
  - A ready `.ogg` is not converted. The tool checks it the way the runtime will open it (Ogg Vorbis, 48 kHz, 1 or 2
    channels, decoder memory), computes the envelope from the decoded sound and copies the file. Vorbis sounds are
    always streamed (`format="vorbis"` in sounds.xml). The tool has no Vorbis encoder: files come prepared (later: the
    authoring tool).
  - Stereo is played mono for now: (L + R) / 2.
- **hrtf_tool.py** `<diffuse.zip> <hrtf.bin> [--taps 128]` converts the MIT KEMAR HRTF set for the binaural plugin
  (needs numpy).

---

## Terms

| Term | Meaning |
|---|---|
| **event** | a description in the bank (`<event>` in meta.xml): layers, curves, filters, sends, limits |
| **event instance** | one playing copy of an event (`audio_playing_t`); the unit of voice management, has one filter and one spatial source |
| **layer / voice** | a layer of the event plays one sound; a playing sound is a mixer voice. An instance has 1..16 voices |
| **object** | a position plus parameter / switch values (`audio_object_t`). Has no audio node; its instances read from it |
| **bus** | a mixer node (`<bus>` in init.xml): volume, mute, pause, voice limit, optionally a reverb |
| **parameter / switch** | a float / a named value. Lookup: instance override → object → global → default of the event |

---

## Threads

```text
 game thread                 audio thread (miniaudio callback)        IO thread            loader thread
 ───────────                 ────────────────────────────────         ─────────            ─────────────
 API calls                   apply commands                           stream rings         load_bank: read
 update():                   render periods of 256 frames:            (open, prefill,      resident WAV files
   parameters, curves          decode ADPCM, mix voices,              refill; Vorbis
                                                                      decoding)
   score / select voices       filters, sends, spatial,
   commands ───────────────►   buses, reverb, master          ◄────── ring blocks
           ◄─────────────── notifications (voice ended)
```

- **Game thread:** only decisions (≈12 µs per frame in saltline).
- **Audio thread:** all signal processing (≈50 µs per 5.3 ms period in saltline, ~1 %).
- The threads are connected by lock-free single-producer / single-consumer queues and rings. No thread waits for
  another.
- Exception: with `audio_desc_t::start_device = false` (tests, tools) there is no device. `render()` then runs on the
  thread that calls it.

---

## The path of a sound

### 1. Registration and loading (game thread + loader thread)

1. `init()` reads `init.xml` (buses, reverb presets, air absorption) and, if present, `hrtf.bin`. It creates the mixer
   (all memory is allocated once), the stream system and the voice policy, then starts the miniaudio device.
2. `register_bank("weapons")` reads `meta.xml` and `sounds/sounds.xml` synchronously. It parses the events, curves and
   sound table, including each sound's length and envelope. No WAV file is opened yet.
3. `load_bank()` queues a job. The **loader thread** reads every resident (non-streamed) WAV through `audio_file_io`.
   On the next `update()`, the game thread attaches the data to the sounds: the WAV header is checked against
   `sounds.xml`. The bank becomes `ready`.
4. Streamed sounds are not read at load time. Each one is opened when an instance starts it (step 4).

### 2. post_event (game thread)

`post_event(object, event)` takes a **record** (an event instance) and resolves everything that is fixed for its
lifetime:

- `chance` of each layer;
- the `switch` case (from the object's / global switch values);
- the `random` choice (with `avoid_repeat`);
- random pitch and layer delays.

Nothing is sent to the mixer yet. The decision is made in the next `update()`. If the bank is not loaded yet, the record
is **pending**. A one-shot waits up to `pending_timeout` (150 ms by default); a loop waits for its bank as long as it
takes.

### 3. update(): the voice policy (game thread)

Once per frame `audio_policy::update()`:

1. **Notifications.** Takes "voice ended" from the mixer and returns voices to the pool. An instance whose voices are
   all done is released.
2. **Score.** Scores every candidate (playing, virtual, starting):
   - **priority** from `priority_near` / `priority_far` by distance;
   - **audibility (dB)** = the sound's envelope at its current position + layer volumes (curves) + distance attenuation
     + bus volume, summed by power over the layers.
3. **Select**, in this order: threshold (−60 dB) → `max_instances` of the event → `voices` limit of each bus up the
   chain → global voice budget. Playing instances get a +3 dB bonus (hysteresis) so they do not flicker.
4. **Apply:**
   - stolen instances fade out (a reserve of voices is kept for those fades);
   - inaudible ones become **virtual**. `virtual="kill"` ends them, `update` keeps their time running, `restart`
     starts them over when they come back;
   - newly selected ones start (step 4).
5. **Live instances.** For every playing instance:
   - its position relative to the listener;
   - layer volume / pitch curves;
   - filter cutoffs, including air absorption: `cutoff = cutoff_100m · √(100 / distance)`;
   - send levels;
   - the binaural flag: the 16 most audible `spatial="hrtf"` instances while headphones mode is on.

   Values are sent only when they change noticeably (more than 1 %).
6. **Commit.** All commands of the frame are published to the mixer as **one batch** (`commit`). The audio thread
   applies a batch as a whole, so a frame's changes never land half way.

### 4. Starting an instance

- **Resident sound:** `instance_start`, `instance_spatial`, `instance_filter`, `instance_send` and one `voice_play` per
  layer go into the batch. The voice starts at its layer delay, counted in frames. Delays are sample-accurate, so the
  layers of a shot stay aligned.
- **Streamed sound:**
  1. The policy opens a stream slot (`audio_stream_system::open`) and the instance waits in state **starting**.
  2. The **IO thread** opens the file and looks at its first bytes:
     - **IMA ADPCM WAV:** it parses the header and fills the whole ring with file blocks (8 × 1024 bytes ≈ 340 ms);
     - **Ogg Vorbis** (`OggS`): it takes a free decoder (`max_vorbis_streams`, 4 by default), opens the file through
       `audio_file_io` (stb_vorbis reads with `read_at`, 4 KB at a time), checks the rate, seeks to the start block and
       fills the decoder's own ring with **decoded** mono int16 blocks (16 × 1024 frames ≈ 340 ms).

     Then it marks the stream `ready`.
  3. The next `update()` sees `ready` and sends `voice_play` with the stream instead of a clip.
  4. While the voice plays, the IO thread keeps the ring full, wrapping to the first block for loops (Vorbis: a seek
     to frame 0, sample-exact, so loops are seamless).
  5. If the ring runs empty (**underrun**), the voice outputs silence and does not move. It continues when data
     arrives.
  6. The stream is closed only after the mixer has reported the voice finished, so the ring never changes under a
     reading voice.

### 5. The audio thread: one period (256 frames, 5.3 ms at 48 kHz)

miniaudio calls `audio_mixer::render(out, frames)`:

1. **Commands.** Every published batch is applied first: voices start / stop, gains, filters, bus volumes, reverb
   presets.
2. **FIFO.** The device may ask for any number of frames. The mixer always renders whole periods of 256 frames and
   keeps the remainder in a small FIFO.
3. **One period** (`render_period`). Instances are processed in groups of 4:
   - **voices → instance mono buffer:**
     - the voice's current IMA ADPCM block (2041 samples) is decoded into its cache. A resident clip is decoded from
       memory; a stream from its ring. A Vorbis block is already decoded: it is only copied (1024 samples);
     - the block is mixed into the instance's mono buffer with a gain ramp (gain × fade, linear over the segment);
     - pitch ≠ 1 goes through linear-interpolation resampling (16.16 fixed point).

     Delays and fades are cut into segments, so starts and stops are sample-accurate.
   - **filters:** low-pass then high-pass (2nd-order Butterworth biquads). Four instances run in the 4 SIMD lanes.
   - **sends:** the filtered mono signal goes to aux buses (reverb), post filter and pre panning.
   - **spatial plugin:**
     - not binaural: equal-power panning;
     - binaural: HRTF — a pair of 128-tap FIRs plus a fractional per-ear delay (ITD). A direction change crossfades
       the old and the new pair over the period.

     The output is added into the instance's bus (stereo).
4. **Buses**, from the leaves up to master (each bus's volume already includes its ducking gain, computed on the game
   thread in `update()`):
   - a reverb bus runs its FDN (8 delay lines, Hadamard matrix) on the summed sends. After a preset change, the old
     reverb rings out on its own;
   - each bus is mixed into its parent with a volume ramp. Mute and pause apply here.
5. **Master:** volume ramp, a soft clip (transparent below 0.75), interleaved stereo float into miniaudio's buffer.
6. **Notifications.** "Voice ended / stopped" notifications are published for the game thread (step 3.1).

Denormals are flushed to zero during `render()` (FTZ / DAZ on x86).

---

## Voice management, in short

- A budget of audible voices (`max_voices`) plus a reserve for fade-outs (`voice_reserve`). There are also mixer
  instances (`max_instances`) and records (`max_virtual`).
- The most important and audible events play. The rest are virtual and cost only a record.
- Limits apply in this order: event `max_instances` → bus `voices` → global budget.
- A stolen instance fades out (10 ms by default). A returning `update` instance resumes at the right position.
- `get_stats()` shows the pools as used / peak / capacity. Size the limits by the peaks.

## Ogg Vorbis

- **What for:** music and long ambience. IMA ADPCM stays for short sounds: it costs nothing to decode and has no
  decoder memory.
- **Where it is decoded:** on the IO thread, not on the audio thread. stb_vorbis decodes packets of up to 1024 frames
  at once; one call can take up to 1.2 ms on a PC (a fifth of a period), several times that on phones. On the IO thread
  this is harmless: the mixer gets ready int16 blocks and copies them.
- **Cost** (PC, stereo 48 kHz → mono): ≈1.3 ms of the IO thread per second of audio (0.13 % of a core), ≈35–45 µs
  per 1024-frame block. Seeking is sample-exact, ≈0.1 ms.
- **Memory:** every decoder has a fixed block (`vorbis_memory`, 256 KB: codebooks and work buffers; the files measured
  need 190–220 KB) and its own int16 ring (32 KB). `max_vorbis_streams` decoders (4) are allocated at init: ≈1.15 MB.
  A file that needs more memory, a file at another rate or a stream with no free decoder ends in state `error`; the
  reason is in `debug_streams()`.
- **Limits of the first step:** streams only; mono (stereo is mixed down); 48 kHz; no encoder in the tool.

---

## Bank format

**init.xml**

```xml
<audio_init>
    <bus name="master"/>
    <bus name="sfx"      parent="master"/>
    <bus name="weapon"   parent="sfx"   voices="24">                  <!-- voice limit of the bus -->
        <duck bus="ambience" gain="0.5" attack="0.05" release="1.2"/>  <!-- while it sounds: ambience x 0.5 -->
    </bus>
    <bus name="ambience" parent="master" volume="0.7"/>
    <reverb name="outdoor"  decay="0.9" predelay="0.035" damping="4500" size="0.9" level="0.45"/>
    <reverb name="workshop" decay="1.5" predelay="0.012" damping="7500" size="0.65" level="0.8"/>
    <bus name="reverb" parent="sfx" effect="reverb" preset="outdoor"/>   <!-- the game switches presets: set_bus_reverb -->
    <air_absorption cutoff_100m="2500" min_cutoff="900"/>             <!-- default 4000 / 800; 0: off -->
</audio_init>
```

**meta.xml** (one event)

```xml
<audio_bank name="weapons">
    <event name="bullet_impact" bus="world" spatial="hrtf" priority_near="40" priority_far="10" max_instances="8">
        <attenuation min="4" max="120"/>
        <parameters> <parameter name="inside" min="0" max="1" default="0"/> </parameters>
        <switches>
            <switch name="surface" default="concrete"> <value name="concrete"/> <value name="metal"/> </switch>
        </switches>
        <send bus="reverb"> <curve parameter="inside"> <point x="0" y="0.15"/> <point x="1" y="0.5"/> </curve> </send>
        <layer name="close" volume="-6">
            <switch switch="surface">
                <case value="metal"> <random avoid_repeat="1"> <sound name="hit_metal_01"/> <sound name="hit_metal_02"/> </random> </case>
                <default sound="hit_concrete_01"/>
            </switch>
            <volume><curve parameter="distance"> <point x="0" y="1"/> <point x="30" y="0"/> </curve></volume>
            <pitch min="0.95" max="1.05"/>
        </layer>
    </event>
</audio_bank>
```

Rules:

- **Units:** time in seconds; static layer volume in dB; volume curves are linear 0..1; pitch is a ratio.
- **Resolved once at post:** chance, switch case, random choice, random pitch, delay. **Live every update:** curves.
- **Built-in curve parameters:** `distance` (m) and `angle` (0..180°).
- **Layer grammar is flat:** `layer` → `sound` | `random` | `switch`.
- **`spatial`:** `2d` | `pan` (default) | `hrtf`.
- **`virtual`:** `kill` (default) | `update` | `restart`.
- **Filters:** `<lowpass cutoff="..."/>` or `<lowpass><curve parameter="...">` (y in Hz), the same for `<highpass>`.
  Air absorption applies on top; the lower cutoff wins. `air="0.6"` on an event scales the distance; `air="0"` turns
  it off for that event.
- **Sends:** `<send bus="..." level="..."/>` or with a curve, at most 2 per event. Optional condition
  `switch="..." value="..."`: the send is on only for that switch value.
- **Names:** event names are unique across all banks; sound names are unique within a bank.
- **Ducking** (`<duck>` under a bus in init.xml): while the source bus or a bus under it has a playing instance and is
  not paused, the target bus is multiplied by `gain`. The fade is linear in dB: down over `attack`, up over `release`
  (seconds). With several sources the lowest gain wins. `get_bus_duck(bus)` shows the current value.

The full list of decisions and why they were made is in [requrements.txt](requrements.txt) (in Russian), sections 9–13.

---

## Spatial audio

- **Built-in panning:** equal power, a gain ramp per period. It folds to the centre inside 0.25 m.
- **Built-in HRTF:** loaded when `hrtf.bin` sits next to `init.xml` (see `audio_desc_t::hrtf_file`).
  - It is off until the game calls `set_binaural(true)` (headphones mode: on speakers panning sounds better).
  - Only `spatial="hrtf"` events go binaural, at most `max_binaural` (16) of them, the most audible ones. The rest are
    panned.
  - Cost: ~5 µs per source per period, ~9.4 µs while the head turns.
- **Own plugin:** pass `audio_desc_t::plugin` (an `audio_spatial_plugin`, see [audio_plugins.h](audio_plugins.h)). The
  core does the distance attenuation itself and passes the gain in. The plugin gets the mono signal and the
  listener-space position, and adds stereo into the bus.
- **Acoustic geometry** (`create_geometry` ...): only handles for now; meant for Steam Audio as a plugin.

---

## Profiling and debugging

| Call | What it gives |
|---|---|
| `get_stats()` | counters: voices, instances, virtual, steals, underruns; pools used / peak / capacity; dropped commands |
| `get_profile()` | time per stage of the audio thread (commands, decode, voices, filters, sends, spatial, buses, output), periods over 50 / 80 / 100 % of the budget, callback interval; `update()` and its phases; IO read time, Vorbis decode time per block, bytes/s, stream start latency, bank load time |
| `get_memory()` | sounds, envelopes, metadata, mixer, reverbs, queues, stream rings and Vorbis decoders, spatial plugin, policy; total and peak |
| `debug_streams()` | every open stream: format, ring fill and its minimum, underruns, bytes, position, the reason of an error |
| `debug_instances()` | every instance: event, object, state, bus, position, distance, audibility, priority, voices, filters, sends |
| `set_profiling(bool)` | per-stage timers of the audio thread (~3 µs per period; whole periods are always timed) |
| `reset_stats()` | peaks and all-time maxima start over |

Values cover the last full second and are refreshed once a second. Any number of readers can call them.

The next steps (planned) are a capture stream (API calls, policy decisions, periodic snapshots to a file or a socket,
session replay) and a profiler window in the editor.

---

## Code map

| File | Role | Thread |
|---|---|---|
| [src/audio.h](audio.h) / [src/audio.cpp](audio.cpp) | public API, glue: objects, parameters, bank loading, stats | game (+ loader thread) |
| [src/audio_bank.h](audio_bank.h) / [.cpp](audio_bank.cpp) | init.xml / meta.xml / sounds.xml parsing, data of a bank | game |
| [src/audio_policy.h](audio_policy.h) / [.cpp](audio_policy.cpp) | event instances, voice management, commands to the mixer | game |
| [src/audio_mixer.h](audio_mixer.h) / [.cpp](audio_mixer.cpp) | the graph: voices → instances → buses → master; miniaudio device | audio |
| [src/audio_mix.h](audio_mix.h) | SIMD kernels: voice mix, resampling, panning, biquads, HRTF FIR, output, float → int16 | audio (IO) |
| [src/audio_adpcm.h](audio_adpcm.h) | IMA ADPCM: WAV parsing, block decode / encode | audio, tools |
| [src/audio_reverb.h](audio_reverb.h) | FDN reverb (8 lines), SIMD | audio |
| [src/audio_stream.h](audio_stream.h) / [.cpp](audio_stream.cpp) | streaming: slots, rings, Vorbis decoders, IO thread | IO |
| [src/audio_vorbis.h](audio_vorbis.h) / [.cpp](audio_vorbis.cpp) | Ogg Vorbis decoder in fixed memory: open, seek, mono int16 | IO, tools |
| [src/audio_plugins.h](audio_plugins.h) / [.cpp](audio_plugins.cpp) | spatial plugin interface, built-in panning | audio |
| [src/audio_hrtf.h](audio_hrtf.h) / [.cpp](audio_hrtf.cpp) | built-in binaural plugin (HRTF) | audio |
| [src/audio_geometry.h](audio_geometry.h) / [.cpp](audio_geometry.cpp) | acoustic geometry handles → plugin | game |
| [src/audio_queue.h](audio_queue.h) | SPSC queue with batch commit | — |
| [tools/](tools/) | audio_bank_tool (sounds → ADPCM bank, .ogg checked + copied), hrtf_tool.py (HRIR set → hrtf.bin), stb_vorbis_patch.py (upstream stb_vorbis → external/stb/stb_vorbis.c) | — |

Every file starts with a summary: role, contents, thread, what is not there, what it uses, tests.

---

## Building and tests

```text
cmake -S src/audio -B build_audio
cmake --build build_audio --config Release
ctest --test-dir build_audio -C Release
```

- Options: `AUDIO_BUILD_TOOLS` (audio_bank_tool), `AUDIO_BUILD_TESTS` (on by default).
- The tests are in [tests/](../../tests/): kernels (`audio_mix_test`), mixer, streams (real-time, ~2 min), policy and
  bank, the whole API, HRTF, the Vorbis decoder (`audio_vorbis_test`). `audio_hrtf_bench` and `audio_proto_test` are
  experiments, not part of ctest. Vorbis test files are in `tests/data/audio` (CC0 / public domain, see its README).
- An example game using the module: [games/saltline](../../games/saltline/) (`src/sl_audio.*`, `assets/audio/`).

---

## Third-party

- [miniaudio](https://miniaud.io) — the playback device and the bank tool's decoders (public domain / MIT-0).
- [pugixml](https://pugixml.org) — XML (MIT).
- [sse2neon](https://github.com/DLTcollab/sse2neon) — SSE on ARM (MIT).
- [stb_vorbis](https://github.com/nothings/stb) v1.25 — Ogg Vorbis decoding (public domain / MIT). A patched copy:
  SIMD for the IMDCT, overlap and channel coupling (bit-exact with the original, ×1.2–1.5 per file), reading through
  callbacks. It is generated by `tools/stb_vorbis_patch.py`; do not edit it by hand.
- HRTF data: MIT KEMAR, Bill Gardner and Keith Martin, MIT Media Lab, 1994. Free to use, including commercially,
  provided the authors are cited (game credits).
