// audio_bank.cpp - bank loader (summary in audio_bank.h).
// Parts: data lookups, init.json (buses), meta.json + sounds.json (sounds, stream overrides, events, layers, curves), attach.

#include "audio_bank.h"

#include <math.h>
#include "audio_format.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ---------------------------------------------------------------------------
// data
// ---------------------------------------------------------------------------
float audio_curve::eval(float v) const
{
    if (count == 0) return 1.0f;
    if (v <= x[0]) return y[0];
    for (uint32_t i = 1; i < count; ++i) {
        if (v <= x[i]) {
            float t = (v - x[i - 1]) / (x[i] - x[i - 1]);
            return y[i - 1] + (y[i] - y[i - 1]) * t;
        }
    }
    return y[count - 1];
}

const audio_sound* audio_bank_data::find_sound(uint32_t id) const
{
    for (const audio_sound& s : sounds) if (s.id == id) return &s;
    return nullptr;
}

const audio_event_desc* audio_bank_data::find_event(uint32_t id) const
{
    for (const audio_event_desc& e : events) if (e.id == id) return &e;
    return nullptr;
}

float audio_buses::air_cutoff(float distance, float event_air) const
{
    if (air_cutoff_100m <= 0 || event_air <= 0) return 0.0f;
    const float d = distance * event_air;
    if (d <= 0.01f) return 0.0f;
    float hz = air_cutoff_100m * sqrtf(100.0f / d);
    if (hz >= 20000.0f) return 0.0f;                    // inaudible: the stage stays off
    return hz < air_min_cutoff ? air_min_cutoff : hz;
}

uint32_t audio_buses::find(uint32_t id) const
{
    for (uint32_t i = 0; i < buses.size(); ++i) if (buses[i].id == id) return i;
    return ~0u;
}

const audio_reverb_preset* audio_buses::find_reverb(uint32_t id) const
{
    for (const auto& r : reverbs) if (r.first == id) return &r.second;
    return nullptr;
}

namespace {

uint32_t hash(const char* s) { return audio_id(s); }

bool fail(std::string& error, const std::string& msg) { error = msg; return false; }

int hexval(char c) { return c >= '0' && c <= '9' ? c - '0' : (c | 32) >= 'a' && (c | 32) <= 'f' ? (c | 32) - 'a' + 10 : -1; }

std::string audio_fmt_range(float lo, float hi)
{
    char buf[64];
    if (hi >= 1e29f) snprintf(buf, sizeof(buf), ">= %g", double(lo));
    else snprintf(buf, sizeof(buf), "%g .. %g", double(lo), double(hi));
    return buf;
}

} // namespace

// ---------------------------------------------------------------------------
// init.json
// ---------------------------------------------------------------------------
bool audio_build_init(const audio_fmt_init& in, audio_buses& out, std::string& error)
{
    error.clear();
    out.buses.clear();
    out.reverbs.clear();
    out.ducks.clear();
    out.air_cutoff_100m = in.air_absorption.cutoff_100m;
    out.air_min_cutoff  = in.air_absorption.min_cutoff;
    if ((out.air_cutoff_100m != 0 && (out.air_cutoff_100m < 100 || out.air_cutoff_100m > 20000)) || out.air_min_cutoff < 10 || out.air_min_cutoff > 20000)
        return fail(error, "init.json: air_absorption: cutoff_100m 0 (off) or 100 .. 20000 Hz, min_cutoff 10 .. 20000 Hz");
    for (const audio_fmt_reverb& r : in.reverbs) {
        if (r.name.empty()) return fail(error, "init.json: reverb without name");
        audio_reverb_preset p;
        p.decay = r.decay; p.predelay = r.predelay; p.damping = r.damping; p.size = r.size; p.level = r.level;
        if (p.decay < 0.05f || p.decay > 20 || p.predelay < 0 || p.predelay > 0.1f || p.size < 0.1f || p.size > 1 || p.damping < 100)
            return fail(error, "init.json: reverb '" + r.name + "': decay 0.05 .. 20 s, predelay 0 .. 0.1 s, size 0.1 .. 1, damping >= 100 Hz");
        uint32_t id = hash(r.name.c_str());
        if (out.find_reverb(id)) return fail(error, "init.json: reverb '" + r.name + "' twice (or hash collision)");
        out.reverbs.push_back({ id, p });
    }
    struct pending_duck { uint32_t source; std::string target; float gain, attack, release; };
    std::vector<pending_duck> ducks;                    // targets may be declared after their source
    for (const audio_fmt_bus& b : in.buses) {
        for (const audio_fmt_duck& dn : b.ducks) {
            pending_duck pd = { uint32_t(out.buses.size()), dn.bus, dn.gain, dn.attack, dn.release };
            if (!(pd.gain >= 0 && pd.gain <= 1) || !(pd.attack >= 0 && pd.attack <= 10) || !(pd.release >= 0 && pd.release <= 10))
                return fail(error, "init.json: bus '" + b.name + "': duck gain 0 .. 1, attack / release 0 .. 10 s");
            ducks.push_back(pd);
        }
        audio_bus_desc d;
        d.name = b.name;
        if (d.name.empty()) return fail(error, "init.json: bus without name");
        d.id = hash(d.name.c_str());
        if (out.find(d.id) != ~0u) return fail(error, "init.json: bus '" + d.name + "' twice (or hash collision)");
        d.volume = b.volume;
        d.voices = b.voices;
        d.reverb = b.effect == audio_fmt_effect_reverb;
        d.preset = 0;
        if (d.reverb) {
            if (b.preset.empty() && out.reverbs.empty()) return fail(error, "init.json: bus '" + d.name + "': a reverb bus needs reverb presets");
            d.preset = !b.preset.empty() ? hash(b.preset.c_str()) : out.reverbs[0].first;
            if (!out.find_reverb(d.preset)) return fail(error, "init.json: bus '" + d.name + "': unknown reverb preset '" + b.preset + "'");
        }
        if (out.buses.empty()) {
            if (!b.parent.empty()) return fail(error, "init.json: the first bus is master and has no parent");
            d.parent = 0;
        } else {
            uint32_t p = out.find(hash(!b.parent.empty() ? b.parent.c_str() : "master"));
            if (p == ~0u) return fail(error, "init.json: bus '" + d.name + "': parent '" + b.parent + "' must be declared before it");
            d.parent = p;
        }
        out.buses.push_back(d);
    }
    if (out.buses.empty()) return fail(error, "init.json: no buses");
    for (const pending_duck& pd : ducks) {
        const std::string& src = out.buses[pd.source].name;
        const uint32_t t = out.find(hash(pd.target.c_str()));
        if (t == ~0u) return fail(error, "init.json: bus '" + src + "': duck of an unknown bus '" + pd.target + "'");
        for (uint32_t a = pd.source;; a = out.buses[a].parent) {           // the source itself or a bus above it: it would duck itself
            if (a == t) return fail(error, "init.json: bus '" + src + "': ducks '" + pd.target + "', which it plays through");
            if (a == 0) break;
        }
        out.ducks.push_back({ pd.source, t, pd.gain, pd.attack, pd.release });
    }
    return true;
}

bool audio_parse_init(const char* text, size_t size, audio_buses& out, std::string& error)
{
    audio_fmt_init in;
    if (!audio_format_read(text, size, in, error)) { error = "init.json: " + error; return false; }
    return audio_build_init(in, out, error);
}

bool audio_duck_update(const audio_buses& b, const uint8_t* active, float dt, audio_duck_state* state)
{
    const float floor_db = -60.0f;                      // gain 0 fades to -60 dB, then it is 0
    auto to_db = [&](float g) { return g > 0.001f ? 20.0f * log10f(g) : floor_db; };
    bool changed = false;
    for (uint32_t t = 0; t < b.buses.size(); ++t) {
        const audio_duck_desc* win = nullptr;
        bool target = false;
        for (const audio_duck_desc& d : b.ducks) {
            if (d.target != t) continue;
            target = true;
            if (active[d.source] && (!win || d.gain < win->gain)) win = &d;
        }
        if (!target) continue;
        audio_duck_state& s = state[t];
        const float want = win ? win->gain : 1.0f;
        float cur = to_db(s.gain);
        const float want_db = to_db(want);
        // the release of the source holding the bus down; while it rises to a weaker source it keeps the departed one's
        if (win && want_db <= cur + 0.001f) { s.depth_db = -to_db(win->gain); s.release = win->release; }
        if (want_db < cur) {
            const float rate = win->attack > 0 ? -to_db(win->gain) / win->attack : 1e9f;    // dB per second: the full depth in `attack`
            cur = cur - rate * dt < want_db ? want_db : cur - rate * dt;
        } else if (want_db > cur) {
            const float rate = s.release > 0 ? s.depth_db / s.release : 1e9f;                // back up in `release` of the last source
            cur = cur + rate * dt > want_db ? want_db : cur + rate * dt;
        }
        const float g = cur <= floor_db ? 0.0f : cur >= -0.0001f ? 1.0f : powf(10.0f, cur / 20.0f);
        if (g != s.gain) { s.gain = g; changed = true; }
    }
    return changed;
}

// ---------------------------------------------------------------------------
// meta.json + sounds.json
// ---------------------------------------------------------------------------
bool audio_build_bank(const audio_fmt_bank& bank, const audio_fmt_sounds& snd, const audio_buses& buses, uint32_t sample_rate,
                      float stream_seconds, audio_bank_data& out, std::string& error)
{
    error.clear();
    out.sounds.clear(); out.curves.clear(); out.events.clear();

    // sounds.json: what the tool knows about every sound
    out.sounds.reserve(snd.sounds.size());              // pointers into it are kept by layers
    for (const audio_fmt_sound& s : snd.sounds) {
        audio_sound a;
        a.name = s.name;
        if (a.name.empty()) return fail(error, "sounds.json: sound without name");
        a.id = hash(a.name.c_str());
        if (out.find_sound(a.id)) return fail(error, "sounds.json: sound '" + a.name + "' twice (or hash collision)");
        a.vorbis = s.format == audio_fmt_format_vorbis;
        a.file = !s.file.empty() ? s.file : a.name + (a.vorbis ? ".ogg" : ".wav");
        a.path = out.sound_folder.empty() ? a.file : out.sound_folder + "/" + a.file;
        a.frames = s.frames;
        a.sample_rate = s.rate;
        if (a.vorbis) {                                 // decoded on the IO thread into mono int16 blocks
            a.samples_per_block = audio_vorbis::k_block_frames;
            a.block_align = a.samples_per_block * 2;
        } else {
            a.block_align = s.block_align;
            a.samples_per_block = (a.block_align - 4) * 2 + 1;
        }
        if (!a.frames) return fail(error, "sounds.json: sound '" + a.name + "' without frames");
        if (a.sample_rate != sample_rate) return fail(error, "sounds.json: sound '" + a.name + "' is not at the mixer rate");
        a.stream = s.stream || double(a.frames) > double(stream_seconds) * sample_rate;
        if (a.vorbis) a.stream = true;
        const size_t n = s.envelope.size();
        if (n & 1) return fail(error, "sounds.json: sound '" + a.name + "': odd envelope length");
        a.envelope.resize(n / 2);
        for (size_t i = 0; i < n / 2; ++i) {
            int hi = hexval(s.envelope[i * 2]), lo = hexval(s.envelope[i * 2 + 1]);
            if (hi < 0 || lo < 0) return fail(error, "sounds.json: sound '" + a.name + "': bad envelope");
            a.envelope[i] = uint8_t(hi * 16 + lo);
        }
        out.sounds.push_back(std::move(a));
    }

    // meta.json
    out.name = bank.name;
    out.id = hash(out.name.c_str());
    for (const audio_fmt_stream& s : bank.streams) {
        audio_sound* a = const_cast<audio_sound*>(out.find_sound(hash(s.name.c_str())));
        if (!a) return fail(error, "meta.json: streams: unknown sound '" + s.name + "'");
        if (a->vorbis && !s.stream) return fail(error, "meta.json: streams: '" + a->name + "' is Vorbis: streamed only");
        a->stream = s.stream;
    }

    // curves are stored in one vector: count them first so pointers stay valid
    size_t curves = 0;
    for (const audio_fmt_event& e : bank.events) {
        curves += !e.lowpass_curve.empty() + !e.highpass_curve.empty();
        for (const audio_fmt_send& s : e.sends) curves += !s.curve.empty();
        for (const audio_fmt_layer& l : e.layers) curves += !l.volume_curve.empty() + !l.pitch_curve.empty() + !l.delay_curve.empty();
    }
    out.curves.reserve(curves);
    out.events.reserve(bank.events.size());

    for (const audio_fmt_event& e : bank.events) {
        audio_event_desc ev;
        ev.name = e.name;
        if (ev.name.empty()) return fail(error, "meta.json: event without name");
        ev.id = hash(ev.name.c_str());
        if (out.find_event(ev.id)) return fail(error, "meta.json: event '" + ev.name + "' twice (or hash collision)");
        const std::string where = "event '" + ev.name + "'";
        ev.bank = &out;

        const std::string bus = e.bus.empty() ? "master" : e.bus;
        ev.bus = buses.find(hash(bus.c_str()));
        if (ev.bus == ~0u) return fail(error, where + ": unknown bus '" + bus + "'");
        ev.spatial = e.spatial == audio_fmt_spatial_2d ? audio_spatial_2d : e.spatial == audio_fmt_spatial_hrtf ? audio_spatial_hrtf : audio_spatial_pan;
        if (e.priority_near < 0 || e.priority_near > 255 || e.priority_far < -1 || e.priority_far > 255)
            return fail(error, where + ": priorities 0 .. 255");
        ev.priority_near = uint8_t(e.priority_near);
        ev.priority_far = uint8_t(e.priority_far < 0 ? e.priority_near : e.priority_far);
        ev.max_instances = e.max_instances;
        ev.air = e.air;
        if (ev.air < 0 || ev.air > 100) return fail(error, where + ": air must be 0 (none) .. 100 (distance multiplier)");
        if (e.pending_timeout < 0) return fail(error, where + ": pending_timeout must be >= 0");
        ev.pending_timeout = uint32_t(e.pending_timeout * sample_rate);
        ev.lowpass_hz = e.lowpass;
        ev.highpass_hz = e.highpass;
        if (ev.lowpass_hz < 0 || ev.highpass_hz < 0 || (ev.lowpass_hz && ev.lowpass_hz < 10) || ev.highpass_hz > 20000)
            return fail(error, where + ": filter cutoff out of range (lowpass >= 10 Hz, highpass <= 20 kHz)");
        ev.min_distance = e.min_distance;
        ev.max_distance = e.max_distance;
        if (ev.max_distance <= ev.min_distance) return fail(error, where + ": max_distance must be greater than min_distance");

        for (const audio_fmt_parameter& p : e.parameters) {
            if (p.name.empty()) return fail(error, where + ": parameter without name");
            audio_parameter_desc d;
            d.id = hash(p.name.c_str());
            d.min = p.min;
            d.max = p.max;
            if (d.max < d.min) return fail(error, where + ": parameter '" + p.name + "': max < min");
            d.def = p.def < p.min ? p.min : p.def > p.max ? p.max : p.def;
            if (d.id == k_audio_param_distance || d.id == k_audio_param_angle) return fail(error, where + ": 'distance' and 'angle' are built-in parameters");
            for (const audio_parameter_desc& x : ev.parameters) if (x.id == d.id) return fail(error, where + ": parameter '" + p.name + "' twice");
            ev.parameters.push_back(d);
        }
        for (const audio_fmt_switch& s : e.switches) {
            if (s.name.empty()) return fail(error, where + ": switch without name");
            audio_switch_desc d;
            d.id = hash(s.name.c_str());
            for (const std::string& v : s.values) d.values.push_back(hash(v.c_str()));
            if (d.values.empty()) return fail(error, where + ": switch '" + s.name + "' without values");
            d.def = !s.def.empty() ? hash(s.def.c_str()) : d.values[0];
            bool known = false;
            for (uint32_t v : d.values) if (v == d.def) known = true;
            if (!known) return fail(error, where + ": switch '" + s.name + "': default '" + s.def + "' is not one of its values");
            ev.switches.push_back(d);
        }
        auto switch_of = [&](const std::string& name) -> const audio_switch_desc* {
            for (const audio_switch_desc& d : ev.switches) if (d.id == hash(name.c_str())) return &d;
            return nullptr;
        };

        // a curve over a declared or built-in parameter; nullptr for none
        auto curve = [&](const audio_fmt_curve& c, const std::string& what, float ymin, float ymax, const audio_curve*& dst) -> bool {
            dst = nullptr;
            if (c.empty()) return true;
            audio_curve cd;
            cd.parameter = hash(c.parameter.c_str());
            bool known = cd.parameter == k_audio_param_distance || cd.parameter == k_audio_param_angle;
            for (const audio_parameter_desc& d : ev.parameters) if (d.id == cd.parameter) known = true;
            if (!known) return fail(error, where + ": " + what + ": parameter '" + c.parameter + "' is not declared in parameters");
            if (c.points.empty()) return fail(error, where + ": " + what + ": curve without points");
            if (c.points.size() > 16) return fail(error, where + ": " + what + ": more than 16 points");
            for (const audio_fmt_point& p : c.points) {
                if (cd.count && p.x <= cd.x[cd.count - 1]) return fail(error, where + ": " + what + ": points must have ascending x");
                if (p.y < ymin || p.y > ymax) return fail(error, where + ": " + what + ": y must be " + audio_fmt_range(ymin, ymax));
                cd.x[cd.count] = p.x;
                cd.y[cd.count] = p.y;
                ++cd.count;
            }
            out.curves.push_back(cd);
            dst = &out.curves.back();
            return true;
        };
        const float big = 1e30f;
        if (!curve(e.lowpass_curve, "lowpass_curve", 10.0f, 24000.0f, ev.lowpass_curve)) return false;
        if (!curve(e.highpass_curve, "highpass_curve", 10.0f, 24000.0f, ev.highpass_curve)) return false;

        if (e.sends.size() > 2) return fail(error, where + ": more than 2 sends");
        for (const audio_fmt_send& s : e.sends) {
            audio_send_desc sd;
            sd.bus = buses.find(hash(s.bus.c_str()));
            if (sd.bus == ~0u || !buses.buses[sd.bus].reverb) return fail(error, where + ": send to '" + s.bus + "': not a reverb bus (init.json: effect \"reverb\")");
            sd.level = s.level;
            if (sd.level < 0) return fail(error, where + ": send level must be >= 0");
            if (!s.switch_name.empty()) {
                const audio_switch_desc* sw = switch_of(s.switch_name);
                if (!sw) return fail(error, where + ": send switch '" + s.switch_name + "' is not declared in switches");
                sd.switch_id = sw->id;
                sd.switch_value = hash(s.value.c_str());
            }
            if (!curve(s.curve, "send curve", 0.0f, big, sd.curve)) return false;
            ev.sends.push_back(sd);
        }

        if (e.layers.empty()) return fail(error, where + ": no layers");
        if (e.layers.size() > 16) return fail(error, where + ": more than 16 layers");
        for (size_t li = 0; li < e.layers.size(); ++li) {
            const audio_fmt_layer& l = e.layers[li];
            const std::string lw = "layer '" + (l.name.empty() ? std::to_string(li + 1) : l.name) + "'";
            audio_layer_desc ld;
            ld.name = l.name;
            ld.volume_db = l.volume;
            ld.chance = l.chance;
            if (l.delay < 0) return fail(error, where + ": " + lw + ": delay must be >= 0");
            ld.delay = uint32_t(l.delay * sample_rate + 0.5f);
            ld.loop = l.loop;
            ld.avoid_repeat = l.avoid_repeat > 8 ? 8 : l.avoid_repeat;
            auto pool = [&](const std::vector<std::string>& names, audio_choice& c) -> bool {
                for (const std::string& n : names) {
                    const audio_sound* s = out.find_sound(hash(n.c_str()));
                    if (!s) return fail(error, where + ": " + lw + ": unknown sound '" + n + "'");
                    c.sounds.push_back(s);
                }
                return true;
            };
            if (!l.switch_name.empty()) {
                const audio_switch_desc* sw = switch_of(l.switch_name);
                if (!sw) return fail(error, where + ": " + lw + ": switch '" + l.switch_name + "' is not declared in switches");
                ld.switch_id = sw->id;
                for (const audio_fmt_case& c : l.cases) {
                    audio_choice ch;
                    ch.value = hash(c.value.c_str());
                    bool known = false;
                    for (uint32_t v : sw->values) if (v == ch.value) known = true;
                    if (!known) return fail(error, where + ": " + lw + ": case '" + c.value + "' is not a value of switch '" + l.switch_name + "'");
                    if (!pool(c.sounds, ch)) return false;
                    ld.cases.push_back(std::move(ch));
                }
                if (!pool(l.default_sounds, ld.fallback)) return false;
                if (!l.sounds.empty()) return fail(error, where + ": " + lw + ": a switch layer takes its sounds from cases / default, not 'sounds'");
            } else {
                if (l.sounds.empty()) return fail(error, where + ": " + lw + ": no sound");
                if (!pool(l.sounds, ld.pool)) return false;
                if (!l.cases.empty() || !l.default_sounds.empty()) return fail(error, where + ": " + lw + ": cases / default without a switch");
            }
            ld.pitch_min = l.pitch_min;
            ld.pitch_max = l.pitch_max < l.pitch_min ? l.pitch_min : l.pitch_max;
            if (ld.pitch_min <= 0) return fail(error, where + ": " + lw + ": pitch must be > 0");
            if (!curve(l.volume_curve, lw + " volume_curve", 0.0f, big, ld.volume_curve)) return false;
            if (!curve(l.pitch_curve, lw + " pitch_curve", 0.001f, big, ld.pitch_curve)) return false;
            if (!curve(l.delay_curve, lw + " delay_curve", 0.0f, big, ld.delay_curve)) return false;
            if (ld.loop) ev.loop = true;
            ev.layers.push_back(std::move(ld));
        }
        switch (e.virtual_mode) {
        case audio_fmt_virtual_kill:    ev.virtual_behavior = audio_virtual_kill; break;
        case audio_fmt_virtual_update:  ev.virtual_behavior = audio_virtual_update; break;
        case audio_fmt_virtual_restart: ev.virtual_behavior = audio_virtual_restart; break;
        default:                        ev.virtual_behavior = ev.loop ? audio_virtual_update : audio_virtual_kill; break;
        }
        out.events.push_back(std::move(ev));
    }
    return true;
}

bool audio_parse_bank(const char* bank_json, size_t bank_size, const char* sounds_json, size_t sounds_size,
                      const audio_buses& buses, uint32_t sample_rate, float stream_seconds,
                      audio_bank_data& out, std::string& error)
{
    audio_fmt_sounds snd;
    if (!audio_format_read(sounds_json, sounds_size, snd, error)) { error = "sounds.json: " + error; return false; }
    audio_fmt_bank bank;
    if (!audio_format_read(bank_json, bank_size, bank, error)) { error = "meta.json: " + error; return false; }
    return audio_build_bank(bank, snd, buses, sample_rate, stream_seconds, out, error);
}

bool audio_sound_attach(audio_sound& s, std::vector<uint8_t>&& wav, std::string& error)
{
    s.data = std::move(wav);
    audio_adpcm::clip c;
    if (!audio_adpcm::parse_wav(s.data.data(), s.data.size(), c)) return fail(error, "sound '" + s.name + "': not an IMA ADPCM mono WAV");
    if (c.frames != s.frames || c.block_align != s.block_align) return fail(error, "sound '" + s.name + "': does not match sounds.json (rebuild the bank)");
    if (size_t(c.blocks - s.data.data()) + size_t(c.block_count) * c.block_align > s.data.size()) return fail(error, "sound '" + s.name + "': truncated");
    s.clip = c;
    s.loaded = true;
    return true;
}
