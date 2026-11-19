// audio_bank.cpp - bank loader (summary in audio_bank.h).
// Parts: data lookups, init.xml (buses), meta.xml + sounds.xml (sounds, stream overrides, events, layers, curves), attach.

#include "audio_bank.h"

#include <math.h>
#include "pugixml.hpp"

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

bool parse_curve(pugi::xml_node n, audio_curve& c, std::string& error, const char* where)
{
    c = audio_curve();
    const char* p = n.attribute("parameter").as_string();
    if (!*p) return fail(error, std::string(where) + ": curve without parameter");
    c.parameter = hash(p);
    for (pugi::xml_node pt : n.children("point")) {
        if (c.count == 16) return fail(error, std::string(where) + ": more than 16 points");
        c.x[c.count] = pt.attribute("x").as_float();
        c.y[c.count] = pt.attribute("y").as_float();
        if (c.count && c.x[c.count] <= c.x[c.count - 1]) return fail(error, std::string(where) + ": points must have ascending x");
        ++c.count;
    }
    if (!c.count) return fail(error, std::string(where) + ": curve without points");
    return true;
}

} // namespace

// ---------------------------------------------------------------------------
// init.xml
// ---------------------------------------------------------------------------
bool audio_parse_init(const char* xml, size_t size, audio_buses& out, std::string& error)
{
    error.clear();
    out.buses.clear();
    out.reverbs.clear();
    pugi::xml_document doc;
    pugi::xml_parse_result r = doc.load_buffer(xml, size);
    if (!r) return fail(error, std::string("init.xml: ") + r.description());
    pugi::xml_node root = doc.child("audio_init");
    if (!root) return fail(error, "init.xml: no <audio_init>");
    out.air_cutoff_100m = 4000.0f; out.air_min_cutoff = 800.0f;
    if (pugi::xml_node air = root.child("air_absorption")) {
        out.air_cutoff_100m = air.attribute("cutoff_100m").as_float(out.air_cutoff_100m);
        out.air_min_cutoff  = air.attribute("min_cutoff").as_float(out.air_min_cutoff);
        if ((out.air_cutoff_100m != 0 && (out.air_cutoff_100m < 100 || out.air_cutoff_100m > 20000)) || out.air_min_cutoff < 10 || out.air_min_cutoff > 20000)
            return fail(error, "init.xml: air_absorption: cutoff_100m 0 (off) or 100 .. 20000 Hz, min_cutoff 10 .. 20000 Hz");
    }
    for (pugi::xml_node r : root.children("reverb")) {
        std::string name = r.attribute("name").as_string();
        if (name.empty()) return fail(error, "init.xml: reverb without name");
        audio_reverb_preset p;
        p.decay    = r.attribute("decay").as_float(p.decay);
        p.predelay = r.attribute("predelay").as_float(p.predelay);
        p.damping  = r.attribute("damping").as_float(p.damping);
        p.size     = r.attribute("size").as_float(p.size);
        p.level    = r.attribute("level").as_float(p.level);
        if (p.decay < 0.05f || p.decay > 20 || p.predelay < 0 || p.predelay > 0.1f || p.size < 0.1f || p.size > 1 || p.damping < 100)
            return fail(error, "init.xml: reverb '" + name + "': decay 0.05 .. 20 s, predelay 0 .. 0.1 s, size 0.1 .. 1, damping >= 100 Hz");
        uint32_t id = hash(name.c_str());
        if (out.find_reverb(id)) return fail(error, "init.xml: reverb '" + name + "' twice (or hash collision)");
        out.reverbs.push_back({ id, p });
    }
    out.ducks.clear();
    struct pending_duck { uint32_t source; std::string target; float gain, attack, release; };
    std::vector<pending_duck> ducks;                    // targets may be declared after their source
    for (pugi::xml_node b : root.children("bus")) {
        for (pugi::xml_node dn : b.children("duck")) {
            pending_duck pd = { uint32_t(out.buses.size()), dn.attribute("bus").as_string(""),
                                dn.attribute("gain").as_float(0.5f), dn.attribute("attack").as_float(0.1f), dn.attribute("release").as_float(0.5f) };
            if (!(pd.gain >= 0 && pd.gain <= 1) || !(pd.attack >= 0 && pd.attack <= 10) || !(pd.release >= 0 && pd.release <= 10))
                return fail(error, std::string("init.xml: bus '") + b.attribute("name").as_string() + "': duck gain 0 .. 1, attack / release 0 .. 10 s");
            ducks.push_back(pd);
        }
        audio_bus_desc d;
        d.name = b.attribute("name").as_string();
        if (d.name.empty()) return fail(error, "init.xml: bus without name");
        d.id = hash(d.name.c_str());
        if (out.find(d.id) != ~0u) return fail(error, "init.xml: bus '" + d.name + "' twice (or hash collision)");
        d.volume = b.attribute("volume").as_float(1.0f);
        d.voices = b.attribute("voices").as_uint(0);
        std::string effect = b.attribute("effect").as_string("");
        d.reverb = effect == "reverb";
        if (!effect.empty() && !d.reverb) return fail(error, "init.xml: bus '" + d.name + "': effect must be 'reverb'");
        d.preset = 0;
        if (d.reverb) {
            const char* preset = b.attribute("preset").as_string("");
            if (!*preset && out.reverbs.empty()) return fail(error, "init.xml: bus '" + d.name + "': a reverb bus needs <reverb> presets");
            d.preset = *preset ? hash(preset) : out.reverbs[0].first;
            if (!out.find_reverb(d.preset)) return fail(error, "init.xml: bus '" + d.name + "': unknown reverb preset '" + preset + "'");
        }
        const char* parent = b.attribute("parent").as_string();
        if (out.buses.empty()) {
            if (*parent) return fail(error, "init.xml: the first bus is master and has no parent");
            d.parent = 0;
        } else {
            uint32_t p = out.find(hash(*parent ? parent : "master"));
            if (p == ~0u) return fail(error, "init.xml: bus '" + d.name + "': parent '" + parent + "' must be declared before it");
            d.parent = p;
        }
        out.buses.push_back(d);
    }
    if (out.buses.empty()) return fail(error, "init.xml: no buses");
    for (const pending_duck& pd : ducks) {
        const std::string& src = out.buses[pd.source].name;
        const uint32_t t = out.find(hash(pd.target.c_str()));
        if (t == ~0u) return fail(error, "init.xml: bus '" + src + "': duck of an unknown bus '" + pd.target + "'");
        for (uint32_t a = pd.source;; a = out.buses[a].parent) {           // the source itself or a bus above it: it would duck itself
            if (a == t) return fail(error, "init.xml: bus '" + src + "': ducks '" + pd.target + "', which it plays through");
            if (a == 0) break;
        }
        out.ducks.push_back({ pd.source, t, pd.gain, pd.attack, pd.release });
    }
    return true;
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
// meta.xml + sounds.xml
// ---------------------------------------------------------------------------
bool audio_parse_bank(const char* bank_xml, size_t bank_size, const char* sounds_xml, size_t sounds_size,
                      const audio_buses& buses, uint32_t sample_rate, float stream_seconds,
                      audio_bank_data& out, std::string& error)
{
    error.clear();
    out.sounds.clear(); out.curves.clear(); out.events.clear();

    // sounds.xml: what the tool knows about every sound
    pugi::xml_document sdoc;
    pugi::xml_parse_result sr = sdoc.load_buffer(sounds_xml, sounds_size);
    if (!sr) return fail(error, std::string("sounds.xml: ") + sr.description());
    pugi::xml_node sroot = sdoc.child("audio_sounds");
    if (!sroot) return fail(error, "sounds.xml: no <audio_sounds>");
    size_t sound_count = 0;
    for (pugi::xml_node s : sroot.children("sound")) { (void)s; ++sound_count; }
    out.sounds.reserve(sound_count);                    // pointers into it are kept by layers
    for (pugi::xml_node s : sroot.children("sound")) {
        audio_sound a;
        a.name = s.attribute("name").as_string();
        if (a.name.empty()) return fail(error, "sounds.xml: sound without name");
        a.id = hash(a.name.c_str());
        if (out.find_sound(a.id)) return fail(error, "sounds.xml: sound '" + a.name + "' twice (or hash collision)");
        const std::string format = s.attribute("format").as_string("adpcm");
        if (format != "adpcm" && format != "vorbis") return fail(error, "sounds.xml: sound '" + a.name + "': format must be adpcm / vorbis");
        a.vorbis = format == "vorbis";
        a.file = s.attribute("file").as_string((a.name + (a.vorbis ? ".ogg" : ".wav")).c_str());
        a.path = out.sound_folder.empty() ? a.file : out.sound_folder + "/" + a.file;
        a.frames = s.attribute("frames").as_ullong();
        a.sample_rate = s.attribute("rate").as_uint(sample_rate);
        if (a.vorbis) {                                 // decoded on the IO thread into mono int16 blocks
            a.samples_per_block = audio_vorbis::k_block_frames;
            a.block_align = a.samples_per_block * 2;
        } else {
            a.block_align = s.attribute("block_align").as_uint(1024);
            a.samples_per_block = (a.block_align - 4) * 2 + 1;
        }
        if (!a.frames) return fail(error, "sounds.xml: sound '" + a.name + "' without frames");
        if (a.sample_rate != sample_rate) return fail(error, "sounds.xml: sound '" + a.name + "' is not at the mixer rate");
        a.stream = s.attribute("stream") ? s.attribute("stream").as_bool() : double(a.frames) > double(stream_seconds) * sample_rate;
        if (a.vorbis && !a.stream) return fail(error, "sounds.xml: sound '" + a.name + "': Vorbis sounds are streamed only");
        const char* env = s.attribute("envelope").as_string();
        size_t n = strlen(env);
        if (n & 1) return fail(error, "sounds.xml: sound '" + a.name + "': odd envelope length");
        a.envelope.resize(n / 2);
        for (size_t i = 0; i < n / 2; ++i) {
            int hi = hexval(env[i * 2]), lo = hexval(env[i * 2 + 1]);
            if (hi < 0 || lo < 0) return fail(error, "sounds.xml: sound '" + a.name + "': bad envelope");
            a.envelope[i] = uint8_t(hi * 16 + lo);
        }
        out.sounds.push_back(std::move(a));
    }

    // meta.xml
    pugi::xml_document doc;
    pugi::xml_parse_result r = doc.load_buffer(bank_xml, bank_size);
    if (!r) return fail(error, std::string("meta.xml: ") + r.description());
    pugi::xml_node root = doc.child("audio_bank");
    if (!root) return fail(error, "meta.xml: no <audio_bank>");
    out.name = root.attribute("name").as_string();
    out.id = hash(out.name.c_str());

    // stream overrides
    for (pugi::xml_node s : root.child("sounds").children("sound")) {
        audio_sound* a = const_cast<audio_sound*>(out.find_sound(hash(s.attribute("name").as_string())));
        if (!a) return fail(error, std::string("meta.xml: <sounds>: unknown sound '") + s.attribute("name").as_string() + "'");
        if (s.attribute("stream")) a->stream = s.attribute("stream").as_bool();
        if (a->vorbis && !a->stream) return fail(error, "meta.xml: <sounds>: '" + a->name + "' is Vorbis: streamed only (stream=\"false\" is not supported)");
    }

    // curves are stored in one vector: count them first so pointers stay valid
    size_t curves = 0;
    for (pugi::xml_node e : root.children("event"))
        for (pugi::xml_node l : e.children("layer"))
            curves += (l.child("volume").child("curve") ? 1 : 0) + (l.child("pitch").child("curve") ? 1 : 0) + (l.child("delay").child("curve") ? 1 : 0);
    for (pugi::xml_node e : root.children("event")) {
        curves += (e.child("lowpass").child("curve") ? 1 : 0) + (e.child("highpass").child("curve") ? 1 : 0);
        for (pugi::xml_node sn : e.children("send")) curves += sn.child("curve") ? 1 : 0;
    }
    out.curves.reserve(curves);

    size_t event_count = 0;
    for (pugi::xml_node e : root.children("event")) { (void)e; ++event_count; }
    out.events.reserve(event_count);

    for (pugi::xml_node e : root.children("event")) {
        audio_event_desc ev;
        ev.name = e.attribute("name").as_string();
        if (ev.name.empty()) return fail(error, "meta.xml: event without name");
        ev.id = hash(ev.name.c_str());
        if (out.find_event(ev.id)) return fail(error, "meta.xml: event '" + ev.name + "' twice (or hash collision)");
        std::string where = "meta.xml: event '" + ev.name + "'";
        ev.bank = &out;

        const char* bus = e.attribute("bus").as_string("master");
        ev.bus = buses.find(hash(bus));
        if (ev.bus == ~0u) return fail(error, where + ": unknown bus '" + bus + "'");

        std::string sp = e.attribute("spatial").as_string("pan");
        if (sp == "2d") ev.spatial = audio_spatial_2d;
        else if (sp == "pan") ev.spatial = audio_spatial_pan;
        else if (sp == "hrtf") ev.spatial = audio_spatial_hrtf;
        else return fail(error, where + ": spatial must be 2d / pan / hrtf");

        ev.priority_near = uint8_t(e.attribute("priority_near").as_uint(0));
        ev.priority_far  = uint8_t(e.attribute("priority_far").as_uint(ev.priority_near));
        ev.max_instances = e.attribute("max_instances").as_uint(8);
        ev.air = e.attribute("air").as_float(1.0f);
        if (ev.air < 0 || ev.air > 100) return fail(error, where + ": air must be 0 (none) .. 100 (distance multiplier)");
        ev.pending_timeout = uint32_t(e.attribute("pending_timeout").as_float(0.0f) * sample_rate);

        // filters on the instance: <lowpass cutoff="..."/> or <lowpass><curve parameter="..."> y in Hz </curve></lowpass>
        // (curves are checked once parameters are known, below)
        pugi::xml_node lp_node = e.child("lowpass"), hp_node = e.child("highpass");
        ev.lowpass_hz  = lp_node.attribute("cutoff").as_float(0.0f);
        ev.highpass_hz = hp_node.attribute("cutoff").as_float(0.0f);
        if (ev.lowpass_hz < 0 || ev.highpass_hz < 0 || (ev.lowpass_hz && ev.lowpass_hz < 10) || ev.highpass_hz > 20000)
            return fail(error, where + ": filter cutoff out of range (lowpass >= 10 Hz, highpass <= 20 kHz)");

        pugi::xml_node att = e.child("attenuation");
        ev.min_distance = att.attribute("min").as_float(1.0f);
        ev.max_distance = att.attribute("max").as_float(50.0f);
        if (ev.max_distance <= ev.min_distance) return fail(error, where + ": attenuation max must be greater than min");
        std::string curve = att.attribute("curve").as_string("inverse");
        if (curve != "inverse") return fail(error, where + ": attenuation curve '" + curve + "' is not supported (inverse)");

        for (pugi::xml_node p : e.child("parameters").children("parameter")) {
            audio_parameter_desc d;
            d.id = hash(p.attribute("name").as_string());
            d.min = p.attribute("min").as_float(0.0f);
            d.max = p.attribute("max").as_float(1.0f);
            d.def = p.attribute("default").as_float(d.min);
            if (d.id == k_audio_param_distance || d.id == k_audio_param_angle) return fail(error, where + ": 'distance' and 'angle' are built-in parameters");
            ev.parameters.push_back(d);
        }
        for (pugi::xml_node s : e.child("switches").children("switch")) {
            audio_switch_desc d;
            d.id = hash(s.attribute("name").as_string());
            for (pugi::xml_node v : s.children("value")) d.values.push_back(hash(v.attribute("name").as_string()));
            d.def = s.attribute("default") ? hash(s.attribute("default").as_string()) : (d.values.empty() ? 0 : d.values[0]);
            ev.switches.push_back(d);
        }

        // a pool: <sound name="..."/> or <random> of sounds; `name` may also come as an attribute (case sound="...")
        auto pool = [&](pugi::xml_node n, const char* attr, audio_choice& out_pool, bool& loop) -> bool {
            auto add = [&](const char* name, pugi::xml_node at) -> bool {
                const audio_sound* s = out.find_sound(hash(name));
                if (!s) return fail(error, where + ": unknown sound '" + name + "'");
                out_pool.sounds.push_back(s);
                if (at && at.attribute("loop").as_bool()) loop = true;
                return true;
            };
            if (attr && n.attribute(attr)) return add(n.attribute(attr).as_string(), pugi::xml_node());
            if (pugi::xml_node s = n.child("sound")) return add(s.attribute("name").as_string(), s);
            if (pugi::xml_node rn = n.child("random")) {
                for (pugi::xml_node s : rn.children("sound")) if (!add(s.attribute("name").as_string(), s)) return false;
                return !out_pool.sounds.empty() || fail(error, where + ": empty <random>");
            }
            return true;
        };

        struct { pugi::xml_node node; const audio_curve** dst; const char* tag; } fcv[2] = { { lp_node, &ev.lowpass_curve, "lowpass" }, { hp_node, &ev.highpass_curve, "highpass" } };
        for (auto& k : fcv) {
            pugi::xml_node c = k.node.child("curve");
            if (!c) continue;
            audio_curve curve_data;
            if (!parse_curve(c, curve_data, error, (where + " " + k.tag).c_str())) return false;
            bool known = curve_data.parameter == k_audio_param_distance || curve_data.parameter == k_audio_param_angle;
            for (const audio_parameter_desc& d : ev.parameters) if (d.id == curve_data.parameter) known = true;
            if (!known) return fail(error, where + ": " + k.tag + " curve parameter '" + c.attribute("parameter").as_string() + "' is not declared in <parameters>");
            for (uint32_t i = 0; i < curve_data.count; ++i)
                if (curve_data.y[i] < 10.0f || curve_data.y[i] > 24000.0f) return fail(error, where + ": " + k.tag + " curve y must be a frequency 10 .. 24000 Hz");
            out.curves.push_back(curve_data);
            *k.dst = &out.curves.back();
        }

        for (pugi::xml_node sn : e.children("send")) {
            if (ev.sends.size() == 2) return fail(error, where + ": more than 2 sends");
            audio_send_desc sd;
            const char* bus_name = sn.attribute("bus").as_string("");
            sd.bus = buses.find(hash(bus_name));
            if (sd.bus == ~0u || !buses.buses[sd.bus].reverb) return fail(error, where + ": send to '" + bus_name + "': not a reverb bus (init.xml: effect=\"reverb\")");
            sd.level = sn.attribute("level").as_float(0.0f);
            if (sd.level < 0) return fail(error, where + ": send level must be >= 0");
            if (pugi::xml_attribute sw = sn.attribute("switch")) {
                sd.switch_id = hash(sw.as_string());
                bool declared = false;
                for (const audio_switch_desc& d : ev.switches) if (d.id == sd.switch_id) declared = true;
                if (!declared) return fail(error, where + ": send switch '" + sw.as_string() + "' is not declared in <switches>");
                sd.switch_value = hash(sn.attribute("value").as_string());
            }
            if (pugi::xml_node c = sn.child("curve")) {
                audio_curve curve_data;
                if (!parse_curve(c, curve_data, error, (where + " send").c_str())) return false;
                bool known = curve_data.parameter == k_audio_param_distance || curve_data.parameter == k_audio_param_angle;
                for (const audio_parameter_desc& d : ev.parameters) if (d.id == curve_data.parameter) known = true;
                if (!known) return fail(error, where + ": send curve parameter '" + c.attribute("parameter").as_string() + "' is not declared in <parameters>");
                out.curves.push_back(curve_data);
                sd.curve = &out.curves.back();
            }
            ev.sends.push_back(sd);
        }

        for (pugi::xml_node l : e.children("layer")) {
            if (ev.layers.size() == 16) return fail(error, where + ": more than 16 layers");
            audio_layer_desc ld;
            ld.name = l.attribute("name").as_string();
            ld.volume_db = l.attribute("volume").as_float(0.0f);
            ld.chance = l.attribute("chance").as_float(1.0f);
            ld.delay = uint32_t(l.attribute("delay").as_float(0.0f) * sample_rate + 0.5f);
            ld.loop = l.attribute("loop").as_bool(false);
            pugi::xml_node rnd = l.child("random");
            ld.avoid_repeat = rnd.attribute("avoid_repeat").as_uint(0);
            if (ld.avoid_repeat > 8) ld.avoid_repeat = 8;

            if (pugi::xml_node sw = l.child("switch")) {
                const char* sname = sw.attribute("switch").as_string(sw.attribute("parameter").as_string());
                ld.switch_id = hash(sname);
                bool declared = false;
                for (const audio_switch_desc& d : ev.switches) if (d.id == ld.switch_id) declared = true;
                if (!declared) return fail(error, where + ": layer switch '" + sname + "' is not declared in <switches>");
                for (pugi::xml_node c : sw.children("case")) {
                    audio_choice ch;
                    ch.value = hash(c.attribute("value").as_string());
                    if (!pool(c, "sound", ch, ld.loop)) return false;
                    if (pugi::xml_node rr = c.child("random")) ld.avoid_repeat = rr.attribute("avoid_repeat").as_uint(ld.avoid_repeat);
                    ld.cases.push_back(std::move(ch));
                }
                if (pugi::xml_node d = sw.child("default")) if (!pool(d, "sound", ld.fallback, ld.loop)) return false;
            } else {
                if (!pool(l, nullptr, ld.pool, ld.loop)) return false;
                if (ld.pool.sounds.empty()) return fail(error, where + ": layer without sound");
            }

            if (pugi::xml_node p = l.child("pitch")) {
                ld.pitch_min = p.attribute("min").as_float(1.0f);
                ld.pitch_max = p.attribute("max").as_float(ld.pitch_min);
            }
            struct { const char* tag; const audio_curve** dst; } cv[3] = { { "volume", &ld.volume_curve }, { "pitch", &ld.pitch_curve }, { "delay", &ld.delay_curve } };
            for (auto& k : cv) {
                if (pugi::xml_node c = l.child(k.tag).child("curve")) {
                    audio_curve curve_data;
                    if (!parse_curve(c, curve_data, error, (where + " layer '" + ld.name + "' " + k.tag).c_str())) return false;
                    bool known = curve_data.parameter == k_audio_param_distance || curve_data.parameter == k_audio_param_angle;
                    for (const audio_parameter_desc& d : ev.parameters) if (d.id == curve_data.parameter) known = true;
                    if (!known) return fail(error, where + ": curve parameter '" + c.attribute("parameter").as_string() + "' is not declared in <parameters>");
                    out.curves.push_back(curve_data);
                    *k.dst = &out.curves.back();
                }
            }
            if (ld.loop) ev.loop = true;
            ev.layers.push_back(std::move(ld));
        }
        if (ev.layers.empty()) return fail(error, where + ": no layers");
        ev.virtual_behavior = ev.loop ? audio_virtual_update : audio_virtual_kill;
        if (pugi::xml_attribute v = e.attribute("virtual")) {
            std::string vb = v.as_string();
            if (vb == "kill") ev.virtual_behavior = audio_virtual_kill;
            else if (vb == "update") ev.virtual_behavior = audio_virtual_update;
            else if (vb == "restart") ev.virtual_behavior = audio_virtual_restart;
            else return fail(error, where + ": virtual must be kill / update / restart");
        }
        out.events.push_back(std::move(ev));
    }
    return true;
}

bool audio_sound_attach(audio_sound& s, std::vector<uint8_t>&& wav, std::string& error)
{
    s.data = std::move(wav);
    audio_adpcm::clip c;
    if (!audio_adpcm::parse_wav(s.data.data(), s.data.size(), c)) return fail(error, "sound '" + s.name + "': not an IMA ADPCM mono WAV");
    if (c.frames != s.frames || c.block_align != s.block_align) return fail(error, "sound '" + s.name + "': does not match sounds.xml (rebuild the bank)");
    if (size_t(c.blocks - s.data.data()) + size_t(c.block_count) * c.block_align > s.data.size()) return fail(error, "sound '" + s.name + "': truncated");
    s.clip = c;
    s.loaded = true;
    return true;
}
