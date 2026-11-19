// audio_policy.cpp - event instances and voice management (summary in audio_policy.h).
// Parts: init, parameters / layers (resolve, eval_layers), handles (post, stop, state), internals (release, commands),
//        score, start / launch, update (notes -> pending / starting -> score -> select -> apply -> live instances).

#include "audio_policy.h"
#include "audio_mixer.h"
#include "audio_stream.h"

#include <math.h>
#include <string.h>
#include <algorithm>
#include <chrono>

namespace {

float dot3(const float* a, const float* b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }

float envelope_db(const audio_sound& s, uint64_t frame, bool loop)
{
    if (s.envelope.empty() || !s.samples_per_block) return 0.0f;      // unknown: assume full scale
    if (loop) { uint8_t m = 255; for (uint8_t e : s.envelope) if (e < m) m = e; return -m / 2.55f; }
    uint64_t b = frame / s.samples_per_block;
    if (b >= s.envelope.size()) return -100.0f;
    return -s.envelope[b] / 2.55f;
}

const float k_silent = 1e-5f;      // below: a voice is not started / counted

} // namespace

float audio_policy::attenuation(float d, float mn, float mx)
{
    float c = d < mn ? mn : d > mx ? mx : d;
    return mn / (mn + (c - mn));
}

bool audio_policy::init(const audio_policy_desc& desc, audio_mixer* mixer, audio_stream_system* streams,
                        const audio_buses* buses, const audio_param_source* params)
{
    m_desc = desc;
    m_mixer = mixer;
    m_streams = streams;
    m_buses = buses;
    m_params = params ? *params : audio_param_source{};
    m_rng = desc.seed ? desc.seed : 1;
    uint32_t voices = desc.max_voices + desc.voice_reserve;
    if (!mixer || mixer->desc().max_voices < voices || mixer->desc().max_instances < desc.max_instances) return false;
    m_records.assign(desc.max_records, record());
    m_voice_owner.assign(voices, ~0u);
    m_free_voices.clear();
    m_free_instances.clear();
    for (uint32_t i = voices; i-- > 0;) m_free_voices.push_back(i);
    for (uint32_t i = desc.max_instances; i-- > 0;) m_free_instances.push_back(i);
    m_cand.reserve(desc.max_records);
    m_selected.assign(desc.max_records, 0);
    m_bus_used.assign(buses ? buses->buses.size() : 0, 0);
    m_stats = {};
    return true;
}

void audio_policy::shutdown()
{
    for (record& r : m_records)
        if (r.state != audio_instance_none)
            for (layer_state& l : r.layers) if (l.stream && m_streams) { m_streams->close(l.stream); l.stream = nullptr; }
    m_records.clear();
}

void audio_policy::set_listener(const float pos[3], const float right[3], const float up[3], const float forward[3])
{
    memcpy(m_listener[0], pos, 12); memcpy(m_listener[1], right, 12); memcpy(m_listener[2], up, 12); memcpy(m_listener[3], forward, 12);
}

float audio_policy::rnd()
{
    m_rng = m_rng * 1664525u + 1013904223u;
    return float((m_rng >> 8) & 0xffffff) / float(0x1000000);
}

// ---------------------------------------------------------------------------
// parameters, layers
// ---------------------------------------------------------------------------
float audio_policy::param(const record& r, uint32_t id) const
{
    if (id == k_audio_param_distance) return r.distance;
    if (id == k_audio_param_angle) return r.angle;
    for (const audio_parameter_desc& p : r.desc->parameters) {
        if (p.id != id) continue;
        uint32_t index = uint32_t(&r - m_records.data());
        float v = m_params.parameter ? m_params.parameter(m_params.user, r.object, index, r.generation, id, p.def) : p.def;
        return v < p.min ? p.min : v > p.max ? p.max : v;
    }
    return 0.0f;
}

static void locate(const float listener[4][3], const float pos[3], float rel_out[3], float& distance, float& angle)
{
    float rel[3] = { pos[0] - listener[0][0], pos[1] - listener[0][1], pos[2] - listener[0][2] };
    rel_out[0] = dot3(rel, listener[1]); rel_out[1] = dot3(rel, listener[2]); rel_out[2] = dot3(rel, listener[3]);
    distance = sqrtf(dot3(rel, rel));
    float c = distance > 1e-6f ? rel_out[2] / distance : 1.0f;
    c = c < -1.0f ? -1.0f : c > 1.0f ? 1.0f : c;
    angle = acosf(c) * 57.29578f;
}

// Once at post: chance, switch, random (avoid_repeat), random pitch, delay.
void audio_policy::resolve(record& r)
{
    const audio_event_desc& d = *r.desc;
    const float rate = float(m_mixer->desc().sample_rate);
    locate(m_listener, r.pos, r.rel, r.distance, r.angle);
    for (uint32_t l = 0; l < d.layers.size(); ++l) {
        const audio_layer_desc& L = d.layers[l];
        layer_state& s = r.layers[l];
        s = layer_state();
        if (L.chance < 1.0f && rnd() >= L.chance) continue;
        const audio_choice* pool = &L.pool;
        if (L.switch_id) {
            uint32_t def = 0;
            for (const audio_switch_desc& sw : d.switches) if (sw.id == L.switch_id) def = sw.def;
            uint32_t v = m_params.switch_value ? m_params.switch_value(m_params.user, r.object, L.switch_id, def) : def;
            pool = &L.fallback;
            for (const audio_choice& c : L.cases) if (c.value == v) { pool = &c; break; }
        }
        uint32_t n = uint32_t(pool->sounds.size());
        if (n == 0) continue;
        uint32_t pick = 0;
        if (n > 1) {
            uint32_t avoid = L.avoid_repeat < n ? L.avoid_repeat : n - 1;
            uint32_t allowed[64], count = 0;
            for (uint32_t i = 0; i < n && count < 64; ++i) {
                bool recent = false;
                for (uint32_t h = 0; h < avoid; ++h) if (L.history[h] == i) recent = true;
                if (!recent) allowed[count++] = i;
            }
            pick = count ? allowed[uint32_t(rnd() * count) % count] : uint32_t(rnd() * n) % n;
            for (uint32_t h = 7; h > 0; --h) L.history[h] = L.history[h - 1];
            L.history[0] = pick;
        }
        s.sound = pool->sounds[pick];
        s.pitch = L.pitch_max > L.pitch_min ? L.pitch_min + (L.pitch_max - L.pitch_min) * rnd() : L.pitch_min;
        float delay_s = L.delay_curve ? L.delay_curve->eval(param(r, L.delay_curve->parameter)) : 0.0f;
        s.delay = L.delay + uint32_t(delay_s > 0.0f ? delay_s * rate + 0.5f : 0.0f);
    }
    // sends that depend on a switch (e.g. tail_mode = reverb): decided once, like the layers
    for (uint32_t k = 0; k < d.sends.size() && k < 2; ++k) {
        const audio_send_desc& sd = d.sends[k];
        if (!sd.switch_id) { r.send_on[k] = true; continue; }
        uint32_t def = 0;
        for (const audio_switch_desc& sw : d.switches) if (sw.id == sd.switch_id) def = sw.def;
        uint32_t v = m_params.switch_value ? m_params.switch_value(m_params.user, r.object, sd.switch_id, def) : def;
        r.send_on[k] = v == sd.switch_value;
    }
}

// Every update: volume and pitch curves.
void audio_policy::eval_layers(record& r)
{
    const audio_event_desc& d = *r.desc;
    for (uint32_t l = 0; l < d.layers.size(); ++l) {
        layer_state& s = r.layers[l];
        if (!s.sound) continue;
        const audio_layer_desc& L = d.layers[l];
        float g = powf(10.0f, L.volume_db / 20.0f);
        if (L.volume_curve) { float c = L.volume_curve->eval(param(r, L.volume_curve->parameter)); g *= c > 0.0f ? c : 0.0f; }
        s.gain = g;
        float p = s.pitch;
        if (L.pitch_curve) { float c = L.pitch_curve->eval(param(r, L.pitch_curve->parameter)); p *= c > 0.01f ? c : 0.01f; }
        s.pitch_now = p;
    }
}

// ---------------------------------------------------------------------------
// handles
// ---------------------------------------------------------------------------
audio_instance_handle audio_policy::post(const audio_event_desc* desc, uint32_t object, const float pos[3])
{
    if (!desc || desc->layers.empty() || desc->layers.size() > 16) { ++m_stats.post_failed; return {}; }
    uint32_t idx = ~0u;
    for (uint32_t i = 0; i < m_records.size(); ++i) if (m_records[i].state == audio_instance_none) { idx = i; break; }
    if (idx == ~0u) {
        // all records taken: steal the weakest virtual one (score from the last update)
        for (uint32_t i = 0; i < m_records.size(); ++i) {
            const record& r = m_records[i];
            if (r.state != audio_instance_virtual) continue;
            if (idx == ~0u || r.prio < m_records[idx].prio || (r.prio == m_records[idx].prio && r.aud < m_records[idx].aud)) idx = i;
        }
        if (idx == ~0u) { ++m_stats.post_failed; return {}; }
        release(m_records[idx]);
        ++m_stats.record_steals;
    }
    record& r = m_records[idx];
    uint32_t gen = r.generation + 1;
    r = record();
    r.generation = gen;
    r.desc = desc;
    r.object = object;
    memcpy(r.pos, pos, 12);
    r.start = r.posted = m_mixer->time();
    resolve(r);
    bool available = true;
    for (uint32_t l = 0; l < desc->layers.size(); ++l) if (r.layers[l].sound && !r.layers[l].sound->available()) available = false;
    r.state = available ? audio_instance_virtual : audio_instance_pending;
    return { idx, gen };
}

audio_instance_state audio_policy::state(audio_instance_handle h) const
{
    if (h.index >= m_records.size() || m_records[h.index].generation != h.generation) return audio_instance_none;
    return m_records[h.index].state;
}

void audio_policy::set_position(audio_instance_handle h, const float pos[3])
{
    if (state(h) == audio_instance_none) return;
    memcpy(m_records[h.index].pos, pos, 12);
}

float audio_policy::audibility_db(audio_instance_handle h) const { return state(h) == audio_instance_none ? -1000.0f : m_records[h.index].aud; }
int   audio_policy::priority(audio_instance_handle h) const     { return state(h) == audio_instance_none ? 0 : m_records[h.index].prio; }
const audio_sound* audio_policy::layer_sound(audio_instance_handle h, uint32_t l) const { return state(h) == audio_instance_none || l >= 16 ? nullptr : m_records[h.index].layers[l].sound; }
uint32_t audio_policy::layer_voice(audio_instance_handle h, uint32_t l) const { return state(h) == audio_instance_none || l >= 16 ? ~0u : m_records[h.index].layers[l].voice; }

void audio_policy::stop(audio_instance_handle h, uint32_t fade)
{
    if (state(h) == audio_instance_none) return;
    record& r = m_records[h.index];
    r.stop_requested = true;
    switch (r.state) {
        case audio_instance_playing:  stop_voices(r, fade); r.state = audio_instance_fading; break;
        case audio_instance_pending:
        case audio_instance_starting:
        case audio_instance_virtual:  release(r); break;
        default: break;
    }
}

void audio_policy::stop_matching(uint32_t object, const audio_event_desc* desc, uint32_t fade, bool loops_only)
{
    for (uint32_t i = 0; i < m_records.size(); ++i) {
        record& r = m_records[i];
        if (r.state == audio_instance_none || r.state == audio_instance_fading) continue;
        if (object != ~0u && r.object != object) continue;
        if (desc && r.desc != desc) continue;
        if (loops_only && !r.desc->loop) continue;
        stop({ i, r.generation }, fade);
    }
}

void audio_policy::stop_bank(const audio_bank_data* bank, uint32_t fade)
{
    for (uint32_t i = 0; i < m_records.size(); ++i) {
        record& r = m_records[i];
        if (r.state != audio_instance_none && r.desc->bank == bank) stop({ i, r.generation }, fade);
    }
}

uint32_t audio_policy::count_bank(const audio_bank_data* bank) const
{
    uint32_t n = 0;
    for (const record& r : m_records) if (r.state != audio_instance_none && r.desc->bank == bank) ++n;
    return n;
}

// ---------------------------------------------------------------------------
// internals
// ---------------------------------------------------------------------------
void audio_policy::free_reserved(record& r)
{
    // voices reserved but never launched (starting) go straight back; launched ones come back via notes
    for (layer_state& l : r.layers) {
        if (l.voice != ~0u) { m_voice_owner[l.voice] = ~0u; m_free_voices.push_back(l.voice); l.voice = ~0u; }
        if (l.stream) { m_streams->close(l.stream); l.stream = nullptr; }
    }
    if (r.instance != ~0u) { m_free_instances.push_back(r.instance); r.instance = ~0u; }
}

void audio_policy::release(record& r)
{
    if (r.state == audio_instance_starting) free_reserved(r);
    for (layer_state& l : r.layers) if (l.stream) { m_streams->close(l.stream); l.stream = nullptr; }
    if (r.instance != ~0u && r.alive == 0) { m_free_instances.push_back(r.instance); r.instance = ~0u; }
    r.state = audio_instance_none;
    r.desc = nullptr;
}

void audio_policy::stop_voices(record& r, uint32_t fade)
{
    for (layer_state& l : r.layers) {
        if (l.voice == ~0u) continue;
        audio_mixer_cmd c = {}; c.type = audio_mixer_cmd_voice_stop; c.target = l.voice; c.stop.fade = fade;
        m_mixer->push(c);
    }
}

void audio_policy::send_spatial(const record& r)
{
    audio_mixer_cmd c = {}; c.type = audio_mixer_cmd_instance_spatial; c.target = r.instance;
    bool spatial = r.desc->spatial != audio_spatial_2d;
    c.spatial.position[0] = r.rel[0]; c.spatial.position[1] = r.rel[1]; c.spatial.position[2] = r.rel[2];
    c.spatial.distance = r.distance;
    c.spatial.attenuation = spatial ? attenuation(r.distance, r.desc->min_distance, r.desc->max_distance) : 1.0f;
    c.spatial.spatial = spatial;
    c.spatial.binaural = r.binaural;
    m_mixer->push(c);
}

void audio_policy::pick_binaural()
{
    m_bin.clear();
    for (uint32_t i = 0; i < m_records.size(); ++i) {
        record& r = m_records[i];
        const bool live = (r.state == audio_instance_playing || r.state == audio_instance_starting) && r.instance != ~0u;
        if (live && m_binaural && r.desc->spatial == audio_spatial_hrtf)                    // spatial="hrtf" events only
            m_bin.push_back({ r.aud + (r.binaural ? m_desc.hysteresis_db : 0.0f), i });     // the current ones keep it
        r.binaural = false;
    }
    const size_t n = m_bin.size() < m_desc.max_binaural ? m_bin.size() : m_desc.max_binaural;
    if (n < m_bin.size())
        std::nth_element(m_bin.begin(), m_bin.begin() + n, m_bin.end(), [](const std::pair<float, uint32_t>& a, const std::pair<float, uint32_t>& b) { return a.first > b.first; });
    for (size_t k = 0; k < n; ++k) m_records[m_bin[k].second].binaural = true;
}

bool audio_policy::binaural(audio_instance_handle h) const
{
    if (h.index >= m_records.size() || m_records[h.index].generation != h.generation) return false;
    const record& r = m_records[h.index];
    return r.state != audio_instance_none && r.binaural;
}

void audio_policy::send_filter(record& r, bool force)
{
    const audio_event_desc& d = *r.desc;
    const bool own = d.lowpass_hz || d.lowpass_curve || d.highpass_hz || d.highpass_curve;
    // air absorption (init.xml): every spatial instance by distance; with the event's own low-pass the lower cutoff wins
    const float air = m_buses && d.spatial != audio_spatial_2d ? m_buses->air_cutoff(r.distance, d.air) : 0.0f;
    if (!own && air == 0.0f && !(r.sent_lowpass > 0.0f)) return;     // never filtered and nothing now
    float lp = d.lowpass_curve ? d.lowpass_curve->eval(param(r, d.lowpass_curve->parameter)) : d.lowpass_hz;
    float hp = d.highpass_curve ? d.highpass_curve->eval(param(r, d.highpass_curve->parameter)) : d.highpass_hz;
    if (lp >= 20000.0f) lp = 0.0f;                       // fully open: the stage is off (no cost, bit-exact)
    if (air > 0.0f && (lp <= 0.0f || air < lp)) lp = air;
    // resend only on a change the ear could notice: on / off, or more than 1 % of the cutoff
    auto changed = [](float now, float sent) { return (now > 0) != (sent > 0) || (now > 0 && fabsf(now - sent) > 0.01f * sent); };
    if (!force && !changed(lp, r.sent_lowpass) && !changed(hp, r.sent_highpass)) return;
    audio_mixer_cmd c = {}; c.type = audio_mixer_cmd_instance_filter; c.target = r.instance;
    c.filter.lowpass_hz = lp; c.filter.highpass_hz = hp;
    m_mixer->push(c);
    r.sent_lowpass = lp; r.sent_highpass = hp;
}

void audio_policy::send_sends(record& r, bool force)
{
    const audio_event_desc& d = *r.desc;
    float att = d.spatial != audio_spatial_2d ? attenuation(r.distance, d.min_distance, d.max_distance) : 1.0f;
    for (uint32_t k = 0; k < d.sends.size() && k < 2; ++k) {
        const audio_send_desc& sd = d.sends[k];
        float g = r.send_on[k] ? (sd.curve ? sd.curve->eval(param(r, sd.curve->parameter)) : sd.level) : 0.0f;
        if (g < 0) g = 0;
        g *= att;                                           // the reverb hears what reaches the listener (far = quieter)
        bool changed = force || (g > 0) != (r.sent_send[k] > 0) || fabsf(g - r.sent_send[k]) > 0.01f * r.sent_send[k] + 1e-5f;
        if (!changed) continue;
        audio_mixer_cmd c = {}; c.type = audio_mixer_cmd_instance_send; c.target = r.instance;
        c.send.slot = k; c.send.bus = sd.bus; c.send.gain = g;
        m_mixer->push(c);
        r.sent_send[k] = g;
    }
}

void audio_policy::send_curves(record& r)
{
    for (layer_state& s : r.layers) {
        if (s.voice == ~0u) continue;
        if (fabsf(s.gain - s.sent_gain) > 1e-4f) {
            audio_mixer_cmd c = {}; c.type = audio_mixer_cmd_voice_gain; c.target = s.voice; c.gain.gain = s.gain;
            m_mixer->push(c); s.sent_gain = s.gain;
        }
        if (fabsf(s.pitch_now - s.sent_pitch) > 1e-4f) {
            audio_mixer_cmd c = {}; c.type = audio_mixer_cmd_voice_pitch; c.target = s.voice; c.pitch.step = uint32_t(lrintf(s.pitch_now * 65536.0f));
            m_mixer->push(c); s.sent_pitch = s.pitch_now;
        }
    }
}

// Score at mixer time `now`. Returns false if a one-shot has nothing left to play.
bool audio_policy::score(record& r, uint64_t now)
{
    const audio_event_desc& d = *r.desc;
    if (r.object != ~0u && m_params.position) { float p[3]; if (m_params.position(m_params.user, r.object, p)) memcpy(r.pos, p, 12); }
    locate(m_listener, r.pos, r.rel, r.distance, r.angle);
    eval_layers(r);

    float att_db = 0.0f;
    if (d.spatial != audio_spatial_2d) {
        float t = (r.distance - d.min_distance) / (d.max_distance - d.min_distance);
        t = t < 0 ? 0 : t > 1 ? 1 : t;
        r.prio = int(lrintf(d.priority_near + (float(d.priority_far) - float(d.priority_near)) * t));
        att_db = r.distance > d.max_distance ? -1000.0f : 20.0f * log10f(attenuation(r.distance, d.min_distance, d.max_distance));
    } else {
        r.prio = d.priority_near;
    }

    double power = 0.0;
    bool any_left = false;
    r.need = 0;
    int64_t elapsed = int64_t(now - r.start);
    for (uint32_t l = 0; l < d.layers.size(); ++l) {
        const layer_state& s = r.layers[l];
        if (!s.sound) continue;
        bool loop = d.layers[l].loop;
        int64_t off = int64_t(double(elapsed - int64_t(s.delay)) * s.pitch_now);
        if (!loop && off >= int64_t(s.sound->frames)) continue;                       // layer done
        any_left = true;
        bool counts = s.gain > k_silent || s.sound->stream || s.voice != ~0u;          // silent resident loops wait without a voice
        if (!counts) continue;
        ++r.need;
        if (s.gain <= k_silent) continue;
        float a = envelope_db(*s.sound, off < 0 ? 0 : uint64_t(off), loop) + 20.0f * log10f(s.gain) + att_db;
        if (a > -300.0f) power += pow(10.0, a / 10.0);
    }
    r.aud = power > 0 ? float(10.0 * log10(power)) : -1000.0f;
    return any_left;
}

bool audio_policy::start(uint32_t index, uint64_t now)
{
    record& r = m_records[index];
    const audio_event_desc& d = *r.desc;
    if (m_free_instances.empty() || m_free_voices.size() < r.need) { ++m_stats.waits_for_voices; return false; }
    if (d.virtual_behavior == audio_virtual_restart) r.start = now;      // also for an event that was virtual from its post

    r.instance = m_free_instances.back(); m_free_instances.pop_back();
    audio_mixer_cmd c = {}; c.type = audio_mixer_cmd_instance_start; c.target = r.instance; c.instance.bus = d.bus;
    m_mixer->push(c);
    r.binaural = false;                                     // pick_binaural (same update, same batch) decides before the first period
    send_spatial(r);
    send_filter(r, true);                                   // before the first period: no unfiltered start
    send_sends(r, true);

    bool waits = false;
    int64_t elapsed = int64_t(now - r.start);
    for (uint32_t l = 0; l < d.layers.size(); ++l) {
        layer_state& s = r.layers[l];
        if (!s.sound) continue;
        bool loop = d.layers[l].loop;
        int64_t off = int64_t(double(elapsed - int64_t(s.delay)) * s.pitch_now);
        uint64_t len = s.sound->frames;
        if (!loop && off >= int64_t(len)) continue;
        if (s.gain <= k_silent && !s.sound->stream) continue;                           // silent: not started (a loop may start later)
        uint32_t v = m_free_voices.back(); m_free_voices.pop_back();
        m_voice_owner[v] = index;
        s.voice = v;
        if (s.sound->stream) {
            uint64_t from = off > 0 ? (loop && len ? uint64_t(off) % len : uint64_t(off)) : 0;
            s.stream = m_streams ? m_streams->open(s.sound->path.c_str(), from, loop) : nullptr;
            waits = true;
        }
    }
    r.state = audio_instance_starting;
    if (!waits) launch_voices(index, now);
    return true;
}

bool audio_policy::launch_layer(uint32_t index, uint32_t l, uint64_t now, bool late)
{
    record& r = m_records[index];
    layer_state& s = r.layers[l];
    bool loop = r.desc->layers[l].loop;
    if (s.sound->stream && (!s.stream || m_streams->state(s.stream) != audio_stream_ready)) {
        // the stream failed to open: the layer is dropped
        if (s.stream) { m_streams->close(s.stream); s.stream = nullptr; }
        m_voice_owner[s.voice] = ~0u; m_free_voices.push_back(s.voice); s.voice = ~0u;
        s.sound = nullptr;
        return false;
    }
    int64_t elapsed = int64_t(now - r.start);
    int64_t wait = int64_t(s.delay) - elapsed;                                          // > 0: the layer has not started yet
    int64_t off = wait > 0 ? 0 : int64_t(double(-wait) * s.pitch_now);
    if (loop && s.sound->frames) off %= int64_t(s.sound->frames);
    audio_mixer_cmd c = {}; c.type = audio_mixer_cmd_voice_play; c.target = s.voice;
    c.play.clip = s.sound->stream ? nullptr : &s.sound->clip;
    c.play.stream = s.sound->stream ? s.stream : nullptr;
    c.play.instance = r.instance;
    c.play.delay = wait > 0 ? uint32_t(wait) : 0;
    c.play.offset = uint64_t(off);
    c.play.step = uint32_t(lrintf(s.pitch_now * 65536.0f));
    c.play.gain = s.gain;
    c.play.fade_in = ((r.ever_played || late) && off > 0) ? m_desc.return_fade : 0;
    c.play.loop = loop;
    m_mixer->push(c);
    s.sent_gain = s.gain;
    s.sent_pitch = s.pitch_now;
    ++r.alive;
    return true;
}

void audio_policy::launch_voices(uint32_t index, uint64_t now)
{
    record& r = m_records[index];
    bool returning = r.ever_played;
    r.alive = 0;
    for (uint32_t l = 0; l < r.desc->layers.size(); ++l) if (r.layers[l].voice != ~0u) launch_layer(index, l, now, false);
    r.state = audio_instance_playing;
    if (returning) ++m_stats.returns; else ++m_stats.started;
    r.ever_played = true;
    if (r.alive == 0) {             // nothing could be launched
        if (r.instance != ~0u) { m_free_instances.push_back(r.instance); r.instance = ~0u; }
        release(r);
        ++m_stats.ended;
    }
}

// ---------------------------------------------------------------------------
// update
// ---------------------------------------------------------------------------
void audio_policy::update()
{
    const uint64_t now = m_mixer->time();
    using clock = std::chrono::steady_clock;
    clock::time_point tp = clock::now();
    auto lap = [&](int phase) { const clock::time_point t = clock::now(); m_phase_us[phase] = std::chrono::duration<double, std::micro>(t - tp).count(); tp = t; };

    // a) mixer notifications: voices back to the pool, instances done when all their voices are done
    audio_mixer_note n;
    while (m_mixer->pop_note(n)) {
        if (n.voice >= m_voice_owner.size()) continue;
        uint32_t idx = m_voice_owner[n.voice];
        m_voice_owner[n.voice] = ~0u;
        m_free_voices.push_back(n.voice);
        if (idx == ~0u) continue;
        record& r = m_records[idx];
        for (layer_state& s : r.layers) if (s.voice == n.voice) {
            s.voice = ~0u;
            if (s.stream) { m_streams->close(s.stream); s.stream = nullptr; }          // its voice is done: safe to close
            if (r.state == audio_instance_playing && n.reason == audio_mixer_note_ended) s.sound = nullptr;   // a one-shot layer is over
        }
        if (r.alive > 0 && --r.alive > 0) continue;
        if (r.state == audio_instance_fading) {
            if (r.instance != ~0u) { m_free_instances.push_back(r.instance); r.instance = ~0u; }
            if (r.stop_requested || r.desc->virtual_behavior == audio_virtual_kill) { release(r); ++m_stats.kills; }
            else r.state = audio_instance_virtual;
        } else if (r.state == audio_instance_playing) {
            // all launched voices ended; the instance stays if a silent loop layer may still start (score decides)
            bool waiting_loop = false;
            for (uint32_t l = 0; l < r.desc->layers.size(); ++l) if (r.layers[l].sound && r.desc->layers[l].loop) waiting_loop = true;
            if (!waiting_loop) {
                if (r.instance != ~0u) { m_free_instances.push_back(r.instance); r.instance = ~0u; }
                release(r); ++m_stats.ended;
            }
        }
    }

    // b) pending: wait for the bank's sounds (one-shots up to the timeout), starting: wait for the streams
    for (record& r : m_records) {
        if (r.state == audio_instance_pending) {
            bool ok = true;
            for (uint32_t l = 0; l < r.desc->layers.size(); ++l) if (r.layers[l].sound && !r.layers[l].sound->available()) ok = false;
            if (ok) { r.state = audio_instance_virtual; r.start = now; continue; }          // late, but from its beginning
            uint64_t limit = r.desc->pending_timeout ? r.desc->pending_timeout : (r.desc->loop ? ~0ull : m_desc.pending_timeout);
            if (now - r.posted > limit) { release(r); ++m_stats.pending_timeouts; }
        }
    }
    for (uint32_t i = 0; i < m_records.size(); ++i) {
        record& r = m_records[i];
        if (r.state != audio_instance_starting) continue;
        bool ready = true;
        for (layer_state& s : r.layers) if (s.stream && m_streams->state(s.stream) == audio_stream_requested) ready = false;
        if (ready) launch_voices(i, now);
    }

    lap(0);
    // c) score
    m_cand.clear();
    for (uint32_t i = 0; i < m_records.size(); ++i) {
        record& r = m_records[i];
        if (r.state != audio_instance_virtual && r.state != audio_instance_playing && r.state != audio_instance_starting) continue;
        if (!score(r, now)) {
            if (r.state == audio_instance_virtual) { release(r); ++m_stats.ended; }   // a virtual one-shot ran out of time
            continue;
        }
        m_cand.push_back(i);
    }
    auto live = [&](uint32_t i) { return m_records[i].state == audio_instance_playing || m_records[i].state == audio_instance_starting; };
    lap(1);
    auto eff  = [&](uint32_t i) { return m_records[i].aud + (live(i) ? m_desc.hysteresis_db : 0.0f); };
    auto effp = [&](uint32_t i) { return m_records[i].prio + (live(i) ? int(m_desc.hysteresis_priority) : 0); };
    std::sort(m_cand.begin(), m_cand.end(), [&](uint32_t a, uint32_t b) {
        if (effp(a) != effp(b)) return effp(a) > effp(b);
        return eff(a) > eff(b);
    });

    // d) select: threshold -> max_instances of the event -> bus limits (with parents) -> global voice budget
    uint32_t budget = m_desc.max_voices;
    std::fill(m_bus_used.begin(), m_bus_used.end(), 0u);
    std::vector<std::pair<const audio_event_desc*, uint32_t>> per_desc;     // small: distinct events in the scene
    for (uint32_t i : m_cand) {
        record& r = m_records[i];
        m_selected[i] = 0;
        if (eff(i) <= m_desc.threshold_db) continue;
        uint32_t* cnt = nullptr;
        for (auto& p : per_desc) if (p.first == r.desc) { cnt = &p.second; break; }
        if (!cnt) { per_desc.push_back({ r.desc, 0 }); cnt = &per_desc.back().second; }
        if (*cnt >= r.desc->max_instances) continue;
        uint32_t need = r.need;
        if (need > budget) continue;
        bool bus_ok = true;
        if (m_buses && r.desc->bus < m_bus_used.size()) {
            for (uint32_t b = r.desc->bus;; b = m_buses->buses[b].parent) {
                uint32_t lim = m_buses->buses[b].voices;
                if (lim && m_bus_used[b] + need > lim) { bus_ok = false; break; }
                if (b == 0) break;
            }
            if (bus_ok) for (uint32_t b = r.desc->bus;; b = m_buses->buses[b].parent) { m_bus_used[b] += need; if (b == 0) break; }
        }
        if (!bus_ok) continue;
        budget -= need; ++*cnt; m_selected[i] = 1;
    }

    // e) apply
    for (uint32_t i : m_cand) {
        record& r = m_records[i];
        if (m_selected[i]) continue;
        if (r.state == audio_instance_playing) {
            if (r.alive) {
                stop_voices(r, m_desc.steal_fade);
                r.state = audio_instance_fading;
                if (eff(i) <= m_desc.threshold_db) ++m_stats.culls; else ++m_stats.steals;
            } else {                                                                     // only silent loops left
                if (r.instance != ~0u) { m_free_instances.push_back(r.instance); r.instance = ~0u; }
                if (r.desc->virtual_behavior == audio_virtual_kill) { release(r); ++m_stats.kills; } else r.state = audio_instance_virtual;
            }
        } else if (r.state == audio_instance_starting) {
            free_reserved(r);
            if (r.desc->virtual_behavior == audio_virtual_kill) { r.state = audio_instance_virtual; release(r); ++m_stats.kills; }
            else r.state = audio_instance_virtual;
        } else if (r.state == audio_instance_virtual && r.desc->virtual_behavior == audio_virtual_kill) {
            release(r); ++m_stats.kills;
        }
    }
    for (uint32_t i : m_cand) if (m_selected[i] && m_records[i].state == audio_instance_virtual) start(i, now);

    lap(2);
    // f) live instances: position, curves, silent loops that became audible
    pick_binaural();
    uint32_t audible = 0;
    for (uint32_t i = 0; i < m_records.size(); ++i) {
        record& r = m_records[i];
        if ((r.state != audio_instance_playing && r.state != audio_instance_starting) || r.instance == ~0u) continue;
        send_spatial(r);
        send_filter(r, false);
        send_sends(r, false);
        if (r.state == audio_instance_playing) {
            for (uint32_t l = 0; l < r.desc->layers.size(); ++l) {
                layer_state& s = r.layers[l];
                if (!s.sound || s.voice != ~0u || !r.desc->layers[l].loop || s.gain <= k_silent || s.sound->stream) continue;
                if (m_free_voices.empty()) { ++m_stats.waits_for_voices; continue; }
                s.voice = m_free_voices.back(); m_free_voices.pop_back();
                m_voice_owner[s.voice] = i;
                launch_layer(i, l, now, true);
                ++m_stats.late_layers;
            }
            send_curves(r);
        }
        audible += r.need;
    }
    m_stats.voices_audible = audible;
    m_mixer->commit();
    lap(3);
}

void audio_policy::bus_activity(uint32_t* counts, uint32_t bus_count) const
{
    for (uint32_t i = 0; i < bus_count; ++i) counts[i] = 0;
    for (const record& r : m_records)
        if ((r.state == audio_instance_playing || r.state == audio_instance_starting) && r.instance != ~0u && r.desc->bus < bus_count)
            ++counts[r.desc->bus];
}

size_t audio_policy::memory_bytes() const
{
    return m_records.capacity() * sizeof(record) + (m_free_voices.capacity() + m_free_instances.capacity() + m_voice_owner.capacity()
         + m_cand.capacity() + m_bus_used.capacity()) * sizeof(uint32_t) + m_selected.capacity() + m_bin.capacity() * sizeof(m_bin[0]);
}

void audio_policy::debug_records(void (*fn)(void* user, const audio_policy_record_info& r), void* user) const
{
    for (uint32_t i = 0; i < m_records.size(); ++i) {
        const record& r = m_records[i];
        if (r.state == audio_instance_none || !r.desc) continue;
        audio_policy_record_info info;
        info.index = i; info.generation = r.generation; info.desc = r.desc; info.object = r.object; info.state = r.state;
        info.rel[0] = r.rel[0]; info.rel[1] = r.rel[1]; info.rel[2] = r.rel[2];
        info.distance = r.distance; info.audibility_db = r.aud; info.priority = r.prio;
        info.voices = 0;
        for (uint32_t l = 0; l < r.desc->layers.size() && l < 16; ++l) info.voices += r.layers[l].voice != ~0u;
        info.binaural = r.binaural;
        info.lowpass_hz = r.sent_lowpass; info.highpass_hz = r.sent_highpass;
        info.send[0] = r.sent_send[0]; info.send[1] = r.sent_send[1];
        fn(user, info);
    }
}

audio_policy_stats audio_policy::stats() const
{
    audio_policy_stats s = m_stats;
    s.records = s.pending = s.playing = s.starting = s.virtual_ = s.fading = 0;
    for (const record& r : m_records) {
        if (r.state == audio_instance_none) continue;
        ++s.records;
        if (r.state == audio_instance_pending) ++s.pending;
        else if (r.state == audio_instance_playing) ++s.playing;
        else if (r.state == audio_instance_starting) ++s.starting;
        else if (r.state == audio_instance_virtual) ++s.virtual_;
        else if (r.state == audio_instance_fading) ++s.fading;
    }
    s.voices_used = uint32_t(m_voice_owner.size() - m_free_voices.size());
    for (const record& r : m_records) s.binaural += r.state != audio_instance_none && r.binaural;
    s.instances_used = uint32_t(m_desc.max_instances - m_free_instances.size());
    return s;
}
