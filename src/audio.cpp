// audio.cpp - the public API glued over the internals (summary in audio.h).
// Parts: impl (state), file IO, parameter source, init / shutdown, update, banks (register / load thread / deferred unload),
//        mixer buses, objects / listener, events, parameters, geometry, debug.

#include "audio.h"
#include "audio_bank.h"
#include "audio_policy.h"
#include "audio_mixer.h"
#include "audio_stream.h"
#include "audio_geometry.h"
#include "audio_hrtf.h"

#include <stdio.h>
#include <math.h>
#include <string.h>
#include <memory>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <deque>
#include <unordered_map>
#include <chrono>

namespace {

void*  stdio_open(void*, const char* path) { return fopen(path, "rb"); }
void   stdio_close(void*, void* f) { fclose((FILE*)f); }
size_t stdio_read_at(void*, void* f, uint64_t off, void* dst, size_t size)
{
#if defined(_WIN32)
    if (_fseeki64((FILE*)f, int64_t(off), SEEK_SET)) return 0;
#else
    if (fseeko((FILE*)f, off_t(off), SEEK_SET)) return 0;
#endif
    return fread(dst, 1, size, (FILE*)f);
}
uint64_t stdio_size(void*, void* f)
{
#if defined(_WIN32)
    if (_fseeki64((FILE*)f, 0, SEEK_END)) return 0;
    return uint64_t(_ftelli64((FILE*)f));
#else
    if (fseeko((FILE*)f, 0, SEEK_END)) return 0;
    return uint64_t(ftello((FILE*)f));
#endif
}

bool read_all(const audio_file_io& io, const std::string& path, std::vector<uint8_t>& out)
{
    out.clear();
    void* f = io.open(io.user, path.c_str());
    if (!f) return false;
    for (;;) {
        size_t at = out.size();
        out.resize(at + 65536);
        size_t got = io.read_at(io.user, f, at, out.data() + at, 65536);
        out.resize(at + got);
        if (got < 65536) break;
    }
    io.close(io.user, f);
    return true;
}

// v' = v + 2 * cross(q.xyz, cross(q.xyz, v) + w * v)
void rotate(const quat& q, const float v[3], float out[3])
{
    float t[3] = { q.y * v[2] - q.z * v[1] + q.w * v[0], q.z * v[0] - q.x * v[2] + q.w * v[1], q.x * v[1] - q.y * v[0] + q.w * v[2] };
    out[0] = v[0] + 2.0f * (q.y * t[2] - q.z * t[1]);
    out[1] = v[1] + 2.0f * (q.z * t[0] - q.x * t[2]);
    out[2] = v[2] + 2.0f * (q.x * t[1] - q.y * t[0]);
}

const uint32_t k_global_object = ~0u;

double wall_us() { return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

// sum / max / count of a value over a profiling window
struct prof_acc {
    double sum = 0, max = 0; uint32_t n = 0;
    void add(double v) { sum += v; if (v > max) max = v; ++n; }
    audio_timing_t timing(double scale = 1.0) const { return { n ? float(sum / n * scale) : 0.0f, float(max * scale) }; }
};

} // namespace

// ---------------------------------------------------------------------------
// state
// ---------------------------------------------------------------------------
struct audio::impl {
    audio_desc_t            desc;
    audio_file_io           io                  = {};
    std::string             error;
    bool                    binaural_ok         = false;    // HRTF data loaded or an own plugin

    audio_mixer             mixer;
    audio_stream_system     streams;
    audio_policy            policy;
    audio_buses             buses;
    audio_geometry_registry geometry;

    struct object {
        uint32_t            generation          = 0;
        bool                used                = false;
        std::string         name;
        float               pos[3]              = { 0, 0, 0 };
        quat                rot;
        float               vel[3]              = { 0, 0, 0 };     // for doppler (spatial plugin), not used by the built-in one
        std::vector<std::pair<uint32_t, float>>    params;
        std::vector<std::pair<uint32_t, uint32_t>> switches;
    };
    struct instance_override { 
        uint32_t index, generation, id; 
        float value; 
    };

    std::vector<object>                         objects;
    std::vector<std::pair<uint32_t, float>>     global_params;
    std::vector<std::pair<uint32_t, uint32_t>>  global_switches;
    std::vector<instance_override>              overrides;
    audio_object_t                              listener;
    float                                       listener_basis[4][3] = { { 0, 0, 0 }, { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, -1 } };

    enum bank_state_internal { bank_registered, bank_loading, bank_ready, bank_error };
    struct bank {
        uint32_t            generation          = 0;
        bool                used                = false;
        audio_bank_data     data;
        bank_state_internal state               = bank_registered;
        int                 refcount            = 0;
        bool                unloading           = false;
        bool                remove              = false;
        bool                job_in_flight       = false;
        std::string         folder;                         // register_bank's name: reload_banks reads it again
        // event descriptions replaced by reload_banks: instances stopped by it may still read them while fading
        std::vector<std::pair<std::vector<audio_event_desc>, std::vector<audio_curve>>> retired;
    };

    std::vector<std::unique_ptr<bank>> banks;
    std::unordered_map<uint32_t, const audio_event_desc*> events;

    struct load_job {
        bank*               b;
        double              queued_us           = 0;            // profiling: load_bank -> attached
        std::vector<std::pair<audio_sound*, std::vector<uint8_t>>> files;
        std::string         error;
    };
    std::thread             loader;
    std::mutex              mutex;
    std::condition_variable cv;
    std::deque<load_job*>   queue;
    std::vector<load_job*>  done;
    bool                    running             = false;

    struct bus_state { float volume = 1, from = 1, to = 1, t = 0, duration = 0; bool mute = false, pause = false; };
    std::vector<bus_state>  bus;
    std::vector<audio_duck_state> duck;                 // per bus as a ducking target
    std::vector<float>      sent_volume;                // volume x duck last sent to the mixer
    std::vector<uint32_t>   bus_count;                  // scratch: playing instances per bus
    std::vector<uint8_t>    bus_active;
    float       bus_out(uint32_t i) const { return bus[i].volume * duck[i].gain; }      // what the mixer gets

    audio_policy_stats      before = {}, delta_base = {};

    // profiling: game / IO thread windows (1 s, closed in update), peaks since init / reset_stats
    prof_acc                win_update, win_policy[4];
    double                  win_start           = 0;
    audio_timing_t          pub_update = {}, pub_policy[4] = {}, pub_io_read = {}, pub_io_decode = {}, pub_stream_start = {};
    float                   pub_io_bytes        = 0;
    audio_stream_system::io_totals io_prev      = {};
    float                   bank_load_last = 0, bank_load_max = 0;
    uint32_t                peak_voices = 0, peak_instances = 0, peak_records = 0, peak_streams = 0;
    uint32_t                peak_memory         = 0;
    audio_memory_t          memory() const;
    void                    profile_frame(double update_us);       // end of update(): windows, peaks
    audio_policy_stats      last_delta = {};

    // ----- helpers
    uint32_t    key(audio_object_t o) const { return o.valid() ? ((o.generation & 0xffffu) << 16) | (o.index & 0xffffu) : k_global_object; }
    object*     obj(audio_object_t o)       { return (o.index < objects.size() && objects[o.index].used && objects[o.index].generation == o.generation) ? &objects[o.index] : nullptr; }
    object*     obj_by_key(uint32_t k)      {
        if (k == k_global_object) return nullptr;
        uint32_t i = k & 0xffffu;
        return (i < objects.size() && objects[i].used && (objects[i].generation & 0xffffu) == (k >> 16)) ? &objects[i] : nullptr;
    }
    bank*       bank_of(audio_bank_t b) { return (b.index < banks.size() && banks[b.index]->used && banks[b.index]->generation == b.generation) ? banks[b.index].get() : nullptr; }
    uint32_t    frames(float seconds) const { return seconds > 0.0f ? uint32_t(seconds * float(desc.sample_rate) + 0.5f) : 0; }
    void        bus_config(uint32_t index) {
        audio_mixer_cmd c = {}; c.type = audio_mixer_cmd_bus_config; c.target = index;
        c.bus.parent = index ? buses.buses[index].parent : 0;
        c.bus.volume = bus_out(index); c.bus.mute = bus[index].mute; c.bus.pause = bus[index].pause;
        sent_volume[index] = c.bus.volume;
        mixer.push(c);
    }
    void        reverb_cmd(uint32_t bus_index, const audio_reverb_preset& p) {
        audio_mixer_cmd c = {}; c.type = audio_mixer_cmd_bus_reverb; c.target = bus_index;
        c.reverb.decay = p.decay; c.reverb.predelay = p.predelay; c.reverb.damping = p.damping; c.reverb.size = p.size; c.reverb.level = p.level;
        mixer.push(c);
    }
    void        loader_thread();
    void        free_bank_data(bank& b) {
        for (audio_sound& s : b.data.sounds) { std::vector<uint8_t>().swap(s.data); s.loaded = false; s.clip = audio_adpcm::clip(); }
    }
};

// ---------------------------------------------------------------------------
// parameter source for the policy: instance override -> object -> global -> default
// ---------------------------------------------------------------------------
static float src_parameter(void* user, uint32_t object, uint32_t instance, uint32_t generation, uint32_t id, float def)
{
    audio::impl* m = (audio::impl*)user;
    for (const auto& o : m->overrides) if (o.index == instance && o.generation == generation && o.id == id) return o.value;
    if (audio::impl::object* ob = m->obj_by_key(object)) for (const auto& p : ob->params) if (p.first == id) return p.second;
    for (const auto& p : m->global_params) if (p.first == id) return p.second;
    return def;
}

static uint32_t src_switch(void* user, uint32_t object, uint32_t id, uint32_t def)
{
    audio::impl* m = (audio::impl*)user;
    if (audio::impl::object* ob = m->obj_by_key(object)) for (const auto& p : ob->switches) if (p.first == id) return p.second;
    for (const auto& p : m->global_switches) if (p.first == id) return p.second;
    return def;
}

static bool src_position(void* user, uint32_t object, float out[3])
{
    audio::impl* m = (audio::impl*)user;
    audio::impl::object* ob = m->obj_by_key(object);
    if (!ob) return false;                          // destroyed: one-shots finish at the last position
    memcpy(out, ob->pos, 12);
    return true;
}

// ---------------------------------------------------------------------------
// init / shutdown
// ---------------------------------------------------------------------------
bool audio::init(const audio_desc_t& desc)
{
    if (m_impl) shutdown();
    m_impl = new impl();
    impl& m = *m_impl;
    m.desc = desc;
    m.io = desc.io ? *desc.io : audio_file_io{ nullptr, stdio_open, stdio_read_at, stdio_close, stdio_size };

    // init bank: buses
    std::string root = desc.root_path ? desc.root_path : "";
    std::vector<uint8_t> xml;
    std::string init_path = root + (desc.init_bank ? desc.init_bank : "init") + ".json";
    if (!read_all(m.io, init_path, xml)) { m.error = "cannot read " + init_path; return false; }
    if (!audio_parse_init((const char*)xml.data(), xml.size(), m.buses, m.error)) return false;
    m.bus.assign(m.buses.buses.size(), impl::bus_state());
    m.duck.assign(m.buses.buses.size(), audio_duck_state());
    m.sent_volume.assign(m.buses.buses.size(), -1.0f);
    m.bus_count.assign(m.buses.buses.size(), 0);
    m.bus_active.assign(m.buses.buses.size(), 0);
    for (uint32_t i = 0; i < m.buses.buses.size(); ++i) m.bus[i].volume = m.bus[i].to = m.buses.buses[i].volume;

    audio_mixer_desc md;
    md.sample_rate   = desc.sample_rate;
    md.period_frames = desc.period_frames;
    md.max_voices    = desc.max_voices + desc.voice_reserve;
    md.max_instances = desc.max_instances;
    md.max_buses     = uint32_t(m.buses.buses.size());
    uint32_t reverb_buses = 0;
    for (const audio_bus_desc& b : m.buses.buses) reverb_buses += b.reverb;
    md.max_reverbs   = reverb_buses * 2 > 0 ? reverb_buses * 2 : 1;          // current + ringing out per reverb bus
    // spatial plugin: the game's own, else HRTF when hrtf.bin is there (a broken file: last_error, panning), else panning
    if (desc.plugin) {
        md.plugin = *desc.plugin;
        m.binaural_ok = true;
    } else if (desc.hrtf_file && *desc.hrtf_file) {
        std::vector<uint8_t> bin;
        const std::string hrtf_path = root + desc.hrtf_file;
        if (read_all(m.io, hrtf_path, bin)) {
            audio_hrtf_data hd;
            std::string err;
            if (!hd.parse(bin.data(), bin.size(), err)) m.error = hrtf_path + ": " + err + " - panning only";
            else if (hd.sample_rate != desc.sample_rate)
                m.error = hrtf_path + ": " + std::to_string(hd.sample_rate) + " Hz, the mixer runs at " + std::to_string(desc.sample_rate) + " - panning only";
            else { md.plugin = audio_plugin_builtin_hrtf(hd); m.binaural_ok = true; }
        }
    }
    if (!m.mixer.init(md)) { m.error = "mixer init failed"; return false; }
    for (uint32_t i = 0; i < m.buses.buses.size(); ++i) m.bus_config(i);
    for (uint32_t i = 0; i < m.buses.buses.size(); ++i)
        if (m.buses.buses[i].reverb) m.reverb_cmd(i, *m.buses.find_reverb(m.buses.buses[i].preset));
    m.mixer.commit();

    audio_stream_desc sd;
    sd.max_streams = desc.max_streams;
    sd.ring_blocks = desc.stream_ring_blocks;
    sd.io = &m.io;
    sd.max_vorbis = desc.max_vorbis_streams;
    sd.vorbis_memory = desc.vorbis_memory;
    sd.sample_rate = desc.sample_rate;
    if (!m.streams.init(sd)) { m.error = "stream system init failed"; return false; }

    audio_policy_desc pd;
    pd.max_voices          = desc.max_voices;
    pd.voice_reserve       = desc.voice_reserve;
    pd.max_instances       = desc.max_instances;
    pd.max_records         = desc.max_virtual;
    pd.threshold_db        = desc.threshold_db;
    pd.hysteresis_db       = desc.hysteresis_db;
    pd.hysteresis_priority = desc.hysteresis_priority;
    pd.max_binaural        = desc.max_binaural;
    audio_param_source ps  = { &m, src_parameter, src_switch, src_position };
    if (!m.policy.init(pd, &m.mixer, &m.streams, &m.buses, &ps)) { m.error = "voice policy init failed"; return false; }

    m.geometry.init(m.mixer.desc().plugin);

    m.running = true;
    m.loader = std::thread([&m] { m.loader_thread(); });

    if (desc.start_device && !m.mixer.start_device()) { m.error = "no playback device"; return false; }
    return true;
}

void audio::shutdown()
{
    if (!m_impl) return;
    impl& m = *m_impl;
    m.mixer.stop_device();
    { std::lock_guard<std::mutex> lock(m.mutex); m.running = false; }
    m.cv.notify_all();
    if (m.loader.joinable()) m.loader.join();
    for (impl::load_job* j : m.queue) delete j;
    for (impl::load_job* j : m.done) delete j;
    m.policy.shutdown();
    m.streams.shutdown();
    m.mixer.shutdown();
    delete m_impl;
    m_impl = nullptr;
}

void audio::render(float* out, uint32_t frames)
{
    if (m_impl) m_impl->mixer.render(out, frames);
}

// ---------------------------------------------------------------------------
// banks: the loader thread reads resident sounds; the game thread attaches them in update()
// ---------------------------------------------------------------------------
void audio::impl::loader_thread()
{
    for (;;) {
        load_job* job = nullptr;
        {
            std::unique_lock<std::mutex> lock(mutex);
            cv.wait(lock, [&] { return !running || !queue.empty(); });
            if (!running) return;
            job = queue.front(); queue.pop_front();
        }
        for (auto& f : job->files)
            if (!read_all(io, f.first->path, f.second) && job->error.empty()) job->error = "cannot read " + f.first->path;
        std::lock_guard<std::mutex> lock(mutex);
        done.push_back(job);
    }
}

audio_bank_t audio::register_bank(const char* name)
{
    if (!m_impl || !name) return {};
    impl& m = *m_impl;
    // <root>/<name>/meta.json - written by a person / the editor; <root>/<name>/sounds/ - what audio_bank_tool generates (sounds.json + *.wav)
    std::string folder = std::string(m.desc.root_path ? m.desc.root_path : "") + name;
    std::vector<uint8_t> bank_xml, sounds_xml;
    if (!read_all(m.io, folder + "/meta.json", bank_xml))           { m.error = "cannot read " + folder + "/meta.json"; return {}; }
    if (!read_all(m.io, folder + "/sounds/sounds.json", sounds_xml)) { m.error = "cannot read " + folder + "/sounds/sounds.json (run audio_bank_tool)"; return {}; }

    uint32_t idx = 0;
    while (idx < m.banks.size() && m.banks[idx]->used) ++idx;
    if (idx == m.banks.size()) m.banks.push_back(std::make_unique<impl::bank>());
    impl::bank& b = *m.banks[idx];
    uint32_t gen = b.generation + 1;
    b.data = audio_bank_data();
    b.data.sound_folder = folder + "/sounds";
    if (!audio_parse_bank((const char*)bank_xml.data(), bank_xml.size(), (const char*)sounds_xml.data(), sounds_xml.size(),
                          m.buses, m.desc.sample_rate, m.desc.stream_seconds, b.data, m.error)) return {};
    for (const audio_event_desc& e : b.data.events)
        if (m.events.count(e.id)) { m.error = "bank '" + b.data.name + "': event '" + e.name + "' already exists in another bank (or hash collision)"; return {}; }
    for (const audio_event_desc& e : b.data.events) m.events[e.id] = &e;
    b.generation = gen;
    b.used = true;
    b.folder = name;
    b.retired.clear();
    b.state = impl::bank_registered;
    b.refcount = 0;
    b.unloading = b.remove = b.job_in_flight = false;
    return { idx, gen };
}

bool audio::reload_banks()
{
    if (!m_impl) return false;
    impl& m = *m_impl;
    bool all = true;
    for (std::unique_ptr<impl::bank>& bp : m.banks) {
        impl::bank& b = *bp;
        if (!b.used || b.remove) continue;
        std::string err;
        const std::string folder = std::string(m.desc.root_path ? m.desc.root_path : "") + b.folder;
        std::vector<uint8_t> meta, sounds;
        if (!read_all(m.io, folder + "/meta.json", meta)) err = "cannot read " + folder + "/meta.json";
        else if (!read_all(m.io, folder + "/sounds/sounds.json", sounds)) err = "cannot read " + folder + "/sounds/sounds.json";
        audio_bank_data nd;
        nd.sound_folder = b.data.sound_folder;
        if (err.empty())
            audio_parse_bank((const char*)meta.data(), meta.size(), (const char*)sounds.data(), sounds.size(),
                             m.buses, m.desc.sample_rate, m.desc.stream_seconds, nd, err);
        // the new events play the sounds the bank already has (loaded, streamed by path)
        auto remap = [&](audio_choice& c) {
            for (const audio_sound*& s : c.sounds) {
                const audio_sound* old = s ? b.data.find_sound(s->id) : nullptr;
                if (!old && err.empty()) err = "sound '" + (s ? s->name : std::string("?")) + "' is new (sounds are not reloaded: register the bank again)";
                s = old;
            }
        };
        for (audio_event_desc& e : nd.events) {
            if (!err.empty()) break;
            e.bank = &b.data;
            for (audio_layer_desc& l : e.layers) { remap(l.pool); remap(l.fallback); for (audio_choice& c : l.cases) remap(c); }
            auto it = m.events.find(e.id);
            if (it != m.events.end() && it->second->bank != &b.data) err = "event '" + e.name + "' already exists in another bank";
        }
        if (!err.empty()) { m.error = "reload of bank '" + b.folder + "': " + err; all = false; continue; }

        // swap: instances follow their event to the new description; the old ones stay until the bank goes
        const uint32_t fade = m.frames(0.05f);
        for (const audio_event_desc& e : b.data.events) {
            const audio_event_desc* to = nullptr;
            for (const audio_event_desc& n : nd.events) if (n.id == e.id) to = &n;
            m.policy.rebind(&e, to, fade);
            m.events.erase(e.id);
        }
        b.retired.emplace_back(std::move(b.data.events), std::move(b.data.curves));      // moved vectors keep their elements in place
        b.data.events = std::move(nd.events);
        b.data.curves = std::move(nd.curves);
        b.data.name = nd.name;
        for (const audio_event_desc& e : b.data.events) m.events[e.id] = &e;
    }
    return all;
}

void audio::unregister_bank(audio_bank_t h)
{
    if (!m_impl) return;
    impl::bank* b = m_impl->bank_of(h);
    if (!b) return;
    b->refcount = 0;
    b->remove = true;
    b->unloading = true;
    m_impl->policy.stop_bank(&b->data, m_impl->frames(0.01f));    // removed in update() once nothing of it plays
}

void audio::load_bank(audio_bank_t h)
{
    if (!m_impl) return;
    impl& m = *m_impl;
    impl::bank* b = m_impl->bank_of(h);
    if (!b || b->remove) return;
    if (++b->refcount > 1) return;
    b->unloading = false;
    if (b->state == impl::bank_ready || b->state == impl::bank_loading) return;
    impl::load_job* job = new impl::load_job();
    job->queued_us = wall_us();
    job->b = b;
    for (audio_sound& s : b->data.sounds) if (!s.stream && !s.loaded) job->files.push_back({ &s, {} });
    if (job->files.empty()) { delete job; b->state = impl::bank_ready; return; }
    b->state = impl::bank_loading;
    b->job_in_flight = true;
    { std::lock_guard<std::mutex> lock(m.mutex); m.queue.push_back(job); }
    m.cv.notify_one();
}

void audio::unload_bank(audio_bank_t h)
{
    if (!m_impl) 
        return;

    impl::bank* b = m_impl->bank_of(h);

    if (!b || b->refcount <= 0) 
        return;
    if (--b->refcount > 0) 
        return;

    b->unloading = true;
    m_impl->policy.stop_bank(&b->data, m_impl->frames(0.01f));    // data freed in update() once nothing of it plays
}

audio_bank_state audio::get_bank_state(audio_bank_t h) const
{
    if (!m_impl) 
        return audio_bank_state_error;

    impl::bank* b = m_impl->bank_of(h);

    if (!b) 
        return audio_bank_state_error;

    switch (b->state) {
        case impl::bank_loading: return audio_bank_state_loading;
        case impl::bank_ready:   return audio_bank_state_ready;
        case impl::bank_error:   return audio_bank_state_error;
        default:                 return audio_bank_state_registered;
    }
}

// ---------------------------------------------------------------------------
// update
// ---------------------------------------------------------------------------
void audio::update(float dt)
{
    if (!m_impl) return;
    impl& m = *m_impl;
    const double update_t0 = wall_us();

    // finished loads: attach the sounds (game thread)
    std::vector<impl::load_job*> done;
    { std::lock_guard<std::mutex> lock(m.mutex); done.swap(m.done); }
    for (impl::load_job* j : done) {
        impl::bank* b = j->b;
        b->job_in_flight = false;
        m.bank_load_last = float((wall_us() - j->queued_us) / 1000.0);
        if (m.bank_load_last > m.bank_load_max) m.bank_load_max = m.bank_load_last;
        if (b->refcount > 0 && !b->remove) {
            std::string err = j->error;
            for (auto& f : j->files) if (err.empty() && !audio_sound_attach(*f.first, std::move(f.second), err)) {}
            b->state = err.empty() ? impl::bank_ready : impl::bank_error;
            if (!err.empty()) m.error = "bank '" + b->data.name + "': " + err;
        }
        delete j;
    }

    // deferred unload / unregister: nothing of the bank plays any more -> the mixer no longer reads its data
    for (uint32_t i = 0; i < m.banks.size(); ++i) {
        impl::bank& b = *m.banks[i];
        if (!b.used || !b.unloading || b.job_in_flight || m.policy.count_bank(&b.data)) continue;
        m.free_bank_data(b);
        b.state = impl::bank_registered;
        b.unloading = false;
        if (b.remove) {
            for (const audio_event_desc& e : b.data.events) m.events.erase(e.id);
            b.data = audio_bank_data();
            b.retired.clear();
            b.used = false;
            b.remove = false;
        }
    }

    // bus fades
    for (uint32_t i = 0; i < m.bus.size(); ++i) {
        impl::bus_state& s = m.bus[i];
        if (s.duration <= 0.0f) continue;
        s.t += dt;
        float k = s.t >= s.duration ? 1.0f : s.t / s.duration;
        s.volume = s.from + (s.to - s.from) * k;
        if (k >= 1.0f) s.duration = 0.0f;
    }

    // ducking: which buses sound (a bus counts with everything under it; a paused one does not), then the targets fade
    if (!m.buses.ducks.empty()) {
        const uint32_t nb = uint32_t(m.buses.buses.size());
        m.policy.bus_activity(m.bus_count.data(), nb);
        for (uint32_t i = nb; i-- > 1;) m.bus_count[m.buses.buses[i].parent] += m.bus_count[i];     // parents have smaller indices
        for (uint32_t i = 0; i < nb; ++i) {
            bool paused = false;
            for (uint32_t a = i;; a = m.buses.buses[a].parent) { if (m.bus[a].pause) { paused = true; break; } if (a == 0) break; }
            m.bus_active[i] = m.bus_count[i] > 0 && !paused;
        }
        audio_duck_update(m.buses, m.bus_active.data(), dt, m.duck.data());
    }
    // volume x duck to the mixer when it changed (the mixer ramps it over a period)
    for (uint32_t i = 0; i < m.bus.size(); ++i) {
        const float v = m.bus_out(i);
        if (fabsf(v - m.sent_volume[i]) <= 1e-5f) continue;
        m.sent_volume[i] = v;
        audio_mixer_cmd c = {}; c.type = audio_mixer_cmd_bus_volume; c.target = i; c.bus_volume.volume = v;
        m.mixer.push(c);
    }

    // listener basis from its object
    if (impl::object* l = m.obj(m.listener)) {
        const float ax[3] = { 1, 0, 0 }, ay[3] = { 0, 1, 0 }, az[3] = { 0, 0, m.desc.right_handed ? -1.0f : 1.0f };
        memcpy(m.listener_basis[0], l->pos, 12);
        rotate(l->rot, ax, m.listener_basis[1]);
        rotate(l->rot, ay, m.listener_basis[2]);
        rotate(l->rot, az, m.listener_basis[3]);
    }
    m.policy.set_listener(m.listener_basis[0], m.listener_basis[1], m.listener_basis[2], m.listener_basis[3]);

    // per-instance overrides of finished instances
    for (size_t i = 0; i < m.overrides.size();) {
        const impl::instance_override& o = m.overrides[i];
        if (m.policy.state({ o.index, o.generation }) == audio_instance_none) { m.overrides[i] = m.overrides.back(); m.overrides.pop_back(); }
        else ++i;
    }

    audio_policy_stats before = m.policy.stats();
    m.policy.update();                      // commits the command batch of this frame
    audio_policy_stats after = m.policy.stats();
    m.last_delta = after;
    m.last_delta.steals  = after.steals  - before.steals;
    m.last_delta.culls   = after.culls   - before.culls;
    m.last_delta.returns = after.returns - before.returns;
    m.last_delta.kills   = after.kills   - before.kills;
    m.profile_frame(wall_us() - update_t0);
}

audio_memory_t audio::impl::memory() const
{
    audio_memory_t r = {};
    for (const auto& b : banks) {
        if (!b->used) continue;
        const audio_bank_data& d = b->data;
        for (const audio_sound& snd : d.sounds) { r.sounds += uint32_t(snd.data.size()); r.envelopes += uint32_t(snd.envelope.size()); }
        r.metadata += uint32_t(d.sounds.capacity() * sizeof(audio_sound) + d.events.capacity() * sizeof(audio_event_desc)
                             + d.curves.capacity() * sizeof(audio_curve));
    }
    r.reverbs  = uint32_t(mixer.reverb_bytes());
    r.mixer    = uint32_t(mixer.memory_bytes() - mixer.reverb_bytes());
    r.queues   = uint32_t(mixer.queue_bytes());
    r.streams  = uint32_t(streams.memory_bytes());
    const audio_spatial_plugin& pl = mixer.desc().plugin;
    r.spatial  = pl.memory_bytes ? uint32_t(pl.memory_bytes(pl.user)) : 0;
    r.policy   = uint32_t(policy.memory_bytes());
    r.total    = r.sounds + r.envelopes + r.metadata + r.mixer + r.reverbs + r.queues + r.streams + r.spatial + r.policy;
    r.peak     = peak_memory > r.total ? peak_memory : r.total;
    return r;
}

void audio::impl::profile_frame(double update_us)
{
    // peaks (every frame: the pools change between frames)
    const audio_policy_stats& p = last_delta;
    if (p.voices_used > peak_voices) peak_voices = p.voices_used;
    if (p.instances_used > peak_instances) peak_instances = p.instances_used;
    if (p.records > peak_records) peak_records = p.records;
    const uint32_t open = streams.open_count();
    if (open > peak_streams) peak_streams = open;
    const uint32_t mem = memory().total;
    if (mem > peak_memory) peak_memory = mem;

    // game thread window
    win_update.add(update_us);
    const double* ph = policy.phase_us();
    for (int k = 0; k < 4; ++k) win_policy[k].add(ph[k]);
    const double now = wall_us();
    if (win_start == 0) win_start = now;
    if (now - win_start < 1e6) return;

    // close the window: game thread + IO totals since the previous one
    const double secs = (now - win_start) * 1e-6;
    pub_update = win_update.timing();
    for (int k = 0; k < 4; ++k) pub_policy[k] = win_policy[k].timing();
    win_update = prof_acc();
    for (prof_acc& a : win_policy) a = prof_acc();
    const audio_stream_system::io_totals t = streams.totals();
    const uint64_t blocks = t.blocks - io_prev.blocks, starts = t.starts - io_prev.starts;
    pub_io_read = { blocks ? float((t.read_us_sum - io_prev.read_us_sum) / double(blocks)) : 0.0f, float(t.read_us_max) };
    const uint64_t decoded = t.decoded - io_prev.decoded;
    pub_io_decode = { decoded ? float((t.decode_us_sum - io_prev.decode_us_sum) / double(decoded)) : 0.0f, float(t.decode_us_max) };
    pub_stream_start = { starts ? float((t.start_us_sum - io_prev.start_us_sum) / double(starts) / 1000.0) : 0.0f, float(t.start_us_max / 1000.0) };
    pub_io_bytes = float(double(t.bytes - io_prev.bytes) / secs);
    io_prev = t;
    streams.reset_max();
    win_start = now;
}

// ---------------------------------------------------------------------------
// mixer buses
// ---------------------------------------------------------------------------
void audio::set_bus_volume(audio_bus_id id, float volume, float fade)
{
    if (!m_impl) return;
    impl& m = *m_impl;
    uint32_t i = m.buses.find(id);
    if (i == ~0u) { m.error = "unknown bus"; return; }
    impl::bus_state& s = m.bus[i];
    if (fade <= 0.0f) {
        s.volume = s.to = volume; s.duration = 0.0f;                    // sent with the next update (x ducking)
    } else {
        s.from = s.volume; s.to = volume; s.t = 0.0f; s.duration = fade;
    }
}

void audio::set_bus_mute(audio_bus_id id, bool mute)
{
    if (!m_impl) return;
    uint32_t i = m_impl->buses.find(id);
    if (i == ~0u) { m_impl->error = "unknown bus"; return; }
    m_impl->bus[i].mute = mute;
    m_impl->bus_config(i);
}

float audio::get_bus_duck(audio_bus_id id) const
{
    if (!m_impl) return 1.0f;
    const uint32_t i = m_impl->buses.find(id);
    return i == ~0u ? 1.0f : m_impl->duck[i].gain;
}

void audio::set_binaural(bool on)
{
    if (!m_impl) return;
    if (on && !m_impl->binaural_ok) { m_impl->error = "set_binaural: no HRTF data (hrtf.bin next to init.json) - panning only"; return; }
    m_impl->policy.set_binaural(on);
}

bool audio::binaural_available() const { return m_impl && m_impl->binaural_ok; }

void audio::set_bus_reverb(audio_bus_id id, uint32_t preset)
{
    if (!m_impl) return;
    uint32_t i = m_impl->buses.find(id);
    if (i == ~0u || !m_impl->buses.buses[i].reverb) { m_impl->error = "set_bus_reverb: not a reverb bus"; return; }
    const audio_reverb_preset* p = m_impl->buses.find_reverb(preset);
    if (!p) { m_impl->error = "set_bus_reverb: unknown reverb preset"; return; }
    if (m_impl->buses.buses[i].preset == preset) return;
    m_impl->buses.buses[i].preset = preset;
    m_impl->reverb_cmd(i, *p);
}

void audio::set_bus_pause(audio_bus_id id, bool pause)
{
    if (!m_impl) return;
    uint32_t i = m_impl->buses.find(id);
    if (i == ~0u) { m_impl->error = "unknown bus"; return; }
    m_impl->bus[i].pause = pause;
    m_impl->bus_config(i);
}

// ---------------------------------------------------------------------------
// objects
// ---------------------------------------------------------------------------
audio_object_t audio::create_object(const char* debug_name)
{
    if (!m_impl) return {};
    impl& m = *m_impl;
    uint32_t idx = 0;
    while (idx < m.objects.size() && m.objects[idx].used) ++idx;
    if (idx >= 0xffffu) { m.error = "too many audio objects"; return {}; }
    if (idx == m.objects.size()) m.objects.push_back(impl::object());
    impl::object& o = m.objects[idx];
    uint32_t gen = o.generation + 1;
    o = impl::object();
    o.generation = gen;
    o.used = true;
    if (debug_name) o.name = debug_name;
    return { idx, gen };
}

void audio::destroy_object(audio_object_t h)
{
    if (!m_impl) return;
    impl& m = *m_impl;
    impl::object* o = m.obj(h);
    if (!o) return;
    m.policy.stop_matching(m.key(h), nullptr, m.frames(0.05f), true);   // loops stop with a fade, one-shots finish in place
    o->used = false;
    o->params.clear(); o->switches.clear();
}

void audio::set_object_transform(audio_object_t h, const vec3& position, const quat& rotation)
{
    if (!m_impl) return;
    if (impl::object* o = m_impl->obj(h)) { o->pos[0] = position.x; o->pos[1] = position.y; o->pos[2] = position.z; o->rot = rotation; }
}

void audio::set_object_velocity(audio_object_t h, const vec3& velocity)
{
    if (!m_impl) return;
    if (impl::object* o = m_impl->obj(h)) { o->vel[0] = velocity.x; o->vel[1] = velocity.y; o->vel[2] = velocity.z; }
}

void audio::set_listener(audio_object_t h)
{
    if (m_impl) m_impl->listener = h;
}

// ---------------------------------------------------------------------------
// events
// ---------------------------------------------------------------------------
audio_playing_t audio::post_event(audio_object_t object, audio_event_id event)
{
    if (!m_impl) return {};
    impl& m = *m_impl;
    auto it = m.events.find(event);
    if (it == m.events.end()) { m.error = "post_event: unknown event (bank not registered?)"; return {}; }
    impl::object* o = m.obj(object);
    if (!o) { m.error = "post_event: stale object"; return {}; }
    audio_instance_handle h = m.policy.post(it->second, m.key(object), o->pos);
    return { h.index, h.generation };
}

audio_playing_t audio::post_event(audio_event_id event)
{
    if (!m_impl) return {};
    impl& m = *m_impl;
    auto it = m.events.find(event);
    if (it == m.events.end()) { m.error = "post_event: unknown event (bank not registered?)"; return {}; }
    audio_instance_handle h = m.policy.post(it->second, k_global_object, m.listener_basis[0]);
    return { h.index, h.generation };
}

void audio::stop(audio_playing_t p, float fade)
{
    if (m_impl) m_impl->policy.stop({ p.index, p.generation }, m_impl->frames(fade));
}

void audio::stop_event(audio_object_t object, audio_event_id event, float fade)
{
    if (!m_impl) return;
    auto it = m_impl->events.find(event);
    if (it == m_impl->events.end()) return;
    m_impl->policy.stop_matching(m_impl->key(object), it->second, m_impl->frames(fade), false);
}

void audio::stop_all(audio_object_t object, float fade)
{
    if (m_impl) m_impl->policy.stop_matching(m_impl->key(object), nullptr, m_impl->frames(fade), false);
}

audio_playing_state audio::get_state(audio_playing_t p) const
{
    if (!m_impl) return audio_playing_state_none;
    switch (m_impl->policy.state({ p.index, p.generation })) {
        case audio_instance_pending:
        case audio_instance_starting: return audio_playing_state_pending;
        case audio_instance_playing:
        case audio_instance_fading:   return audio_playing_state_playing;
        case audio_instance_virtual:  return audio_playing_state_virtual;
        default:                      return audio_playing_state_none;
    }
}

// ---------------------------------------------------------------------------
// parameters / switches
// ---------------------------------------------------------------------------
template<typename T> static void set_value(std::vector<std::pair<uint32_t, T>>& v, uint32_t id, T value)
{
    for (auto& p : v) if (p.first == id) { p.second = value; return; }
    v.push_back({ id, value });
}

void audio::set_parameter(audio_object_t object, audio_parameter_id id, float value)
{
    if (!m_impl) return;
    if (impl::object* o = m_impl->obj(object)) set_value(o->params, id, value);
}

void audio::set_switch(audio_object_t object, audio_switch_id id, audio_switch_value value)
{
    if (!m_impl) return;
    if (impl::object* o = m_impl->obj(object)) set_value(o->switches, id, value);
}

void audio::set_parameter(audio_playing_t p, audio_parameter_id id, float value)
{
    if (!m_impl || m_impl->policy.state({ p.index, p.generation }) == audio_instance_none) return;
    for (auto& o : m_impl->overrides) if (o.index == p.index && o.generation == p.generation && o.id == id) { o.value = value; return; }
    m_impl->overrides.push_back({ p.index, p.generation, id, value });
}

void audio::set_global_parameter(audio_parameter_id id, float value)
{
    if (m_impl) set_value(m_impl->global_params, id, value);
}

void audio::set_global_switch(audio_switch_id id, audio_switch_value value)
{
    if (m_impl) set_value(m_impl->global_switches, id, value);
}

// ---------------------------------------------------------------------------
// geometry (forwarded to the spatial plugin; the built-in one has none)
// ---------------------------------------------------------------------------
audio_geometry_t audio::create_geometry(const audio_geometry_desc_t& desc)
{
    if (!m_impl) return {};
    audio_geometry_registry::handle h = m_impl->geometry.create(desc);
    return { h.index, h.generation };
}

void audio::destroy_geometry(audio_geometry_t g)
{
    if (m_impl) m_impl->geometry.destroy({ g.index, g.generation });
}

audio_geometry_instance_t audio::create_geometry_instance(audio_geometry_t g, const vec3& position, const quat& rotation)
{
    if (!m_impl) return {};
    float p[3] = { position.x, position.y, position.z }, r[4] = { rotation.x, rotation.y, rotation.z, rotation.w };
    audio_geometry_registry::handle h = m_impl->geometry.create_instance({ g.index, g.generation }, p, r);
    return { h.index, h.generation };
}

void audio::destroy_geometry_instance(audio_geometry_instance_t i)
{
    if (m_impl) m_impl->geometry.destroy_instance({ i.index, i.generation });
}

void audio::set_geometry_instance_transform(audio_geometry_instance_t i, const vec3& position, const quat& rotation)
{
    if (!m_impl) return;
    float p[3] = { position.x, position.y, position.z }, r[4] = { rotation.x, rotation.y, rotation.z, rotation.w };
    m_impl->geometry.set_transform({ i.index, i.generation }, p, r);
}

void audio::set_geometry_instance_enabled(audio_geometry_instance_t i, bool enabled)
{
    if (m_impl) m_impl->geometry.set_enabled({ i.index, i.generation }, enabled);
}

// ---------------------------------------------------------------------------
// debug
// ---------------------------------------------------------------------------
const char* audio_stage_name(audio_stage s)
{
    static const char* const names[audio_stage_count] = { "commands", "decode", "voices", "filters", "sends", "spatial", "buses", "output" };
    return s < audio_stage_count ? names[s] : "?";
}

void audio::set_profiling(bool on)
{
    if (m_impl) m_impl->mixer.set_profiling(on);
}

void audio::reset_stats()
{
    if (!m_impl) return;
    impl& m = *m_impl;
    m.peak_voices = m.peak_instances = m.peak_records = m.peak_streams = 0;
    m.peak_memory = 0;
    m.bank_load_max = 0;
    m.mixer.reset_profile();
}

audio_profile_t audio::get_profile() const
{
    audio_profile_t r = {};
    if (!m_impl) return r;
    impl& m = *m_impl;
    audio_mixer_profile mp;
    r.budget_us = float(m.desc.period_frames * 1e6 / m.desc.sample_rate);
    if (m.mixer.profile(mp) && mp.periods) {
        const double inv = 1.0 / mp.periods;
        r.window_seconds = float(mp.periods * double(m.desc.period_frames) / m.desc.sample_rate);
        r.stages_on = mp.stages_on;
        r.period = { float(mp.period_sum * inv), float(mp.period_max) };
        r.period_max_all = float(mp.period_max_all);
        for (int s = 0; s < audio_stage_count; ++s) {
            r.stage[s] = { float(mp.stage_sum[s] * inv), float(mp.stage_max[s]) };
            r.stage_max_all[s] = float(mp.stage_max_all[s]);
        }
        r.periods = mp.periods;
        r.over_50 = mp.over_50; r.over_80 = mp.over_80; r.over_100 = mp.over_100;
        r.callback_interval = { mp.callbacks ? float(mp.callback_sum / mp.callbacks) : 0.0f, float(mp.callback_max) };
        r.instances_max = mp.instances_max; r.voices_max = mp.voices_max;
    }
    r.update = m.pub_update;
    for (int k = 0; k < 4; ++k) r.policy[k] = m.pub_policy[k];
    r.io_read = m.pub_io_read;
    r.io_decode = m.pub_io_decode;
    r.io_bytes_per_second = m.pub_io_bytes;
    r.stream_start_ms = m.pub_stream_start;
    r.bank_load_ms_last = m.bank_load_last;
    r.bank_load_ms_max = m.bank_load_max;
    return r;
}

audio_memory_t audio::get_memory() const
{
    return m_impl ? m_impl->memory() : audio_memory_t{};
}

void audio::debug_streams(void (*fn)(void* user, const audio_stream_info_t& s), void* user) const
{
    if (!m_impl || !fn) return;
    const audio_stream_system& ss = m_impl->streams;
    for (uint32_t i = 0; i < ss.max_streams(); ++i) {
        const audio_stream& s = *ss.slot(i);
        const uint32_t st = s.state.load(std::memory_order_acquire);
        if (st == audio_stream_free) continue;
        audio_stream_info_t info = {};
        info.path = s.path;
        info.state = st == audio_stream_requested ? "opening" : st == audio_stream_ready ? "ready" : st == audio_stream_error ? "error" : "closing";
        info.loop = s.loop;
        if (st == audio_stream_ready) {
            const uint32_t prd = s.produced.load(std::memory_order_acquire), cns = s.consumed.load(std::memory_order_acquire);
            info.ring_fill = prd - cns;
            info.ring_size = s.ring_blocks;
            info.ring_fill_min = s.fill_min.load(std::memory_order_relaxed);
            info.bytes_read = s.bytes_read.load(std::memory_order_relaxed);
            if (s.sample_rate && s.samples_per_block) {
                const uint64_t frame = s.start_frame / s.samples_per_block * s.samples_per_block + uint64_t(cns) * s.samples_per_block;
                info.seconds = float(double(s.frames ? frame % s.frames : frame) / s.sample_rate);
            }
        }
        info.underruns = uint32_t(s.underrun_events.load(std::memory_order_relaxed));
        info.error = st == audio_stream_error && s.error ? s.error : "";
        info.vorbis = s.pcm;
        fn(user, info);
    }
}

void audio::debug_instances(void (*fn)(void* user, const audio_instance_info_t& i), void* user) const
{
    if (!m_impl || !fn) return;
    struct ctx { impl* m; void (*fn)(void*, const audio_instance_info_t&); void* user; } c{ m_impl, fn, user };
    m_impl->policy.debug_records([](void* u, const audio_policy_record_info& r) {
        ctx& c = *(ctx*)u;
        static const char* const states[] = { "none", "pending", "starting", "playing", "virtual", "fading" };
        audio_instance_info_t info = {};
        info.handle = { r.index, r.generation };
        info.event = r.desc->name.c_str();
        const audio::impl::object* o = r.object == k_global_object ? nullptr : c.m->obj_by_key(r.object);
        info.object = r.object == k_global_object ? "global" : o ? o->name.c_str() : "";
        info.state = states[r.state];
        info.bus = r.desc->bus < c.m->buses.buses.size() ? c.m->buses.buses[r.desc->bus].name.c_str() : "";
        memcpy(info.position, r.rel, sizeof(info.position));
        info.distance = r.distance;
        info.audibility_db = r.audibility_db;
        info.priority = r.priority;
        info.voices = r.voices;
        info.binaural = r.binaural;
        info.lowpass_hz = r.lowpass_hz > 0 ? r.lowpass_hz : 0.0f;
        info.highpass_hz = r.highpass_hz > 0 ? r.highpass_hz : 0.0f;
        info.send[0] = r.send[0]; info.send[1] = r.send[1];
        c.fn(c.user, info);
    }, &c);
}

audio_stats_t audio::get_stats() const
{
    audio_stats_t s = {};
    if (!m_impl) return s;
    impl& m = *m_impl;
    const audio_policy_stats& p = m.last_delta;
    audio_mixer_stats ms = m.mixer.stats();
    s.voices_audible    = p.voices_audible;
    s.voices_fading     = p.voices_used > p.voices_audible ? p.voices_used - p.voices_audible : 0;
    s.instances_playing = p.playing;
    s.instances_binaural = p.binaural;
    s.events_virtual    = p.virtual_;
    s.events_pending    = p.pending + p.starting;
    s.steals            = uint32_t(p.steals);
    s.culls             = uint32_t(p.culls);
    s.returns           = uint32_t(p.returns);
    s.kills             = uint32_t(p.kills);
    s.post_failed       = uint32_t(p.post_failed);
    s.stream_underruns  = uint32_t(ms.stream_underruns);
    audio_mixer_profile mp;                                                  // the last full second, else since the last call
    if (m.mixer.profile(mp) && mp.periods) s.mixer_load_max = ms.budget_us > 0 ? float(mp.period_max / ms.budget_us) : 0.0f;
    else s.mixer_load_max = ms.budget_us > 0 ? float(ms.period_us_max / ms.budget_us) : 0.0f;
    s.voices            = { p.voices_used, m.peak_voices, m.desc.max_voices + m.desc.voice_reserve };
    s.instances         = { p.instances_used, m.peak_instances, m.desc.max_instances };
    s.records           = { p.records, m.peak_records, m.desc.max_virtual };
    s.streams           = { m.streams.open_count(), m.peak_streams, m.streams.max_streams() };
    s.commands          = { 0, m.mixer.batch_peak(), m.mixer.batch_capacity() };
    s.command_queue_max = s.commands.peak;
    s.commands_dropped  = uint32_t(m.mixer.commands_dropped());
    s.notes_dropped     = uint32_t(ms.notes_dropped);
    s.waits_for_voices  = uint32_t(p.waits_for_voices);
    s.pending_timeouts  = uint32_t(p.pending_timeouts);
    s.late_layers       = uint32_t(p.late_layers);
    size_t mem = 0;
    for (const auto& b : m.banks) if (b->used) for (const audio_sound& snd : b->data.sounds) mem += snd.data.size() + snd.envelope.size();
    s.memory_bytes      = uint32_t(mem);
    return s;
}

const char* audio::last_error() const
{
    return m_impl ? m_impl->error.c_str() : "audio is not initialised";
}
