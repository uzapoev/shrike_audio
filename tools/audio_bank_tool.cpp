// audio_bank_tool.cpp - prepares the sounds of a bank
//
// Role:      source sounds -> what the runtime loads: IMA ADPCM WAV (mono, 48 kHz, 1024-byte blocks) or a ready Ogg
//            Vorbis file (copied as is) + sounds.xml.
// Contents:  decode (miniaudio: wav / flac / mp3, any rate / channel count -> mono f32 at 48 kHz), int16 with clipping,
//            encode (audio_adpcm::encode_block), envelope (RMS per block on what will play, smoothed over 3 blocks,
//            1 byte = -dB * 2.55), sounds.xml (format, length, stream by length, hex envelope).
//            .ogg: checked (Ogg Vorbis, 48 kHz, 1 or 2 channels, fits the runtime decoder memory), decoded the way the
//            runtime does (audio_vorbis: stereo mixed down) for the envelope (blocks of 1024 frames) and the peak,
//            copied to the bank; always streamed. The tool does not encode Vorbis: an authoring tool will prepare files.
// Thread:    command line tool, single thread.
// Not here:  <bank dir>/meta.xml - written by a person; this tool never touches it.
// Usage:     audio_bank_tool <source dir> <bank dir> [--stream-seconds 10]
//            every *.wav / *.flac / *.mp3 in <source dir> becomes <bank dir>/sounds/<name>.wav, every *.ogg is copied to
//            <bank dir>/sounds/<name>.ogg; <bank dir>/sounds/sounds.xml lists them all. A .wav / .ogg in sounds/ without a
//            source is reported (not deleted).

#define MA_NO_DEVICE_IO
#define MA_NO_ENGINE
#define MA_NO_NODE_GRAPH
#define MA_NO_RESOURCE_MANAGER
#define MA_NO_GENERATION
#define MINIAUDIO_IMPLEMENTATION
#include "miniaudio/miniaudio.h"

#include "audio_adpcm.h"
#include "audio_vorbis.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <string>
#include <vector>
#include <algorithm>
#include <filesystem>

namespace fs = std::filesystem;

static const uint32_t k_rate = 48000, k_block_align = 1024, k_spb = (k_block_align - 4) * 2 + 1;

static void put16(std::vector<uint8_t>& v, uint32_t x) { v.push_back(uint8_t(x)); v.push_back(uint8_t(x >> 8)); }
static void put32(std::vector<uint8_t>& v, uint32_t x) { put16(v, x & 0xffff); put16(v, x >> 16); }

struct converted {
    std::string     name;
    bool            vorbis = false;
    uint64_t        frames = 0;
    float           peak = 0;
    uint64_t        clipped = 0;
    std::string     envelope;
};

static bool decode(const fs::path& path, std::vector<float>& out, std::string& error)
{
    ma_decoder_config dc = ma_decoder_config_init(ma_format_f32, 1, k_rate);     // downmix + resample by miniaudio
    ma_decoder dec;
    if (ma_decoder_init_file(path.string().c_str(), &dc, &dec) != MA_SUCCESS) { error = "cannot decode"; return false; }
    out.clear();
    float buf[4096];
    for (;;) {
        ma_uint64 got = 0;
        ma_result r = ma_decoder_read_pcm_frames(&dec, buf, 4096, &got);
        out.insert(out.end(), buf, buf + got);
        if (got < 4096 || r != MA_SUCCESS) break;
    }
    ma_decoder_uninit(&dec);
    if (out.empty()) { error = "no samples"; return false; }
    return true;
}

static std::vector<uint8_t> encode_wav(const std::vector<int16_t>& pcm)
{
    uint32_t blocks = uint32_t((pcm.size() + k_spb - 1) / k_spb);
    std::vector<uint8_t> data(size_t(blocks) * k_block_align);
    int index = 0;
    for (uint32_t b = 0; b < blocks; ++b) {
        size_t base = size_t(b) * k_spb;
        uint32_t count = uint32_t(std::min<size_t>(k_spb, pcm.size() - base));
        audio_adpcm::encode_block(pcm.data() + base, count, k_spb, index, &data[size_t(b) * k_block_align]);
    }
    std::vector<uint8_t> w;
    w.insert(w.end(), { 'R', 'I', 'F', 'F' }); put32(w, uint32_t(4 + 8 + 20 + 8 + 4 + 8 + data.size()));
    w.insert(w.end(), { 'W', 'A', 'V', 'E', 'f', 'm', 't', ' ' }); put32(w, 20);
    put16(w, 0x11); put16(w, 1); put32(w, k_rate); put32(w, k_rate * k_block_align / k_spb);
    put16(w, k_block_align); put16(w, 4); put16(w, 2); put16(w, k_spb);
    w.insert(w.end(), { 'f', 'a', 'c', 't' }); put32(w, 4); put32(w, uint32_t(pcm.size()));
    w.insert(w.end(), { 'd', 'a', 't', 'a' }); put32(w, uint32_t(data.size()));
    w.insert(w.end(), data.begin(), data.end());
    return w;
}

// RMS per block of what will actually play (the decoded ADPCM), smoothed over 3 blocks, as hex
static std::string envelope_hex(const std::vector<uint8_t>& wav)
{
    audio_adpcm::clip c;
    if (!audio_adpcm::parse_wav(wav.data(), wav.size(), c)) return "";
    std::vector<float> db(c.block_count);
    std::vector<int16_t> tmp(c.samples_per_block);
    for (uint32_t b = 0; b < c.block_count; ++b) {
        audio_adpcm::decode_block(c.blocks + size_t(b) * c.block_align, c.samples_per_block, tmp.data());
        uint64_t n = std::min<uint64_t>(c.samples_per_block, c.frames - uint64_t(b) * c.samples_per_block);
        double sum = 0; for (uint64_t i = 0; i < n; ++i) { double s = tmp[i] / 32768.0; sum += s * s; }
        double rms = n ? sqrt(sum / double(n)) : 0;
        db[b] = rms > 1e-5 ? float(20 * log10(rms)) : -100.0f;
    }
    std::string hex; char h[3];
    for (uint32_t b = 0; b < c.block_count; ++b) {
        float s = db[b], k = 1;
        if (b > 0) { s += db[b - 1]; ++k; }
        if (b + 1 < c.block_count) { s += db[b + 1]; ++k; }
        float v = s / k;
        snprintf(h, 3, "%02x", v <= -100.0f ? 255 : v >= 0 ? 0 : int(lrintf(-v * 2.55f)));
        hex += h;
    }
    return hex;
}

// envelope of mono int16 in blocks of `spb` frames, smoothed over 3 blocks, as hex (the same law as envelope_hex)
static std::string envelope_hex_pcm(const std::vector<int16_t>& pcm, uint32_t spb)
{
    const uint32_t blocks = uint32_t((pcm.size() + spb - 1) / spb);
    std::vector<float> db(blocks);
    for (uint32_t b = 0; b < blocks; ++b) {
        size_t from = size_t(b) * spb, to = (std::min)(pcm.size(), from + spb);
        double sum = 0; for (size_t i = from; i < to; ++i) { double s = pcm[i] / 32768.0; sum += s * s; }
        double rms = to > from ? sqrt(sum / double(to - from)) : 0;
        db[b] = rms > 1e-5 ? float(20 * log10(rms)) : -100.0f;
    }
    std::string hex; char h[3];
    for (uint32_t b = 0; b < blocks; ++b) {
        float s = db[b], k = 1;
        if (b > 0) { s += db[b - 1]; ++k; }
        if (b + 1 < blocks) { s += db[b + 1]; ++k; }
        float v = s / k;
        snprintf(h, 3, "%02x", v <= -100.0f ? 255 : v >= 0 ? 0 : int(lrintf(-v * 2.55f)));
        hex += h;
    }
    return hex;
}

// a ready Ogg Vorbis file: check it the way the runtime will open it, decode it the way the runtime will play it
static bool take_vorbis(const fs::path& path, const fs::path& out, converted& c, std::string& error)
{
    FILE* f = fopen(path.string().c_str(), "rb");
    if (!f) { error = "cannot read"; return false; }
    std::vector<uint8_t> data;
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    data.resize(n > 0 ? size_t(n) : 0);
    bool ok = n > 0 && fread(data.data(), 1, data.size(), f) == data.size();
    fclose(f);
    if (!ok) { error = "cannot read"; return false; }
    std::vector<uint8_t> mem(audio_vorbis::k_default_memory);
    audio_vorbis_decoder dec;
    const char* err = nullptr;
    if (!dec.open_memory(data.data(), data.size(), mem.data(), mem.size(), &err)) { error = err; return false; }
    if (dec.sample_rate() != k_rate) { error = std::to_string(dec.sample_rate()) + " Hz: Vorbis files must be 48000 Hz (re-encode)"; return false; }
    std::vector<int16_t> pcm(size_t(dec.frames()));
    pcm.resize(dec.read(pcm.data(), uint32_t(pcm.size())));
    if (pcm.empty()) { error = "no samples"; return false; }
    for (int16_t s : pcm) { float a = fabsf(s / 32768.0f); if (a > c.peak) c.peak = a; if (s == 32767 || s == -32768) ++c.clipped; }
    c.vorbis = true;
    c.frames = pcm.size();
    c.envelope = envelope_hex_pcm(pcm, audio_vorbis::k_block_frames);
    std::error_code ec;
    fs::copy_file(path, out, fs::copy_options::overwrite_existing, ec);
    if (ec) { error = "cannot write " + out.string(); return false; }
    if (dec.channels() == 2) printf("  note: %s is stereo: played mono ((L + R) / 2)\n", path.filename().string().c_str());
    return true;
}

int main(int argc, char** argv)
{
    if (argc < 3) {
        printf("usage: audio_bank_tool <source dir> <bank dir> [--stream-seconds 10]\n");
        return 2;
    }
    fs::path src = argv[1], dst = fs::path(argv[2]) / "sounds";
    float stream_seconds = 10.0f;
    for (int i = 3; i + 1 < argc; ++i) if (!strcmp(argv[i], "--stream-seconds")) stream_seconds = float(atof(argv[i + 1]));
    if (!fs::is_directory(src)) { printf("error: %s is not a directory\n", src.string().c_str()); return 1; }
    fs::create_directories(dst);

    std::vector<fs::path> files;
    for (const auto& e : fs::directory_iterator(src)) {
        if (!e.is_regular_file()) continue;
        std::string ext = e.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (ext == ".wav" || ext == ".flac" || ext == ".mp3" || ext == ".ogg") files.push_back(e.path());
    }
    std::sort(files.begin(), files.end());

    std::vector<converted> done;
    int errors = 0;
    for (const fs::path& f : files) {
        converted c;
        c.name = f.stem().string();
        bool twice = false;
        for (const converted& o : done) if (o.name == c.name) twice = true;
        if (twice) { printf("error: %s: two sources with the same name\n", c.name.c_str()); ++errors; continue; }
        std::vector<float> pcm;
        std::string err;
        std::string ext = f.extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (ext == ".ogg") {
            if (!take_vorbis(f, dst / (c.name + ".ogg"), c, err)) { printf("error: %s: %s\n", f.filename().string().c_str(), err.c_str()); ++errors; continue; }
            printf("  %-24s %8.3f s  peak %5.2f  vorbis%s\n", c.name.c_str(), double(c.frames) / k_rate, c.peak,
                   c.clipped ? (" FULL SCALE " + std::to_string(c.clipped) + " samples").c_str() : "");
            done.push_back(c);
            continue;
        }
        if (!decode(f, pcm, err)) { printf("error: %s: %s\n", f.filename().string().c_str(), err.c_str()); ++errors; continue; }
        std::vector<int16_t> s16(pcm.size());
        for (size_t i = 0; i < pcm.size(); ++i) {
            float v = pcm[i];
            if (fabsf(v) > c.peak) c.peak = fabsf(v);
            if (v > 1.0f || v < -1.0f) ++c.clipped;
            v = v < -1.0f ? -1.0f : v > 1.0f ? 1.0f : v;
            s16[i] = int16_t(lrintf(v * 32767.0f));
        }
        std::vector<uint8_t> wav = encode_wav(s16);
        fs::path out = dst / (c.name + ".wav");
        FILE* o = fopen(out.string().c_str(), "wb");
        if (!o) { printf("error: cannot write %s\n", out.string().c_str()); ++errors; continue; }
        fwrite(wav.data(), 1, wav.size(), o);
        fclose(o);
        c.frames = s16.size();
        c.envelope = envelope_hex(wav);
        printf("  %-24s %8.3f s  peak %5.2f%s\n", c.name.c_str(), double(c.frames) / k_rate, c.peak,
               c.clipped ? (" CLIPPED " + std::to_string(c.clipped) + " samples").c_str() : "");
        done.push_back(c);
    }

    std::string xml = "<audio_sounds>   <!-- generated by audio_bank_tool from " + src.filename().string() + "; do not edit -->\n";
    for (const converted& c : done) {
        char line[512];
        bool stream = double(c.frames) > double(stream_seconds) * k_rate;
        if (c.vorbis)
            snprintf(line, sizeof(line), "    <sound name=\"%s\" format=\"vorbis\" file=\"%s.ogg\" frames=\"%llu\" rate=\"%u\" stream=\"true\"\n           envelope=\"",
                     c.name.c_str(), c.name.c_str(), (unsigned long long)c.frames, k_rate);
        else
            snprintf(line, sizeof(line), "    <sound name=\"%s\" file=\"%s.wav\" frames=\"%llu\" rate=\"%u\" block_align=\"%u\" stream=\"%s\"\n           envelope=\"",
                     c.name.c_str(), c.name.c_str(), (unsigned long long)c.frames, k_rate, k_block_align, stream ? "true" : "false");
        xml += line + c.envelope + "\"/>\n";
    }
    xml += "</audio_sounds>\n";
    // generated files left over from sources that are gone
    for (const auto& e : fs::directory_iterator(dst)) {
        if (!e.is_regular_file() || (e.path().extension() != ".wav" && e.path().extension() != ".ogg")) continue;
        bool known = false;
        for (const converted& c : done) if (c.name == e.path().stem().string() && (e.path().extension() == ".ogg") == c.vorbis) known = true;
        if (!known) printf("  note: %s has no source any more (not in sounds.xml; delete it if it is not needed)\n", e.path().filename().string().c_str());
    }
    fs::path sx = dst / "sounds.xml";
    FILE* o = fopen(sx.string().c_str(), "wb");
    if (!o) { printf("error: cannot write %s\n", sx.string().c_str()); return 1; }
    fwrite(xml.data(), 1, xml.size(), o);
    fclose(o);
    printf("%zu sounds -> %s%s\n", done.size(), dst.string().c_str(), errors ? " (with errors)" : "");
    return errors ? 1 : 0;
}
