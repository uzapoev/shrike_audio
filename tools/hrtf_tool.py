#!/usr/bin/env python3
# hrtf_tool.py - HRIR set -> hrtf.bin of the audio module (built-in binaural plugin, audio_hrtf.h)
#
# Role:      converts the MIT KEMAR HRTF set (diffuse-field equalized, 44.1 kHz, 128 taps, one half of the azimuths)
#            into the module's file: 48 kHz, minimum phase (the onset delay is taken out and stored as one number per
#            ear: ITD), both halves (the left one mirrored), int16 with one scale, normalized so that the average power
#            over all directions (both ears) is 1 - the same loudness as equal-power panning.
# Contents:  read_kemar (zip or folder), minimum_phase (log magnitude resampled to the target rate, real cepstrum),
#            onset (8x band-limited upsampling, 20 % of the peak), write (format below).
# Not here:  SOFA files (other sets) - later, same output.
# Uses:      numpy.
# Usage:     python hrtf_tool.py <diffuse.zip | folder with elevN/ subfolders> <out hrtf.bin> [--taps 128] [--rate 48000]
#
# Data:      MIT KEMAR, Bill Gardner and Keith Martin, MIT Media Lab, 1994 (https://sound.media.mit.edu/resources/KEMAR.html).
#            "Provided free with no restrictions on use, provided the authors are cited when the data is used in any
#            research or commercial application." Write the credit next to the file (hrtf.txt) and in the game credits.
#
# hrtf.bin, little endian:
#   char[4] "HRTF", u32 version = 1, u32 sample_rate, u32 taps, u32 count, f32 scale (int16 * scale = tap value)
#   count x { f32 azimuth_deg (0 front, 90 right, clockwise), f32 elevation_deg (+ up), f32 delay_l, f32 delay_r (samples) }
#   count x { i16 left[taps], i16 right[taps] }   (forward order, minimum phase)

import io
import os
import re
import struct
import sys
import wave
import zipfile

import numpy as np

NAME = re.compile(r"H(-?\d+)e(\d+)a\.wav$")


def read_kemar(path):
    """-> list of (elev, az, left[128], right[128]) as float, source rate"""
    files = []
    if zipfile.is_zipfile(path):
        z = zipfile.ZipFile(path)
        for n in z.namelist():
            if NAME.search(n):
                files.append((n, z.read(n)))
    else:
        for root, _, names in os.walk(path):
            for n in names:
                if NAME.search(n):
                    with open(os.path.join(root, n), "rb") as f:
                        files.append((n, f.read()))
    out, rate = [], None
    for name, data in files:
        m = NAME.search(name)
        w = wave.open(io.BytesIO(data))
        assert w.getnchannels() == 2 and w.getsampwidth() == 2, name
        rate = w.getframerate()
        d = np.frombuffer(w.readframes(w.getnframes()), "<i2").reshape(-1, 2).astype(np.float64) / 32768.0
        out.append((float(m.group(1)), float(m.group(2)), d[:, 0].copy(), d[:, 1].copy()))
    if not out:
        sys.exit("no KEMAR files (H<elev>e<az>a.wav) in " + path)
    return out, rate


def minimum_phase(h, src_rate, dst_rate, taps, nfft=8192):
    """magnitude of h, resampled to dst_rate (log magnitude interpolated; above the source band: held, then faded),
    -> minimum phase impulse response of `taps` taps at dst_rate"""
    mag = np.abs(np.fft.rfft(h, nfft))
    f_src = np.arange(len(mag)) * src_rate / nfft
    f_dst = np.arange(nfft // 2 + 1) * dst_rate / nfft
    logm = np.log(np.maximum(mag, 1e-6))
    top = 0.95 * src_rate / 2                                   # above: no data
    lm = np.interp(np.minimum(f_dst, top), f_src, logm)
    fade = np.clip((f_dst - top) / (dst_rate / 2 - top), 0, 1)  # -40 dB at Nyquist, smoothly
    lm += np.log(10 ** (-2 * fade ** 2))
    full = np.concatenate([lm, lm[-2:0:-1]])                    # symmetric log magnitude
    c = np.fft.ifft(full).real                                  # real cepstrum
    fold = np.zeros(nfft)
    fold[0] = c[0]
    fold[1:nfft // 2] = 2 * c[1:nfft // 2]
    fold[nfft // 2] = c[nfft // 2]
    hm = np.fft.ifft(np.exp(np.fft.fft(fold))).real[:taps].copy()
    tail = min(16, taps // 4)                                   # fade the cut
    hm[-tail:] *= 0.5 * (1 + np.cos(np.linspace(0, np.pi, tail)))
    return hm


def onset(h, up=8):
    """first crossing of 20 % of the peak on an 8x band-limited upsampled copy, in source samples"""
    n = len(h)
    spec = np.fft.rfft(h, n)
    hu = np.fft.irfft(spec, n * up) * up
    a = np.abs(hu)
    return float(np.argmax(a > 0.2 * a.max())) / up


def main():
    args = sys.argv[1:]
    taps, rate = 128, 48000
    if "--taps" in args:
        i = args.index("--taps"); taps = int(args[i + 1]); del args[i:i + 2]
    if "--rate" in args:
        i = args.index("--rate"); rate = int(args[i + 1]); del args[i:i + 2]
    if len(args) != 2 or taps % 8 or not 16 <= taps <= 256:
        sys.exit(__doc__ or "usage: hrtf_tool.py <diffuse.zip | folder> <out hrtf.bin> [--taps 128 (16..256, x8)] [--rate 48000]")
    src, out_path = args

    data, src_rate = read_kemar(src)
    dirs = []                                                   # (az, el, left, right, delay_l, delay_r)
    for el, az, l, r in data:
        ol, orr = onset(l), onset(r)
        base = min(ol, orr)
        dl, dr = (ol - base) * rate / src_rate, (orr - base) * rate / src_rate
        hl, hr = minimum_phase(l, src_rate, rate, taps), minimum_phase(r, src_rate, rate, taps)
        dirs.append((az, el, hl, hr, dl, dr))
        if 0 < az < 180:                                        # the set has one half: the other is mirrored
            dirs.append((360 - az, el, hr, hl, dr, dl))
    dirs.sort(key=lambda d: (d[1], d[0]))

    power = np.mean([np.sum(d[2] ** 2) + np.sum(d[3] ** 2) for d in dirs])
    norm = 1.0 / np.sqrt(power)
    peak = max(max(np.abs(d[2]).max(), np.abs(d[3]).max()) for d in dirs) * norm
    scale = peak / 32767.0

    with open(out_path, "wb") as f:
        f.write(b"HRTF")
        f.write(struct.pack("<IIIIf", 1, rate, taps, len(dirs), scale))
        for az, el, _, _, dl, dr in dirs:
            f.write(struct.pack("<ffff", az, el, dl, dr))
        for _, _, hl, hr, _, _ in dirs:
            for h in (hl, hr):
                q = np.clip(np.round(h * norm / scale), -32767, 32767).astype("<i2")
                f.write(q.tobytes())

    # report: how much energy the first 32 / 64 taps hold (shorter files are possible), the largest ITD
    e = lambda h, k: np.sum(h[:k] ** 2) / np.sum(h ** 2)
    e32 = np.mean([e(d[2], 32) for d in dirs] + [e(d[3], 32) for d in dirs])
    e64 = np.mean([e(d[2], 64) for d in dirs] + [e(d[3], 64) for d in dirs])
    itd = max(max(d[4], d[5]) for d in dirs)
    print("%s: %d directions, %d taps at %d Hz, %d bytes; energy in 32 taps %.1f %%, in 64 taps %.2f %%; max ITD %.1f samples (%.2f ms)"
          % (out_path, len(dirs), taps, rate, os.path.getsize(out_path), 100 * e32, 100 * e64, itd, 1000 * itd / rate))


if __name__ == "__main__":
    main()
