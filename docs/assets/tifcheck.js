/* Minimal single-band GeoTIFF pre-flight checker -- no dependencies, no CDN.
 *
 * Parses just enough TIFF to validate a GEMS submission in the user's own
 * browser BEFORE they spend one of the 3 uploads per rolling 7 days:
 *   byte order + magic 42, first IFD, and the tags that matter:
 *     256 ImageWidth, 257 ImageLength, 258 BitsPerSample, 277 SamplesPerPixel,
 *     273 StripOffsets, 279 StripByteCounts, 322 TileWidth, 323 TileLength,
 *     324 TileOffsets, 325 TileByteCounts, 259 Compression, 339 SampleFormat,
 *     34264/33550 model transformation, 34735 GeoKeyDirectory, 42113 GDAL_NODATA
 *   plus LZW (5) and Deflate(8)/Zlib decode so the official `compress: lzw`
 *   profile is handled.
 *
 * Then it runs the same hard checks `src/gems22/submission.py` runs server-side:
 *   single band, float32, EPSG:32611, 3292 x 3730, official geotransform,
 *   every in-footprint pixel finite and within [0, 1] -- the check that catches
 *   "Predicted values must be in range [0, 1]".
 */
(function () {
  "use strict";

  const OFFICIAL = {
    width: 3292, height: 3730, bands: 1, bitsPerSample: 32, sampleFormat: 3,
    modelPixelScale: [100.0, 100.0, 0.0],
    modelTiepoint: [0.0, 0.0, 0.0, 243350.0, 4508550.0, 0.0],
    geoKeyProjectedCSType: 32611,
  };

  function lzwDecode(src) {
    // TIFF LZW: 9-bit codes growing to 12, EarlyChange=1 (the TIFF convention).
    const out = [];
    let dict = [], code = 0, bits = 9, prev = -1, bitBuf = 0, bitCnt = 0;
    const CLEAR = 256, EOD = 257;
    function reset() { dict = []; for (let i = 0; i < 256; i++) dict.push([i]);
      dict.push([]); dict.push([]); bits = 9; prev = -1; }
    reset();
    for (let i = 0; i < src.length; i++) {
      bitBuf = (bitBuf << 8) | src[i]; bitCnt += 8;
      while (bitCnt >= bits) {
        bitCnt -= bits;
        const c = (bitBuf >> bitCnt) & ((1 << bits) - 1);
        bitBuf &= (1 << bitCnt) - 1;
        if (c === CLEAR) { reset(); continue; }
        if (c === EOD) { return new Uint8Array(out); }
        let entry;
        if (c < dict.length) {
          entry = dict[c];
        } else if (c === dict.length) {
          if (prev < 0) return new Uint8Array(out);
          entry = dict[prev].concat([dict[prev][0]]);
        } else { return new Uint8Array(out); }
        for (let k = 0; k < entry.length; k++) out.push(entry[k]);
        if (prev >= 0) {
          dict.push(dict[prev].concat([entry[0]]));
          if (dict.length >= (1 << bits) - 1 && bits < 12) bits++;
        }
        prev = c;
      }
    }
    return new Uint8Array(out);
  }

  async function inflateAsync(src) {
    // strip the 2-byte zlib header if present
    const ds = new DecompressionStream("deflate");
    const stream = new Blob([src]).stream().pipeThrough(ds);
    return new Uint8Array(await new Response(stream).arrayBuffer());
  }

  function parseTiff(buf) {
    const u8 = new Uint8Array(buf);
    const dv = new DataView(buf);
    const bo = String.fromCharCode(u8[0], u8[1]);
    if (bo !== "II" && bo !== "MM") throw new Error("not a TIFF (bad byte order)");
    const le = bo === "II";
    const u16 = (o) => dv.getUint16(o, le);
    const u32 = (o) => dv.getUint32(o, le);
    if (u16(2) !== 42) throw new Error("not a TIFF (magic != 42)");
    const ifd0 = u32(4);
    const n = u16(ifd0);
    const tags = {};
    const TYPE_SIZE = { 1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 1, 7: 1, 8: 2, 9: 4, 10: 8, 11: 4, 12: 8 };
    for (let i = 0; i < n; i++) {
      const off = ifd0 + 2 + i * 12;
      const id = u16(off), type = u16(off + 2), count = u32(off + 4);
      const sz = (TYPE_SIZE[type] || 1) * count;
      const valOff = sz <= 4 ? off + 8 : u32(off + 8);
      const read = () => {
        const arr = [];
        for (let k = 0; k < count; k++) {
          const o = valOff + k * (TYPE_SIZE[type] || 1);
          if (type === 3) arr.push(u16(o));
          else if (type === 4) arr.push(u32(o));
          else if (type === 12) arr.push(dv.getFloat64(o, le));
          else if (type === 11) arr.push(dv.getFloat32(o, le));
          else if (type === 5 || type === 10) { arr.push(u32(o)); arr.push(u32(o + 4)); k++; }
          else arr.push(u8[o]);
        }
        return count === 1 ? arr[0] : arr;
      };
      tags[id] = { type, count, size: sz, read };
    }
    return { u8, dv, le, u16, u32, tags };
  }

  async function readPixels(t) {
    const g = (id) => (t.tags[id] ? t.tags[id].read() : undefined);
    const w = g(256), h = g(257);
    const compression = g(259) || 1;
    const spp = g(277) || 1;
    const bps = g(258);
    const fmt = g(339);
    const isTiled = t.tags[322] !== undefined;
    let chunks = [];
    if (isTiled) {
      const tw = g(322), th = g(323);
      const offs = [].concat(g(324)), lens = [].concat(g(325));
      const ntx = Math.ceil(w / tw), nty = Math.ceil(h / th);
      const tileBytes = tw * th * spp * (Array.isArray(bps) ? bps[0] : bps) / 8;
      const canvas = new Float32Array(w * h * spp);
      for (let i = 0; i < offs.length; i++) {
        let raw = t.u8.subarray(offs[i], offs[i] + lens[i]);
        let dec = compression === 5 ? lzwDecode(raw)
          : (compression === 8 || compression === 32946) ? await inflateAsync(raw)
            : raw.slice();
        const ty = Math.floor(i / ntx), tx = i % ntx;
        const x0 = tx * tw, y0 = ty * th;
        const dvb = new DataView(dec.buffer, dec.byteOffset, dec.byteLength);
        for (let yy = 0; yy < th; yy++) {
          const Y = y0 + yy; if (Y >= h) break;
          for (let xx = 0; xx < tw; xx++) {
            const X = x0 + xx; if (X >= w) break;
            const srcIdx = (yy * tw + xx) * spp;
            for (let s = 0; s < spp; s++) {
              const o = (srcIdx + s) * 4;
              if (o + 4 > dec.byteLength) continue;
              canvas[(Y * w + X) * spp + s] = dvb.getFloat32(o, t.le);
            }
          }
        }
      }
      return { width: w, height: h, spp, bps, fmt, compression, data: canvas };
    }
    const offs = [].concat(g(273)), lens = [].concat(g(279));
    let parts = [];
    for (let i = 0; i < offs.length; i++) {
      const raw = t.u8.subarray(offs[i], offs[i] + lens[i]);
      parts.push(compression === 5 ? lzwDecode(raw)
        : (compression === 8 || compression === 32946) ? await inflateAsync(raw)
          : raw.slice());
    }
    let total = 0; parts.forEach((p) => (total += p.length));
    const all = new Uint8Array(total); let k = 0;
    parts.forEach((p) => { all.set(p, k); k += p.length; });
    const dvb = new DataView(all.buffer);
    const data = new Float32Array(w * h * spp);
    for (let i = 0; i < data.length; i++) data[i] = dvb.getFloat32(i * 4, t.le);
    return { width: w, height: h, spp, bps, fmt, compression, data };
  }

  async function check(buf) {
    const res = { hard: [], advisory: [], meta: {}, ok: false };
    const push = (arr, name, pass, detail) =>
      arr.push({ name, pass: !!pass, detail: String(detail) });
    let t;
    try { t = parseTiff(buf); } catch (e) {
      push(res.hard, "parseable TIFF", false, e.message); return res;
    }
    const g = (id) => (t.tags[id] ? t.tags[id].read() : undefined);
    push(res.hard, "parseable TIFF", true, "byte order " + (t.le ? "little" : "big") + " endian, magic 42");
    try {
      const px = await readPixels(t);
      const w = px.width, h = px.height;
      push(res.hard, "single band", px.spp === 1, "SamplesPerPixel=" + px.spp);
      push(res.hard, "dtype float32", px.bps === 32 && px.fmt === 3,
        "BitsPerSample=" + px.bps + " SampleFormat=" + px.fmt + " (3 = IEEE float)");
      push(res.hard, "shape 3730 x 3292", h === OFFICIAL.height && w === OFFICIAL.width,
        h + " x " + w + " vs " + OFFICIAL.height + " x " + OFFICIAL.width);

      const mps = g(33550), mtp = g(34264);
      const gtOk = mps && mtp &&
        Math.abs(mps[0] - OFFICIAL.modelPixelScale[0]) < 1e-6 &&
        Math.abs(mps[1] - OFFICIAL.modelPixelScale[1]) < 1e-6 &&
        Math.abs(mtp[3] - OFFICIAL.modelTiepoint[3]) < 1e-6 &&
        Math.abs(mtp[4] - OFFICIAL.modelTiepoint[4]) < 1e-6;
      push(res.hard, "geotransform matches official", gtOk,
        "ModelPixelScale=" + JSON.stringify(mps) + " ModelTiepoint=" + JSON.stringify(mtp) +
        " (expected 100 m pixels tied at E 243350, N 4508550)");

      let epsg = null;
      const gk = g(34735);
      if (Array.isArray(gk)) {
        for (let i = 4; i + 3 < gk.length + 1 && i < gk.length; i += 4) {
          if (gk[i] === 3072) epsg = gk[i + 3] || (gk[i + 2] ? gk.slice(gk[i + 1] * 1, 0) : null);
        }
        // 3072 = ProjectedCSTypeGeoKey; value may be inline (count 0 -> offset)
        for (let i = 0; i + 3 < gk.length; i += 4) {
          if (gk[i] === 3072 && gk[i + 1] === 0 && gk[i + 2] === 1) epsg = gk[i + 3];
        }
      }
      push(res.hard, "CRS EPSG:32611", epsg === OFFICIAL.geoKeyProjectedCSType,
        "ProjectedCSTypeGeoKey=" + epsg);

      // ---- pixel statistics -------------------------------------------------
      const d = px.data;
      let min = Infinity, max = -Infinity, nan = 0, neg = 0, over = 0, pos = 0;
      for (let i = 0; i < d.length; i++) {
        const v = d[i];
        if (Number.isNaN(v)) { nan++; continue; }
        if (v < min) min = v; if (v > max) max = v;
        if (v < 0) neg++; if (v > 1) over++; if (v > 0) pos++;
      }
      res.meta = { width: w, height: h, total: d.length, nan, min, max,
                   n_negative: neg, n_above_1: over, n_positive: pos,
                   compression: px.compression, epsg };
      push(res.hard, "no value < 0 anywhere", neg === 0, neg + " pixels below 0");
      push(res.hard, "no value > 1 anywhere", over === 0, over + " pixels above 1");
      push(res.hard, "no negative sentinel (-3.4e38)", min > -1e30,
        "min = " + min + (min < -1e30 ? "  <-- this is the float32 nodata sentinel; " +
          "it is what triggers 'Predicted values must be in range [0, 1]'" : ""));
      push(res.advisory, "NaN only outside the footprint",
        nan === 0 || nan === 7111787,
        nan + " NaN pixels (official template has exactly 7,111,787 outside the footprint; " +
        "0 if you downloaded the -allfinite variant)");
      push(res.advisory, "finite values present", pos > 0, pos + " pixels > 0");
      res.ok = res.hard.every((c) => c.pass);
      return res;
    } catch (e) {
      push(res.hard, "pixel data readable", false, e.message);
      return res;
    }
  }

  window.GEMS22TifCheck = { check, OFFICIAL };
})();
