/*
 * gems-tiff.js - dependency-free GeoTIFF reader + DrivenData GEMS pre-flight checks (browser and Node).
 *
 * WHY: the upload form only says "Predicted values must be in range [0, 1]".  This module tells you *which* pixels
 * violate *which* rule, locally, before a submission slot (3 per rolling week) is spent.  Nothing is uploaded.
 *
 * WHAT IT READS: classic little/big-endian TIFF, 1 band float32, strips or tiles, compression none / LZW (5) /
 * deflate (8, 32946), predictor 1 or 3 (GDAL floating-point).  This covers the official template (LZW), files written
 * by this repository (LZW), and the group's earlier deflate/predictor-3 files.
 *
 * WHAT IT CHECKS (mirrors src/gems/submission.py::check_variants; rules quoted from DrivenData page 967):
 *   single band, float32, EPSG:32611, 3292 x 3730, 100 m geotransform, finite values in [0,1] on every footprint pixel,
 *   outside the footprint NaN (official text), nodata tag NaN.
 * The footprint is the set of finite pixels of the official sample_submission.tif (docs/data/footprint.bin).
 *
 * The same file is exercised under Node by tests/test_site_js.py, so "what CI verifies" is "what your browser runs".
 */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else root.GemsTiff = factory();
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  var EXPECT = { width: 3292, height: 3730, epsg: 32611, res: 100, originX: 243350, originY: 4508550, footprintPixels: 5167373 };
  var TYPE_SIZE = { 1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 1, 7: 1, 8: 2, 9: 4, 10: 8, 11: 4, 12: 8, 16: 8 };

  // ----------------------------------------------------------------------------------------- TIFF parsing
  function parseTiff(u8) {
    var dv = new DataView(u8.buffer, u8.byteOffset, u8.byteLength);
    var bom = dv.getUint16(0, false);
    var le;
    if (bom === 0x4949) le = true; else if (bom === 0x4d4d) le = false; else throw new Error('Not a TIFF file (bad byte-order mark).');
    var magic = dv.getUint16(2, le);
    if (magic === 43) throw new Error('BigTIFF is not supported by this checker (DrivenData expects a normal GeoTIFF).');
    if (magic !== 42) throw new Error('Not a TIFF file (bad magic number).');
    var ifd = dv.getUint32(4, le);
    var n = dv.getUint16(ifd, le);
    var tags = {};
    for (var i = 0; i < n; i++) {
      var e = ifd + 2 + i * 12;
      var tag = dv.getUint16(e, le), type = dv.getUint16(e + 2, le), count = dv.getUint32(e + 4, le);
      var size = (TYPE_SIZE[type] || 1) * count;
      tags[tag] = { type: type, count: count, offset: size <= 4 ? e + 8 : dv.getUint32(e + 8, le) };
    }
    var nextIfd = dv.getUint32(ifd + 2 + n * 12, le);
    function vals(t) {
      var out = [], k;
      if (!t) return out;
      if (t.count > 5e6) throw new Error('TIFF tag with an implausible entry count (' + t.count + '); the file looks corrupt.');
      for (k = 0; k < t.count; k++) {
        var o = t.offset + k * TYPE_SIZE[t.type];
        if (t.type === 3) out.push(dv.getUint16(o, le));
        else if (t.type === 4) out.push(dv.getUint32(o, le));
        else if (t.type === 12) out.push(dv.getFloat64(o, le));
        else if (t.type === 11) out.push(dv.getFloat32(o, le));
        else if (t.type === 1 || t.type === 7) out.push(dv.getUint8(o));
        else if (t.type === 16) out.push(Number(dv.getBigUint64(o, le)));
        else out.push(dv.getUint8(o));
      }
      return out;
    }
    function ascii(t) {
      if (!t) return null;
      var s = '';
      for (var k = 0; k < t.count; k++) { var c = dv.getUint8(t.offset + k); if (!c) break; s += String.fromCharCode(c); }
      return s;
    }
    function one(tag, dflt) { var v = vals(tags[tag]); return v.length ? v[0] : dflt; }
    var info = {
      littleEndian: le, extraIfds: nextIfd !== 0,
      width: one(256), height: one(257), bitsPerSample: vals(tags[258]), compression: one(259, 1),
      samplesPerPixel: one(277, 1), rowsPerStrip: one(278, 0xffffffff), planar: one(284, 1),
      predictor: one(317, 1), sampleFormat: vals(tags[339]),
      tileWidth: one(322, 0), tileHeight: one(323, 0),
      stripOffsets: vals(tags[273]), stripByteCounts: vals(tags[279]),
      tileOffsets: vals(tags[324]), tileByteCounts: vals(tags[325]),
      pixelScale: vals(tags[33550]), tiePoint: vals(tags[33922]), geoKeys: vals(tags[34735]),
      nodata: ascii(tags[42113]), gdalMetadata: ascii(tags[42112]),
    };
    info.tiled = !!(info.tileWidth && tags[324]);
    info.epsg = 0;
    for (var g = 4; g + 3 < info.geoKeys.length; g += 4) if (info.geoKeys[g] === 3072) info.epsg = info.geoKeys[g + 3];
    return info;
  }

  // ----------------------------------------------------------------------------------------- decompression
  function lzwDecode(input, expected) {
    var out = new Uint8Array(expected), op = 0;
    var prefix = new Int32Array(4096), suffix = new Uint8Array(4096), first = new Uint8Array(4096), len = new Uint16Array(4096);
    for (var i = 0; i < 256; i++) { prefix[i] = -1; suffix[i] = i; first[i] = i; len[i] = 1; }
    var next = 258, size = 9, buf = 0, bits = 0, ip = 0, prev = -1;
    function read() {
      while (bits < size) { if (ip >= input.length) return 257; buf = ((buf << 8) | input[ip++]) >>> 0; bits += 8; }
      var code = (buf >>> (bits - size)) & ((1 << size) - 1);
      bits -= size; buf = buf & ((1 << bits) - 1);
      return code;
    }
    function emit(code) {
      var L = len[code], p = op + L - 1, c = code;
      while (c >= 0) { if (p >= 0 && p < expected) out[p] = suffix[c]; p--; c = prefix[c]; }
      op += L;
    }
    for (;;) {
      var code = read();
      if (code === 257) break;
      if (code === 256) { next = 258; size = 9; prev = -1; continue; }
      if (prev === -1) { emit(code); prev = code; continue; }
      if (code < next) {
        emit(code);
        prefix[next] = prev; suffix[next] = first[code]; first[next] = first[prev]; len[next] = len[prev] + 1; next++;
      } else {
        prefix[next] = prev; suffix[next] = first[prev]; first[next] = first[prev]; len[next] = len[prev] + 1;
        emit(next); next++;
      }
      prev = code;
      if (next === 511) size = 10; else if (next === 1023) size = 11; else if (next === 2047) size = 12;
      if (op >= expected) break;
    }
    return out;
  }

  async function inflate(u8) {
    if (typeof DecompressionStream !== 'function') throw new Error('This browser has no DecompressionStream (needed for deflate TIFFs). Use a current Chrome/Firefox/Safari/Edge.');
    async function run(fmt) {
      var stream = new Blob([u8]).stream().pipeThrough(new DecompressionStream(fmt));
      return new Uint8Array(await new Response(stream).arrayBuffer());
    }
    try { return await run('deflate'); } catch (e) { return await run('deflate-raw'); }
  }

  function undoFloatPredictor(row, pixels) {          // TIFF predictor 3, 32-bit samples, 1 sample/pixel
    var i, k, b, out = new Uint8Array(row.length);
    for (i = 1; i < row.length; i++) row[i] = (row[i] + row[i - 1]) & 255;
    for (k = 0; k < pixels; k++) for (b = 0; b < 4; b++) out[4 * k + b] = row[(3 - b) * pixels + k];
    return out;                                        // little-endian float32 bytes
  }

  async function decodeChunk(bytes, info, rows, cols) {
    var expected = rows * cols * 4, raw;
    if (info.compression === 1) raw = bytes;
    else if (info.compression === 5) raw = lzwDecode(bytes, expected);
    else if (info.compression === 8 || info.compression === 32946) raw = await inflate(bytes);
    else throw new Error('Unsupported TIFF compression code ' + info.compression + ' (supported: none, LZW, deflate).');
    raw = raw.length >= expected ? raw.subarray(0, expected) : (function () { var t = new Uint8Array(expected); t.set(raw); return t; })();
    if (info.predictor === 3) {
      var out = new Uint8Array(expected);
      for (var r = 0; r < rows; r++) out.set(undoFloatPredictor(raw.slice(r * cols * 4, (r + 1) * cols * 4), cols), r * cols * 4);
      return out;
    }
    if (info.predictor !== 1) throw new Error('Unsupported TIFF predictor ' + info.predictor + ' for float32 data.');
    return raw.slice(0);
  }

  async function readRaster(u8, info) {
    if (info.bitsPerSample[0] !== 32 || info.sampleFormat[0] !== 3) throw new Error('Not a 32-bit float raster.');
    var W = info.width, H = info.height, out = new Uint8Array(W * H * 4), r, c;
    if (info.tiled) {
      var tw = info.tileWidth, th = info.tileHeight, across = Math.ceil(W / tw), down = Math.ceil(H / th);
      for (var t = 0; t < across * down; t++) {
        var tile = await decodeChunk(u8.subarray(info.tileOffsets[t], info.tileOffsets[t] + info.tileByteCounts[t]), info, th, tw);
        var x0 = (t % across) * tw, y0 = Math.floor(t / across) * th;
        for (r = 0; r < th && y0 + r < H; r++) {
          var w = Math.min(tw, W - x0);
          out.set(tile.subarray(r * tw * 4, r * tw * 4 + w * 4), ((y0 + r) * W + x0) * 4);
        }
      }
    } else {
      var rps = Math.min(info.rowsPerStrip, H);
      for (var s = 0; s < info.stripOffsets.length; s++) {
        var rows = Math.min(rps, H - s * rps);
        if (rows <= 0) break;
        var strip = await decodeChunk(u8.subarray(info.stripOffsets[s], info.stripOffsets[s] + info.stripByteCounts[s]), info, rows, W);
        out.set(strip, s * rps * W * 4);
      }
    }
    if (!info.littleEndian) {                         // normalise to little-endian floats
      for (var i = 0; i < out.length; i += 4) { var a = out[i], b = out[i + 1]; out[i] = out[i + 3]; out[i + 1] = out[i + 2]; out[i + 2] = b; out[i + 3] = a; }
    }
    return new Float32Array(out.buffer, 0, W * H);
  }

  // ----------------------------------------------------------------------------------------- footprint payload
  function decodeMaskRuns(bytes, total) {             // alternating runs, first run = outside (0); uleb128 lengths
    var mask = new Uint8Array(total), pos = 0, n = 0, val = 0;
    while (pos < bytes.length) {
      var v = 0, shift = 0, b;
      do { b = bytes[pos++]; v += (b & 0x7f) * Math.pow(2, shift); shift += 7; } while (b & 0x80);
      if (val) mask.fill(1, n, n + v);
      n += v; val ^= 1;
    }
    if (n !== total) throw new Error('Footprint payload covers ' + n + ' pixels, expected ' + total);
    return mask;
  }

  // ----------------------------------------------------------------------------------------- pre-flight checks
  function preflight(field, info, footprint) {
    var checks = [], W = info.width, H = info.height, total = W * H;
    function add(id, title, pass, detail, hard, fix) { checks.push({ id: id, title: title, pass: !!pass, detail: detail, hard: hard !== false, fix: fix || '' }); }
    var geoOk = info.pixelScale.length >= 2 && info.tiePoint.length >= 6 &&
      Math.abs(info.pixelScale[0] - EXPECT.res) < 1e-9 && Math.abs(info.pixelScale[1] - EXPECT.res) < 1e-9 &&
      Math.abs(info.tiePoint[3] - EXPECT.originX) < 1e-6 && Math.abs(info.tiePoint[4] - EXPECT.originY) < 1e-6;
    add('single_band', 'One band', info.samplesPerPixel === 1 && !info.extraIfds, 'samples/pixel = ' + info.samplesPerPixel, true, 'Export exactly one band.');
    add('float32', '32-bit float (float32)', info.bitsPerSample[0] === 32 && info.sampleFormat[0] === 3, 'bits=' + info.bitsPerSample[0] + ', sampleFormat=' + info.sampleFormat[0], true, 'Write dtype float32, not uint8/float64.');
    add('shape', 'Size 3292 x 3730', W === EXPECT.width && H === EXPECT.height, W + ' x ' + H, true, 'Use the official sample_submission.tif grid.');
    add('crs', 'CRS EPSG:32611', info.epsg === EXPECT.epsg, 'EPSG ' + (info.epsg || 'missing'), true, 'Copy the CRS from the sample file.');
    add('geotransform', '100 m grid, origin (243350, 4508550)', geoOk, 'scale=' + info.pixelScale.slice(0, 2).join(',') + ' origin=' + info.tiePoint.slice(3, 5).join(','), true, 'Copy the transform from the sample file.');
    var stats = null;
    if (field && footprint && field.length === total && footprint.length === total) {
      var inNaN = 0, inInf = 0, inMin = Infinity, inMax = -Infinity, inPos = 0, outNaN = 0, outFinite = 0, outNonZero = 0, inCount = 0, mass = 0, n05 = 0;
      for (var i = 0; i < total; i++) {
        var v = field[i];
        if (footprint[i]) {
          inCount++;
          if (v !== v) inNaN++;
          else if (v === Infinity || v === -Infinity) inInf++;
          else { if (v < inMin) inMin = v; if (v > inMax) inMax = v; if (v > 0) inPos++; if (v > 0.5) n05++; mass += v; }
        } else if (v !== v) outNaN++;
        else { outFinite++; if (v !== 0) outNonZero++; }
      }
      stats = { footprintPixels: inCount, inNaN: inNaN, inInf: inInf, inMin: inMin, inMax: inMax, positives: inPos, over05: n05, mass: mass, outNaN: outNaN, outFinite: outFinite, outNonZero: outNonZero };
      add('footprint_nan', 'No NaN/Inf inside the footprint', inNaN === 0 && inInf === 0, inNaN + ' NaN and ' + inInf + ' Inf inside the ' + inCount.toLocaleString() + '-pixel footprint', true,
        'Replace NaN with 0 (and clip) before writing. NaNs usually come from holes in the input features (about 3,061 footprint pixels are invalid in every feature band).');
      add('range', 'Every footprint value within [0, 1]', inMin >= 0 && inMax <= 1, 'min = ' + inMin + ', max = ' + inMax, true,
        'Clip to [0, 1]. Values like 255 or raw logits trigger "Predicted values must be in range [0, 1]".');
      add('outside_nan', 'Outside the footprint is NaN (official text)', outFinite === 0, outNaN.toLocaleString() + ' NaN, ' + outFinite.toLocaleString() + ' finite outside', false,
        'DrivenData: "data outside the bounds is null or nan". An all-finite twin (zeros outside) carries the same prediction; use it only if the form rejects the NaN version.');
      add('nodata_tag', 'nodata tag is NaN', /^nan$/i.test((info.nodata || '').trim()), 'nodata = ' + (info.nodata === null ? 'none' : info.nodata), false, 'The official sample declares nodata=nan.');
    }
    var hardFail = checks.filter(function (c) { return c.hard && !c.pass; }).map(function (c) { return c.id; });
    return { ok: hardFail.length === 0, hardFailures: hardFail, checks: checks, stats: stats };
  }

  async function checkFile(u8, footprint) {
    var info = parseTiff(u8);
    var field = null, readError = null;
    try { field = await readRaster(u8, info); } catch (e) { readError = String(e.message || e); }
    var res = preflight(field, info, footprint);
    if (readError) { res.checks.push({ id: 'read', title: 'Raster could be decoded', pass: false, detail: readError, hard: true, fix: '' }); res.ok = false; res.hardFailures.push('read'); }
    res.info = { width: info.width, height: info.height, compression: info.compression, predictor: info.predictor, tiled: info.tiled, epsg: info.epsg, nodata: info.nodata };
    return res;
  }

  return { EXPECT: EXPECT, parseTiff: parseTiff, readRaster: readRaster, lzwDecode: lzwDecode, decodeMaskRuns: decodeMaskRuns, preflight: preflight, checkFile: checkFile };
});
