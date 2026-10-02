#!/usr/bin/env node
/* Node CLI for tests: node docs/js/check_cli.js <file.tif> <footprint.bin>  ->  JSON on stdout. */
'use strict';
const fs = require('fs');
const path = require('path');
const G = require('./gems-tiff.js');
(async () => {
  const [tif, fp] = process.argv.slice(2);
  if (!tif || !fp) { console.error('usage: check_cli.js file.tif footprint.bin'); process.exit(2); }
  const footprint = G.decodeMaskRuns(new Uint8Array(fs.readFileSync(fp)), G.EXPECT.width * G.EXPECT.height);
  const res = await G.checkFile(new Uint8Array(fs.readFileSync(tif)), footprint);
  process.stdout.write(JSON.stringify(res) + '\n');
})().catch((e) => { process.stdout.write(JSON.stringify({ fatal: String(e.message || e) }) + '\n'); process.exit(1); });
