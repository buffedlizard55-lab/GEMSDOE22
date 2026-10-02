/* UI glue for the pre-upload checker (docs/submit.html). All processing happens locally; nothing is uploaded. */
(function () {
  'use strict';
  var mount = document.getElementById('checker');
  if (!mount || !window.GemsTiff) return;
  var drop = mount.querySelector('.drop'), input = mount.querySelector('input[type=file]'), out = mount.querySelector('.out');
  var footprint = null;
  function esc(s) { return String(s).replace(/[&<>]/g, function (c) { return { '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]; }); }
  function loadFootprint() {
    if (footprint) return Promise.resolve(footprint);
    return fetch('data/footprint.bin').then(function (r) { if (!r.ok) throw new Error('footprint payload missing'); return r.arrayBuffer(); })
      .then(function (b) { footprint = GemsTiff.decodeMaskRuns(new Uint8Array(b), GemsTiff.EXPECT.width * GemsTiff.EXPECT.height); return footprint; });
  }
  function render(name, size, res) {
    var h = '<h3>' + esc(name) + ' <span class="muted small">(' + (size / 1048576).toFixed(2) + ' MB)</span></h3>';
    h += '<p class="' + (res.ok ? 'res-ok' : 'res-bad') + '">' + (res.ok ? 'PASS: every hard requirement we can check locally is met.' : 'FAIL: will likely be rejected (' + res.hardFailures.join(', ') + ').') + '</p>';
    h += '<div class="tw"><table><thead><tr><th>Check</th><th>Result</th><th>Detail</th></tr></thead><tbody>';
    res.checks.forEach(function (c) {
      var state = c.pass ? '\u2714 pass' : (c.hard ? '\u2716 FAIL' : '\u25CB advisory');
      h += '<tr><td>' + esc(c.title) + (c.hard ? '' : ' <span class="muted small">(advisory)</span>') + '</td><td class="' + (c.pass ? 'res-ok' : (c.hard ? 'res-bad' : '')) + '">' + state + '</td><td>' + esc(c.detail) + (c.pass || !c.fix ? '' : '<br><span class="muted">Fix: ' + esc(c.fix) + '</span>') + '</td></tr>';
    });
    h += '</tbody></table></div>';
    if (res.stats) h += '<p class="small muted">Footprint pixels: ' + res.stats.footprintPixels.toLocaleString() + ' \u00b7 positive (&gt;0): ' + res.stats.positives.toLocaleString() + ' \u00b7 outside NaN: ' + res.stats.outNaN.toLocaleString() + ' \u00b7 outside finite: ' + res.stats.outFinite.toLocaleString() + '</p>';
    h += '<p class="small muted">This checks the published format rules only. It cannot tell you whether DrivenData\u2019s private checker will accept the file, and says nothing about the score.</p>';
    out.innerHTML = h;
  }
  function handle(file) {
    if (!file) return;
    if (/\.zip$/i.test(file.name)) { out.innerHTML = '<p class="res-bad">This is a ZIP. Unzip it and drop the single .tif inside (DrivenData accepts the ZIP itself, but this checker reads the GeoTIFF).</p>'; return; }
    out.innerHTML = '<p>Reading ' + esc(file.name) + ' \u2026</p>';
    Promise.all([file.arrayBuffer(), loadFootprint()]).then(function (r) {
      return GemsTiff.checkFile(new Uint8Array(r[0]), r[1]).then(function (res) { render(file.name, file.size, res); });
    }).catch(function (e) { out.innerHTML = '<p class="res-bad">Could not read this file as a GeoTIFF: ' + esc(e.message || e) + '</p>'; });
  }
  input.addEventListener('change', function () { handle(input.files[0]); });
  ['dragenter', 'dragover'].forEach(function (ev) { drop.addEventListener(ev, function (e) { e.preventDefault(); drop.classList.add('over'); }); });
  ['dragleave', 'drop'].forEach(function (ev) { drop.addEventListener(ev, function (e) { e.preventDefault(); drop.classList.remove('over'); }); });
  drop.addEventListener('drop', function (e) { handle(e.dataTransfer.files[0]); });
})();
