/* Small progressive enhancements: copy buttons and the live countdown. The page is fully usable without JS. */
(function () {
  'use strict';

  function selectNode(el) {                              // last resort: select the text so the user can press Ctrl/Cmd+C
    var r = document.createRange();
    r.selectNodeContents(el);
    var s = window.getSelection();
    s.removeAllRanges();
    s.addRange(r);
  }

  function execCopy(text) {
    var t = document.createElement('textarea');
    t.value = text;
    t.setAttribute('readonly', '');
    t.style.position = 'fixed';
    t.style.opacity = '0';
    document.body.appendChild(t);
    t.select();
    var ok = false;
    try { ok = document.execCommand('copy'); } catch (e) { ok = false; }
    t.remove();
    return ok;
  }

  function copyText(text) {                              // resolves true when the text reached the clipboard
    if (navigator.clipboard && window.isSecureContext) {
      return navigator.clipboard.writeText(text).then(function () { return true; }, function () { return execCopy(text); });
    }
    return Promise.resolve(execCopy(text));
  }

  document.querySelectorAll('[data-copy]').forEach(function (btn) {
    btn.addEventListener('click', function () {
      var el = document.getElementById(btn.getAttribute('data-copy'));
      var text = el ? (el.value || el.textContent).trim() : '';
      var old = btn.textContent;
      copyText(text).then(function (ok) {
        if (!ok && el) selectNode(el);
        btn.textContent = ok ? 'Copied \u2714' : 'Selected: press Ctrl/Cmd+C';
        setTimeout(function () { btn.textContent = old; }, 2200);
      });
    });
  });

  var cd = document.getElementById('countdown');
  if (cd) {
    var end = Date.parse(cd.getAttribute('data-end'));
    var days = Math.max(0, Math.ceil((end - Date.now()) / 86400000));
    cd.textContent = days + ' day' + (days === 1 ? '' : 's') + ' left';
  }
})();
