/* Smart Song Search — frontend logic (no framework). */
(() => {
  "use strict";
  const $ = (s, el = document) => el.querySelector(s);
  const $$ = (s, el = document) => [...el.querySelectorAll(s)];

  const MAX_REC_SEC = 20;
  const TARGET_SR = 16000;
  const STOP = new Set("a an and are as at be but by for i i'm im in is it it's its me my of on or so that the this to was we you your oh yeah ooh".split(" "));

  const state = { mode: "type", queryWords: new Set(), selected: null, asrReady: false, busy: false };

  // ---------------------------------------------------------------- helpers
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const norm = (w) => w.toLowerCase().replace(/[‘’`]/g, "'").replace(/[^a-z0-9']/g, "");
  function markWords(text, words) {
    return String(text).split(/([A-Za-z0-9'’]+)/).map((tok, i) => {
      if (i % 2 === 0) return esc(tok);
      const n = norm(tok);
      return words.has(n) && !STOP.has(n) ? `<mark>${esc(tok)}</mark>` : esc(tok);
    }).join("");
  }
  function setQueryWords(normalized) { state.queryWords = new Set((normalized || "").split(/\s+/).filter(Boolean)); }
  const method = () => $('input[name="method"]:checked').value;
  const topK = () => parseInt($("#topk").value, 10);

  function notice(msg) {
    const n = $("#notice");
    if (!msg) { n.hidden = true; n.textContent = ""; return; }
    n.textContent = msg; n.hidden = false;
  }
  function busy(on, label) {
    state.busy = on;
    const go = $("#go");
    go.disabled = on; go.classList.toggle("busy", on);
    go.textContent = on ? (label || "Searching") : "Find the song";
  }
  // Python server address: ?api=... in the page URL, else config.js, else the page's own address.
  const API_BASE = (new URLSearchParams(location.search).get("api") ?? window.SSS_API_BASE ?? "").replace(/\/+$/, "");
  async function api(path, opts = {}) {
    let r;
    try { r = await fetch(API_BASE + path, opts); }
    catch (_) { throw new Error(`Cannot reach the Python server at ${API_BASE || location.origin}. Start it with: python server.py`); }
    let body = null;
    try { body = await r.json(); } catch (_) { /* non-JSON error */ }
    if (!r.ok) throw new Error((body && (body.detail || body.message)) || `Request failed (${r.status})`);
    return body;
  }

  // ---------------------------------------------------------------- status
  async function pollStatus() {
    try {
      const s = await api("/api/status");
      const ps = $("#pill-search");
      ps.className = "pill ready";
      ps.lastChild.textContent = `${s.search.n_songs.toLocaleString()} songs indexed`;
      $("#stat-songs").textContent = s.search.n_songs.toLocaleString();
      $("#pipe-hybrid").textContent = `SBERT + ${s.search.hybrid_lexical}, alpha ${s.search.hybrid_alpha}`;
      const pa = $("#pill-asr");
      const a = s.asr;
      state.asrReady = !!a.ready;
      if (a.disabled) { pa.className = "pill error"; pa.lastChild.textContent = "Voice search off"; }
      else if (a.ready) { pa.className = "pill ready"; pa.lastChild.textContent = a.finetuned ? "Fine-tuned Whisper ready" : `${a.model} ready`; }
      else if (a.error) { pa.className = "pill error"; pa.lastChild.textContent = "Voice model failed to load"; pa.title = a.error; }
      else { pa.className = "pill loading"; pa.lastChild.textContent = "Voice model loading"; }
      $("#pipe-asr").textContent = a.finetuned ? "fine-tuned on sung lyrics" : (a.model || "speech-to-text");
      updateVoiceControls(a);
      if (!a.ready && !a.error && !a.disabled) setTimeout(pollStatus, 3000);
    } catch (e) {
      $("#pill-search").className = "pill error";
      $("#pill-search").lastChild.textContent = "Server not reachable";
      setTimeout(pollStatus, 5000);
    }
  }
  function updateVoiceControls(a) {
    const btn = $("#rec-btn"), hint = $("#sing-hint");
    if (a.ready) { btn.disabled = false; hint.textContent = "Sing one or two lines. Five to fifteen seconds works best. Your browser will ask to use the microphone."; }
    else if (a.error || a.disabled) { btn.disabled = true; hint.textContent = a.disabled ? "Voice search is turned off on this server." : `The voice model could not load: ${a.error}`; }
    else { btn.disabled = true; hint.textContent = "The voice model is loading. This takes up to a minute the first time."; }
  }

  // ---------------------------------------------------------------- tabs
  function selectTab(mode) {
    state.mode = mode;
    $$(".tabs button").forEach((b) => {
      const on = b.dataset.mode === mode;
      b.setAttribute("aria-selected", on); b.tabIndex = on ? 0 : -1;
      $("#" + b.getAttribute("aria-controls")).hidden = !on;
    });
    $("#go").hidden = mode !== "type";
    if (mode === "sing" && !rec.raf) requestAnimationFrame(drawIdle);
  }
  $$(".tabs button").forEach((b, i, all) => {
    b.addEventListener("click", () => selectTab(b.dataset.mode));
    b.addEventListener("keydown", (e) => {
      if (e.key !== "ArrowRight" && e.key !== "ArrowLeft") return;
      const n = all[(i + (e.key === "ArrowRight" ? 1 : all.length - 1)) % all.length];
      n.focus(); selectTab(n.dataset.mode);
    });
  });

  // ---------------------------------------------------------------- text search
  async function textSearch(text) {
    const q = (text ?? $("#query").value).trim();
    if (!q) { notice("Type a few words of the lyrics first."); $("#query").focus(); return; }
    notice(null); busy(true);
    try {
      const res = await api("/api/search", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: q, method: method(), top_k: topK() }),
      });
      renderResults(res);
    } catch (e) { notice(e.message); }
    finally { busy(false); }
  }
  $("#go").addEventListener("click", () => { $("#heard").hidden = true; textSearch(); });
  $("#query").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); $("#heard").hidden = true; textSearch(); }
  });
  $$('input[name="method"], #topk').forEach((el) => el.addEventListener("change", () => {
    if ($("#results-wrap").hidden) return;
    const again = !$("#heard").hidden ? $("#heard-text").value : $("#query").value;
    if (again.trim()) textSearch(again);
  }));
  $("#heard-again").addEventListener("click", () => textSearch($("#heard-text").value));
  $("#heard-text").addEventListener("keydown", (e) => { if (e.key === "Enter") textSearch($("#heard-text").value); });

  // ---------------------------------------------------------------- results
  function renderResults(res) {
    setQueryWords(res.normalized);
    const list = $("#results"); list.innerHTML = "";
    $("#results-wrap").hidden = false;
    const n = res.results.length;
    $("#results-meta").textContent = n ? `${res.method_label}, ${res.ms} ms` : "";
    if (!n) {
      list.innerHTML = `<li class="result"><div class="result-btn" style="display:block">No song matched. Try more words, or switch to the Hybrid method.</div></li>`;
      showSheet(null); return;
    }
    const max = Math.max(...res.results.map((r) => r.score), 1e-9);
    const tpl = $("#result-tpl");
    res.results.forEach((r) => {
      const li = tpl.content.firstElementChild.cloneNode(true);
      $(".rank", li).textContent = r.rank;
      $(".result-title", li).textContent = r.title;
      $(".result-artist", li).textContent = [r.artist, r.album, r.year].filter(Boolean).join(", ");
      $(".result-snippet", li).innerHTML = markWords(r.snippet, state.queryWords);
      $(".bar i", li).style.width = `${Math.max(4, (100 * r.score) / max)}%`;
      $(".num", li).textContent = r.score.toFixed(3);
      const btn = $(".result-btn", li);
      btn.setAttribute("aria-label", `${r.rank}. ${r.title} by ${r.artist}, score ${r.score.toFixed(3)}`);
      btn.addEventListener("click", () => selectSong(r.song_idx, li));
      list.appendChild(li);
    });
    selectSong(res.results[0].song_idx, list.firstElementChild, false);
    $("#results-wrap").scrollIntoView({ behavior: "smooth", block: "start" });
  }

  async function selectSong(idx, li, focusSheet = true) {
    $$(".result").forEach((x) => x.classList.toggle("active", x === li));
    try {
      const s = await api(`/api/songs/${idx}`);
      showSheet(s);
      if (focusSheet && window.matchMedia("(max-width: 900px)").matches) $("#sheet").scrollIntoView({ behavior: "smooth" });
    } catch (e) { notice(e.message); }
  }

  function showSheet(s) {
    $("#sheet-empty").hidden = !!s; $("#sheet-body").hidden = !s;
    if (!s) return;
    $("#sheet-artist").textContent = s.artist;
    $("#sheet-title").textContent = s.title;
    $("#sheet-meta").textContent = [s.album, s.year].filter(Boolean).join(", ");
    $("#sheet-lyrics").innerHTML = markWords(s.lyrics, state.queryWords);
    $("#sheet-note").textContent = s.lyrics_source === "original" ? "" :
      "Lyrics shown in normalised form from the search index. Start the server with --lyrics-csv to show the original text.";
    const first = $("#sheet-lyrics mark");
    if (first) { const sheet = $("#sheet"); sheet.scrollTop = Math.max(0, first.offsetTop - sheet.offsetTop - 120); }
  }

  // ---------------------------------------------------------------- examples
  async function loadExamples() {
    try {
      const ex = await api("/api/examples?n=3");
      const box = $("#examples"); box.innerHTML = "<span>Try a line from a test song:</span>";
      ex.forEach((e) => {
        const b = document.createElement("button");
        b.className = "chip"; b.textContent = e.text; b.title = `${e.title} by ${e.artist}`;
        b.addEventListener("click", () => { selectTab("type"); $("#query").value = e.text; $("#heard").hidden = true; textSearch(e.text); });
        box.appendChild(b);
      });
    } catch (_) { /* examples are optional */ }
  }

  // ---------------------------------------------------------------- audio: decode, resample, WAV
  async function toWav16k(blob) {
    const buf = await blob.arrayBuffer();
    const AC = window.AudioContext || window.webkitAudioContext;
    const ctx = new AC();
    let decoded;
    try { decoded = await ctx.decodeAudioData(buf.slice(0)); } finally { ctx.close && ctx.close(); }
    const seconds = Math.min(decoded.duration, 30);
    const off = new OfflineAudioContext(1, Math.max(1, Math.ceil(seconds * TARGET_SR)), TARGET_SR);
    const src = off.createBufferSource(); src.buffer = decoded; src.connect(off.destination); src.start(0);
    const rendered = await off.startRendering();
    return { wav: encodeWav(rendered.getChannelData(0), TARGET_SR), seconds };
  }
  function encodeWav(samples, sr) {
    const out = new DataView(new ArrayBuffer(44 + samples.length * 2));
    const w = (o, s) => [...s].forEach((c, i) => out.setUint8(o + i, c.charCodeAt(0)));
    w(0, "RIFF"); out.setUint32(4, 36 + samples.length * 2, true); w(8, "WAVE"); w(12, "fmt ");
    out.setUint32(16, 16, true); out.setUint16(20, 1, true); out.setUint16(22, 1, true);
    out.setUint32(24, sr, true); out.setUint32(28, sr * 2, true); out.setUint16(32, 2, true); out.setUint16(34, 16, true);
    w(36, "data"); out.setUint32(40, samples.length * 2, true);
    for (let i = 0; i < samples.length; i++) {
      const v = Math.max(-1, Math.min(1, samples[i]));
      out.setInt16(44 + i * 2, v < 0 ? v * 0x8000 : v * 0x7fff, true);
    }
    return new Blob([out], { type: "audio/wav" });
  }

  async function voiceSearch(blob, filename) {
    notice(null);
    if (!state.asrReady) { notice("The voice model is still loading. Try again in a few seconds."); return; }
    busy(true, "Listening");
    const sub = $("#drop-sub"), recBtn = $("#rec-btn");
    recBtn.disabled = true; $("#rec-label").textContent = "Transcribing";
    try {
      let payload = blob, name = filename || "upload.bin";
      try { const r = await toWav16k(blob); payload = r.wav; name = "voice.wav"; }
      catch (_) { /* browser can't decode this format: let the server try */ }
      const fd = new FormData();
      fd.append("audio", payload, name); fd.append("method", method()); fd.append("top_k", String(topK()));
      const res = await api("/api/voice-search", { method: "POST", body: fd });
      $("#heard").hidden = false;
      $("#heard-text").value = res.transcript;
      $("#heard-meta").textContent = `${res.audio_seconds} s of audio, transcribed in ${(res.asr_ms / 1000).toFixed(1)} s`;
      if (!res.transcript) notice("No words were recognised. Sing a little louder and closer to the microphone.");
      renderResults(res);
    } catch (e) { notice(e.message); }
    finally { busy(false); recBtn.disabled = !state.asrReady; $("#rec-label").textContent = "Sing again"; if (sub) sub.textContent = "WAV, MP3, M4A, OGG or FLAC. The first 30 seconds are used."; }
  }

  // ---------------------------------------------------------------- recording
  const rec = { stream: null, mr: null, chunks: [], ctx: null, analyser: null, raf: 0, t0: 0, timer: 0 };
  const canvas = $("#scope"), g = canvas.getContext("2d");

  function drawIdle() {
    const w = (canvas.width = canvas.clientWidth * devicePixelRatio), h = (canvas.height = 72 * devicePixelRatio);
    g.clearRect(0, 0, w, h); g.fillStyle = "#DADFEA"; g.fillRect(0, h / 2 - devicePixelRatio, w, 2 * devicePixelRatio);
  }
  function drawLive() {
    const w = (canvas.width = canvas.clientWidth * devicePixelRatio), h = (canvas.height = 72 * devicePixelRatio);
    const data = new Uint8Array(rec.analyser.fftSize); rec.analyser.getByteTimeDomainData(data);
    g.clearRect(0, 0, w, h);
    const bars = Math.floor(w / (6 * devicePixelRatio)), step = Math.floor(data.length / bars);
    g.fillStyle = "#1B2238";
    for (let i = 0; i < bars; i++) {
      let peak = 0;
      for (let j = 0; j < step; j++) peak = Math.max(peak, Math.abs(data[i * step + j] - 128));
      const bh = Math.max(2 * devicePixelRatio, (peak / 128) * h * 0.95);
      g.fillRect(i * 6 * devicePixelRatio, (h - bh) / 2, 3 * devicePixelRatio, bh);
    }
    rec.raf = requestAnimationFrame(drawLive);
  }

  async function startRecording() {
    notice(null);
    if (!navigator.mediaDevices?.getUserMedia) { notice("This browser cannot record audio. Use the Upload tab instead."); return; }
    try {
      rec.stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: false, noiseSuppression: true, autoGainControl: true } });
    } catch (e) {
      notice("Microphone access was blocked. Allow it in the browser's address bar, or use the Upload tab."); return;
    }
    const AC = window.AudioContext || window.webkitAudioContext;
    rec.ctx = new AC(); rec.analyser = rec.ctx.createAnalyser(); rec.analyser.fftSize = 2048;
    rec.ctx.createMediaStreamSource(rec.stream).connect(rec.analyser);
    rec.chunks = [];
    rec.mr = new MediaRecorder(rec.stream);
    rec.mr.ondataavailable = (e) => e.data.size && rec.chunks.push(e.data);
    rec.mr.onstop = onRecordingStopped;
    rec.mr.start();
    rec.t0 = performance.now();
    $("#rec-btn").setAttribute("aria-pressed", "true"); $("#rec-label").textContent = "Stop and search";
    rec.timer = setInterval(() => {
      const s = (performance.now() - rec.t0) / 1000;
      $("#rec-time").textContent = `0:${String(Math.floor(s)).padStart(2, "0")}`;
      if (s >= MAX_REC_SEC) stopRecording();
    }, 200);
    drawLive();
  }
  function stopRecording() {
    if (!rec.mr || rec.mr.state === "inactive") return;
    rec.mr.stop(); clearInterval(rec.timer); cancelAnimationFrame(rec.raf); rec.raf = 0;
    rec.stream.getTracks().forEach((t) => t.stop()); rec.ctx.close();
    $("#rec-btn").setAttribute("aria-pressed", "false"); $("#rec-label").textContent = "Sing again";
    drawIdle();
  }
  function onRecordingStopped() {
    const secs = (performance.now() - rec.t0) / 1000;
    const blob = new Blob(rec.chunks, { type: rec.mr.mimeType || "audio/webm" });
    if (secs < 1.2) { notice("That recording was too short. Hold the button while you sing a full line."); return; }
    voiceSearch(blob, "recording.webm");
  }
  $("#rec-btn").addEventListener("click", () => (rec.mr && rec.mr.state === "recording" ? stopRecording() : startRecording()));

  // ---------------------------------------------------------------- upload
  const drop = $("#drop"), file = $("#file");
  file.addEventListener("change", () => file.files[0] && handleFile(file.files[0]));
  ["dragenter", "dragover"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
  drop.addEventListener("drop", (e) => e.dataTransfer.files[0] && handleFile(e.dataTransfer.files[0]));
  function handleFile(f) {
    if (f.size > 40 * 1024 * 1024) { notice("That file is larger than 40 MB. Trim it to a short clip first."); return; }
    $("#drop-sub").textContent = `Transcribing ${f.name}`;
    voiceSearch(f, f.name);
  }

  // ---------------------------------------------------------------- evaluation section
  async function loadMetrics() {
    try {
      const m = await api("/api/metrics");
      renderRetrieval(m.retrieval);
      renderAsr(m.asr);
    } catch (_) { /* optional */ }
  }
  function renderRetrieval(r) {
    const t = $("#ret-table");
    if (!r.rows.length) { t.outerHTML = `<p class="muted">No test_results.csv found in the search artifacts.</p>`; return; }
    const sets = [...new Set(r.rows.map((x) => x.query_set))];
    const methods = [...new Set(r.rows.map((x) => x.method))];
    const get = (m, s, k) => (r.rows.find((x) => x.method === m && x.query_set === s) || {})[k];
    const best = {};
    sets.forEach((s) => ["Hit@1", "MRR@10"].forEach((k) => { best[s + k] = Math.max(...methods.map((m) => get(m, s, k) ?? -1)); }));
    const label = (s) => s.replace(/_/g, " ").replace(/\s*\(.*\)/, (x) => `<br><span class="muted">${esc(x.trim())}</span>`);
    let h = `<caption class="sr-only">Retrieval results on held-out songs</caption><thead><tr class="group"><th></th>${sets.map((s) => `<th colspan="2">${label(esc(s))}</th>`).join("")}</tr>`;
    h += `<tr><th>Method</th>${sets.map(() => "<th>Hit@1</th><th>MRR@10</th>").join("")}</tr></thead><tbody>`;
    methods.forEach((m) => {
      h += `<tr><td>${esc(m)}</td>` + sets.map((s) => ["Hit@1", "MRR@10"].map((k) => {
        const v = get(m, s, k); if (v == null) return "<td>-</td>";
        const txt = v.toFixed(3);
        return `<td>${v === best[s + k] ? `<span class="best-cell">${txt}</span>` : txt}</td>`;
      }).join("")).join("") + "</tr>";
    });
    t.innerHTML = h + "</tbody>";
  }
  function renderAsr(a) {
    if (!a || !a.test) return;
    const pre = a.test.pretrained, ft = a.test.fine_tuned;
    const pct = (v) => `${(100 * v).toFixed(1)}%`;
    const rel = pre.WER > 0 ? (100 * (pre.WER - ft.WER)) / pre.WER : 0;
    let h = `<div class="asr-big"><div><b>${pct(pre.WER)}</b><span>pretrained WER</span></div>
      <div class="after"><b>${pct(ft.WER)}</b><span>fine-tuned WER</span></div></div>
      <table><caption class="sr-only">Speech recognition on held-out songs</caption><tbody>
      <tr><td>Character error rate</td><td>${pct(pre.CER)}</td><td>${pct(ft.CER)}</td></tr>
      <tr><td>Relative WER change</td><td></td><td>${Math.abs(rel) < 0.05 ? "no change" : `${rel > 0 ? "-" : "+"}${Math.abs(rel).toFixed(1)}%`}</td></tr>
      <tr><td>Base model</td><td colspan="2">${esc(a.base_model)}</td></tr>
      <tr><td>Training data</td><td colspan="2">${esc(a.dataset)}, ${a.segments?.train ?? "?"} segments</td></tr>
      <tr><td>Trained layers</td><td colspan="2">${esc(a.trainable)}</td></tr>
      <tr><td>Learning rate, epochs</td><td colspan="2">${a.learning_rate}, ${a.epochs_run}</td></tr>`;
    const e2e = a.end_to_end_tts || {};
    Object.entries(e2e).forEach(([k, v]) => { h += `<tr><td>${esc(k)}</td><td colspan="2">Hit@1 ${(v["Hit@1"] ?? 0).toFixed(2)}, MRR ${(v["MRR@10"] ?? 0).toFixed(2)}</td></tr>`; });
    $("#asr-block").innerHTML = h + "</tbody></table>";
  }

  // ---------------------------------------------------------------- boot
  drawIdle(); window.addEventListener("resize", () => { if (!rec.raf) drawIdle(); });
  selectTab("type");
  pollStatus(); loadExamples(); loadMetrics();
})();
