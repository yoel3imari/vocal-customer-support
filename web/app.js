/* VoxDesk Enterprise Call Console — mic capture, WebSocket transport, streaming playback & settings modal. */

const $ = (id) => document.getElementById(id);
const els = {
  dot: $('conn-dot'),
  tenant: $('tenant-name'),
  status: $('status-pill'),
  canvas: $('waveform'),
  transcript: $('transcript'),
  hint: $('empty-hint'),
  metrics: $('metrics'),
  latencyBars: $('latency-bars'),
  call: $('call-btn'),
  end: $('end-btn'),
  mute: $('mute-btn'),
  muteLabel: $('mute-label'),
  micIconOn: $('mic-icon-on'),
  micIconOff: $('mic-icon-off'),
  interrupt: $('interrupt-btn'),
  timer: $('call-timer'),
  footnote: $('footnote'),
  callIdBadge: $('call-id-badge'),
  stateDetailText: $('state-detail-text'),
  turnCounterBadge: $('turn-counter-badge'),
  clearTranscript: $('clear-transcript-btn'),
  exportTranscript: $('export-transcript-btn'),
  healthModel: $('health-model'),
  healthRag: $('health-rag'),
  healthStatusText: $('health-status-text'),
  statTotalCalls: $('stat-total-calls'),
  statTotalTurns: $('stat-total-turns'),
  statTalkTime: $('stat-talk-time'),
  statAvgLatencies: $('stat-avg-latencies'),
  recentCallsList: $('recent-calls-list'),
  refreshStatsBtn: $('refresh-stats-btn'),
  themeToggle: $('theme-toggle'),
  settingsBtn: $('settings-btn'),
  settingsModal: $('settings-modal'),
  settingsCloseBtn: $('settings-close-btn'),
  settingsBackdrop: $('settings-backdrop'),
};

const UI = {
  idle: { label: 'Idle', hint: 'Press Start Call to talk to support' },
  listening: { label: 'Listening…', hint: 'Go ahead — I’m listening' },
  thinking: { label: 'Thinking…', hint: 'Looking that up…' },
  speaking: { label: 'Speaking…', hint: 'Agent speaking — speak anytime to interrupt' },
};

const STAGE_META = {
  stt: { label: 'Whisper ASR', targetMs: 700 },
  retrieval: { label: 'RAG Vector Search', targetMs: 80 },
  llm_ttft: { label: 'LLM First Token (TTFT)', targetMs: 900 },
  tts_first: { label: 'Piper First Audio', targetMs: 350 },
  llm_total: { label: 'LLM Stream Total', targetMs: 2000 },
  turn_response: { label: 'Turn Response Ready', targetMs: 2500 },
  turn_total: { label: 'Total Turn Duration', targetMs: 3200 },
  summary: { label: 'Rolling Memory Summary', targetMs: 1200 },
};

const state = {
  ws: null,
  ctx: null,
  stream: null,
  worklet: null,
  source: null,
  silentGain: null,
  playHead: 0,
  sources: new Set(),
  sampleRate: 16000,
  muted: false,
  inCall: false,
  callStart: 0,
  timerId: null,
  callId: null,
  turnsCount: 0,
  micAnalyser: null,
  agentAnalyser: null,
  micBuf: null,
  agentBuf: null,
  bubbles: new Map(),
  metrics: new Map(),
};

/* ---------- Settings Modal & Tabs ---------- */

function openSettingsModal() {
  if (!els.settingsModal) return;
  els.settingsModal.classList.remove('hidden');
  els.settingsModal.classList.add('flex');
  fetchHealth();
  fetchTelemetry();
}

function closeSettingsModal() {
  if (!els.settingsModal) return;
  els.settingsModal.classList.add('hidden');
  els.settingsModal.classList.remove('flex');
}

function switchTab(targetId) {
  document.querySelectorAll('.modal-tab').forEach((btn) => {
    const isMatch = btn.dataset.tab === targetId;
    btn.classList.toggle('active-tab', isMatch);
    btn.setAttribute('aria-selected', isMatch ? 'true' : 'false');
  });
  document.querySelectorAll('.tab-panel').forEach((panel) => {
    panel.classList.toggle('hidden', panel.id !== targetId);
  });
}

/* ---------- Theme management ---------- */

function initTheme() {
  const saved = localStorage.getItem('voxdesk-theme');
  if (saved === 'light') {
    document.documentElement.classList.remove('dark');
  } else {
    document.documentElement.classList.add('dark');
  }
}

function toggleTheme() {
  const isDark = document.documentElement.classList.toggle('dark');
  localStorage.setItem('voxdesk-theme', isDark ? 'dark' : 'light');
  if (!state.inCall) renderIdleCanvas();
}

/* ---------- Audio graph ---------- */

async function initAudio() {
  if (state.ctx) return;
  state.ctx = new (window.AudioContext || window.webkitAudioContext)();
  if (state.ctx.state === 'suspended') await state.ctx.resume();

  state.stream = await navigator.mediaDevices.getUserMedia({
    audio: {
      channelCount: 1,
      echoCancellation: true,
      noiseSuppression: true,
      autoGainControl: true,
    },
    video: false,
  });

  await state.ctx.audioWorklet.addModule('/static/pcm-worklet.js');
  state.worklet = new AudioWorkletNode(state.ctx, 'pcm16-resampler', {
    numberOfInputs: 1,
    numberOfOutputs: 1,
    outputChannelCount: [1],
    channelCount: 1,
    channelCountMode: 'explicit',
  });
  state.worklet.port.onmessage = (event) => sendBinary(event.data);
  state.source = state.ctx.createMediaStreamSource(state.stream);
  state.micAnalyser = state.ctx.createAnalyser();
  state.micAnalyser.fftSize = 512;
  state.micBuf = new Uint8Array(state.micAnalyser.fftSize);
  state.source.connect(state.micAnalyser);
  state.source.connect(state.worklet);
  state.silentGain = state.ctx.createGain();
  state.silentGain.gain.value = 0;
  state.worklet.connect(state.silentGain).connect(state.ctx.destination);
}

function teardownAudio() {
  stopPlayback();
  state.worklet?.port.postMessage({ type: 'mute', muted: true });
  state.source?.disconnect();
  state.worklet?.disconnect();
  state.silentGain?.disconnect();
  state.stream?.getTracks().forEach((t) => t.stop());
  state.worklet = state.source = state.silentGain = state.stream = null;
  state.micAnalyser = state.agentAnalyser = null;
  if (state.ctx) {
    state.ctx.close();
    state.ctx = null;
  }
}

function sendBinary(buffer) {
  if (state.ws?.readyState === WebSocket.OPEN && state.inCall && !state.muted) {
    state.ws.send(buffer);
  }
}

function sendJson(payload) {
  if (state.ws?.readyState === WebSocket.OPEN) state.ws.send(JSON.stringify(payload));
}

/* ---------- Playback ---------- */

function enqueueAudio(arrayBuffer) {
  const ctx = state.ctx;
  if (!ctx) return;
  const pcm = new Int16Array(arrayBuffer);
  const buf = ctx.createBuffer(1, pcm.length, state.sampleRate);
  const chan = buf.getChannelData(0);
  for (let i = 0; i < pcm.length; i++) chan[i] = pcm[i] / 32768;

  const node = ctx.createBufferSource();
  node.buffer = buf;
  if (!state.agentAnalyser) {
    state.agentAnalyser = ctx.createAnalyser();
    state.agentAnalyser.fftSize = 512;
    state.agentBuf = new Uint8Array(state.agentAnalyser.fftSize);
    state.agentAnalyser.connect(ctx.destination);
  }
  node.connect(state.agentAnalyser);

  const now = ctx.currentTime;
  const start = Math.max(state.playHead, now + 0.02);
  node.start(start);
  state.playHead = start + buf.duration;
  state.sources.add(node);
  node.onended = () => state.sources.delete(node);
  setState('speaking');
}

function stopPlayback() {
  for (const node of state.sources) {
    try {
      node.stop();
    } catch {}
  }
  state.sources.clear();
  state.playHead = 0;
}

/* ---------- WebSocket ---------- */

function connect() {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  const ws = new WebSocket(`${proto}://${location.host}/ws`);
  ws.binaryType = 'arraybuffer';
  state.ws = ws;

  ws.onopen = () => {
    els.dot.classList.add('live');
    sendJson({ type: 'start' });
  };
  ws.onclose = () => {
    els.dot.classList.remove('live');
    if (state.inCall) hangup(false);
  };
  ws.onerror = () => showError('Connection error — verify the VoxDesk server is running.');
  ws.onmessage = (event) => {
    if (typeof event.data === 'string') handleControl(JSON.parse(event.data));
    else enqueueAudio(event.data);
  };
}

function handleControl(msg) {
  switch (msg.type) {
    case 'state': {
      if (msg.sampleRate) state.sampleRate = msg.sampleRate;
      if (msg.callId) {
        state.callId = msg.callId;
        if (els.callIdBadge) els.callIdBadge.textContent = msg.callId;
      }
      const name = msg.tenantName || (msg.tenant && titleCase(msg.tenant));
      if (name) {
        els.tenant.textContent = name;
        document.title = `${name} — VoxDesk Enterprise`;
      }
      if (msg.voice && els.footnote) {
        els.footnote.textContent = `voice ${msg.voice} · tts ${msg.sampleRate} Hz · asr 16 kHz`;
      }
      if (msg.state !== 'speaking') setState(msg.state, msg.detail);
      break;
    }
    case 'transcript':
      addBubble(msg);
      break;
    case 'metrics':
      updateMetrics(msg);
      break;
    case 'interrupt':
      stopPlayback();
      setState('listening', 'Caller interrupted — listening');
      break;
    case 'error':
      showError(msg.message);
      break;
    case 'call_end':
      hangup(false);
      break;
    default:
      break;
  }
}

function setState(s, customDetail) {
  const ui = UI[s] || UI.idle;
  els.status.dataset.state = s;
  els.status.textContent = ui.label;
  if (els.stateDetailText) {
    els.stateDetailText.textContent = customDetail || ui.hint;
  }
  if (els.interrupt) {
    els.interrupt.disabled = !state.inCall || (s !== 'speaking' && s !== 'thinking');
  }
}

function formatTimeNow() {
  const d = new Date();
  return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
}

function addBubble(msg) {
  if (msg.role === 'caller' && msg.text.trim() === '') return;
  els.hint?.remove();
  const key = msg.role;
  let bubble = state.bubbles.get(key);
  if (!bubble) {
    bubble = document.createElement('div');
    bubble.className = `bubble ${msg.role}`;
    bubble.innerHTML = `
      <span class="who">
        <span class="role-label"></span>
        <span class="ts font-mono text-[10px] opacity-65 font-normal"></span>
      </span>
      <span class="text block"></span>
    `;
    bubble.querySelector('.ts').textContent = formatTimeNow();
    els.transcript.appendChild(bubble);
    state.bubbles.set(key, bubble);
    state.turnsCount += 1;
    updateTurnBadge();
  }
  bubble.classList.toggle('interrupted', !!msg.interrupted);
  const roleLabel = bubble.querySelector('.role-label');
  if (roleLabel) {
    roleLabel.textContent = msg.role === 'caller' ? 'Caller' : 'Agent';
  } else {
    bubble.querySelector('.who').textContent = msg.role === 'caller' ? 'Caller' : 'Agent';
  }
  bubble.querySelector('.text').textContent = msg.text;
  if (msg.final) state.bubbles.delete(key);
  els.transcript.scrollTop = els.transcript.scrollHeight;
}

function updateTurnBadge() {
  if (els.turnCounterBadge) {
    els.turnCounterBadge.textContent = `${state.turnsCount} ${state.turnsCount === 1 ? 'turn' : 'turns'}`;
  }
}

function updateMetrics(msg) {
  state.metrics.set(msg.stage, msg.ms);
  if (els.metrics) {
    els.metrics.hidden = false;
    els.metrics.innerHTML = [...state.metrics.entries()]
      .map(([stage, ms]) => `<span class="chip"><b>${stage}</b> ${Math.round(ms)} ms</span>`)
      .join('');
  }
  renderLatencyBars();
}

function renderLatencyBars() {
  if (!els.latencyBars) return;
  if (state.metrics.size === 0) {
    els.latencyBars.innerHTML = `
      <div class="text-xs text-slate-400 dark:text-slate-500 text-center py-4">
        Stage metrics will populate on the first voice turn.
      </div>
    `;
    return;
  }

  const rows = [...state.metrics.entries()].map(([stage, ms]) => {
    const meta = STAGE_META[stage] || { label: stage.toUpperCase(), targetMs: 1000 };
    const rounded = Math.round(ms);
    const ratio = Math.min(100, Math.max(6, Math.round((ms / (meta.targetMs * 1.6)) * 100)));
    let barColor = 'bg-emerald-500';
    let textColor = 'text-emerald-600 dark:text-emerald-400';
    if (ms > meta.targetMs * 1.35) {
      barColor = 'bg-amber-500';
      textColor = 'text-amber-600 dark:text-amber-400';
    }
    if (ms > meta.targetMs * 2.1) {
      barColor = 'bg-rose-500';
      textColor = 'text-rose-600 dark:text-rose-400';
    }

    return `
      <div class="space-y-1">
        <div class="flex items-center justify-between text-xs">
          <span class="font-medium text-slate-700 dark:text-slate-300">${meta.label}</span>
          <span class="font-mono font-semibold ${textColor}">${rounded} ms</span>
        </div>
        <div class="w-full h-1.5 rounded-full bg-slate-200 dark:bg-slate-800 overflow-hidden">
          <div class="h-full rounded-full ${barColor} transition-all duration-300" style="width: ${ratio}%"></div>
        </div>
      </div>
    `;
  });

  els.latencyBars.innerHTML = rows.join('');
}

function showError(text) {
  const div = document.createElement('div');
  div.className = 'bubble error';
  div.textContent = text;
  els.transcript.appendChild(div);
  els.transcript.scrollTop = els.transcript.scrollHeight;
}

/* ---------- Call lifecycle ---------- */

async function startCall() {
  try {
    await initAudio();
  } catch (err) {
    teardownAudio();
    showError(`Microphone unavailable: ${err.message}. Use localhost or HTTPS.`);
    return;
  }
  state.inCall = true;
  state.callStart = Date.now();
  els.call.disabled = true;
  els.end.disabled = false;
  els.mute.disabled = false;
  els.timer.textContent = '00:00';
  state.timerId = setInterval(tickTimer, 500);
  connect();
  drawLoop();
}

function hangup(sendEnd = true) {
  state.inCall = false;
  if (sendEnd) sendJson({ type: 'end' });
  clearInterval(state.timerId);
  teardownAudio();
  try {
    state.ws?.close();
  } catch {}
  state.ws = null;
  state.bubbles.clear();
  setState('idle');
  els.call.disabled = false;
  els.end.disabled = true;
  els.mute.disabled = true;
  if (els.interrupt) els.interrupt.disabled = true;
  els.mute.classList.remove('on');
  state.muted = false;
  updateMuteButtonUI();
  renderIdleCanvas();
  setTimeout(fetchTelemetry, 350);
}

function toggleMute() {
  state.muted = !state.muted;
  els.mute.classList.toggle('on', state.muted);
  updateMuteButtonUI();
  state.worklet?.port.postMessage({ type: 'mute', muted: state.muted });
  sendJson({ type: 'mute', muted: state.muted });
}

function updateMuteButtonUI() {
  if (els.muteLabel) {
    els.muteLabel.textContent = state.muted ? 'Muted' : 'Mute';
  }
  if (els.micIconOn && els.micIconOff) {
    els.micIconOn.classList.toggle('hidden', state.muted);
    els.micIconOff.classList.toggle('hidden', !state.muted);
  }
}

function triggerBargeIn() {
  if (!state.inCall) return;
  stopPlayback();
  sendJson({ type: 'interrupt' });
  setState('listening', 'Interrupted — listening');
}

function tickTimer() {
  const s = Math.floor((Date.now() - state.callStart) / 1000);
  els.timer.textContent = `${String(Math.floor(s / 60)).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}`;
}

/* ---------- Transcript Tools ---------- */

function clearTranscriptView() {
  state.bubbles.clear();
  state.turnsCount = 0;
  updateTurnBadge();
  els.transcript.innerHTML = '';
  if (els.hint) {
    els.transcript.appendChild(els.hint);
  }
}

function exportTranscriptFile() {
  const items = els.transcript.querySelectorAll('.bubble');
  if (!items.length) return;
  const lines = [];
  items.forEach((b) => {
    const role = b.querySelector('.role-label')?.textContent || b.querySelector('.who')?.textContent || 'System';
    const ts = b.querySelector('.ts')?.textContent || '';
    const txt = b.querySelector('.text')?.textContent || b.textContent || '';
    lines.push(`[${ts}] ${role}: ${txt.trim()}`);
  });
  const blob = new Blob([lines.join('\n\n')], { type: 'text/plain;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `${state.callId || 'voxdesk-call'}-transcript.txt`;
  a.click();
  URL.revokeObjectURL(url);
}

/* ---------- Waveform + visualizers ---------- */

function syncCanvasSize() {
  const canvas = els.canvas;
  if (!canvas) return { width: 720, height: 96, bars: 56 };
  const rect = canvas.getBoundingClientRect();
  const w = Math.max(260, Math.floor(rect.width || 720));
  const h = Math.max(64, Math.floor(rect.height || 96));
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w;
    canvas.height = h;
  }
  const bars = Math.max(28, Math.min(72, Math.floor(w / 14)));
  return { width: w, height: h, bars };
}

function renderIdleCanvas() {
  const canvas = els.canvas;
  if (!canvas) return;
  const ctx2d = canvas.getContext('2d');
  const { width, height, bars } = syncCanvasSize();
  ctx2d.clearRect(0, 0, width, height);

  const gap = 4;
  const bw = (width - gap * (bars - 1)) / bars;
  ctx2d.fillStyle = 'rgba(148, 163, 184, 0.2)';
  for (let i = 0; i < bars; i++) {
    const x = i * (bw + gap);
    const h = 4;
    ctx2d.beginPath();
    ctx2d.roundRect(x, (height - h) / 2, bw, h, 2);
    ctx2d.fill();
  }
}

function drawLoop() {
  if (!state.inCall) return;
  requestAnimationFrame(drawLoop);
  const canvas = els.canvas;
  const ctx2d = canvas.getContext('2d');
  const { width, height, bars } = syncCanvasSize();
  ctx2d.clearRect(0, 0, width, height);

  const gap = 4;
  const bw = (width - gap * (bars - 1)) / bars;
  drawBars(ctx2d, state.micAnalyser, state.micBuf, bars, bw, gap, height, '#3b82f6', 0.65);
  drawBars(ctx2d, state.agentAnalyser, state.agentBuf, bars, bw, gap, height, '#14b8a6', 0.95);
}

function drawBars(g, analyser, buf, bars, bw, gap, height, color, alpha) {
  if (!analyser || !buf) return;
  analyser.getByteFrequencyData(buf);
  g.fillStyle = color;
  g.globalAlpha = alpha;
  const step = Math.floor(buf.length / bars) || 1;
  for (let i = 0; i < bars; i++) {
    let sum = 0;
    for (let j = 0; j < step; j++) sum += buf[i * step + j] || 0;
    const level = sum / step / 255;
    const h = Math.max(4, level * height * 0.85);
    const x = i * (bw + gap);
    g.beginPath();
    g.roundRect(x, (height - h) / 2, bw, h, 2);
    g.fill();
  }
  g.globalAlpha = 1;
}

/* ---------- Enterprise Backend Telemetry (/healthz & /api/calls) ---------- */

async function fetchHealth() {
  try {
    const res = await fetch('/healthz');
    if (!res.ok) return;
    const data = await res.json();
    if (data.tenant && els.tenant.textContent === 'VoxDesk') {
      els.tenant.textContent = titleCase(data.tenant);
    }
    if (els.healthModel) {
      els.healthModel.textContent = data.model || 'unconfigured';
    }
    if (els.healthRag) {
      els.healthRag.textContent = `${data.rag_chunks ?? 0} chunks`;
    }
    if (els.healthStatusText) {
      if (data.ok && data.llm_configured) {
        els.healthStatusText.textContent = 'System Ready';
        els.healthStatusText.className = 'font-semibold text-emerald-600 dark:text-emerald-400';
      } else if (data.ok && !data.llm_configured) {
        els.healthStatusText.textContent = 'API Key Missing';
        els.healthStatusText.className = 'font-semibold text-amber-600 dark:text-amber-400';
      }
    }
  } catch {
    // Non-fatal
  }
}

async function fetchTelemetry() {
  try {
    const res = await fetch('/api/calls');
    if (!res.ok) return;
    const data = await res.json();
    const stats = data.stats || {};
    if (els.statTotalCalls) els.statTotalCalls.textContent = String(stats.calls ?? 0);
    if (els.statTotalTurns) els.statTotalTurns.textContent = String(stats.turns ?? 0);
    if (els.statTalkTime) {
      const secs = Math.round(stats.talk_seconds ?? 0);
      els.statTalkTime.textContent = secs >= 60 ? `${Math.floor(secs / 60)}m ${secs % 60}s` : `${secs}s`;
    }

    if (els.statAvgLatencies) {
      const avg = stats.avg_ms || {};
      const entries = Object.entries(avg);
      if (entries.length === 0) {
        els.statAvgLatencies.innerHTML = '<span>No historical metrics recorded</span>';
      } else {
        els.statAvgLatencies.innerHTML = entries
          .map(
            ([k, v]) =>
              `<span class="px-2 py-0.5 rounded bg-slate-200/70 dark:bg-slate-800 text-slate-700 dark:text-slate-300"><b>${k}:</b> ${Math.round(v)}ms</span>`
          )
          .join('');
      }
    }

    if (els.recentCallsList) {
      const calls = (data.calls || []).slice(0, 8);
      if (calls.length === 0) {
        els.recentCallsList.innerHTML =
          '<div class="p-4 text-center text-slate-400 dark:text-slate-500">No calls recorded yet.</div>';
      } else {
        els.recentCallsList.innerHTML = calls
          .map((c) => {
            const dur = c.duration_s != null ? `${Math.round(c.duration_s)}s` : 'active';
            const started = c.started_at ? c.started_at.replace('T', ' ').slice(0, 19) : '';
            return `
              <div class="px-4 py-2.5 flex items-center justify-between gap-2 hover:bg-slate-50 dark:hover:bg-slate-800/40 transition-colors">
                <div class="min-w-0">
                  <div class="flex items-center gap-2">
                    <span class="font-mono font-medium text-slate-700 dark:text-slate-200 truncate">${c.id}</span>
                    <span class="text-[10px] px-1.5 py-0.5 rounded bg-slate-100 dark:bg-slate-800 text-slate-500 dark:text-slate-400">${c.tenant}</span>
                  </div>
                  <div class="text-[10px] text-slate-400 dark:text-slate-500 font-mono mt-0.5">${started} UTC</div>
                </div>
                <div class="text-right shrink-0 font-mono">
                  <div class="text-slate-700 dark:text-slate-300 font-medium">${dur}</div>
                  <div class="text-[10px] text-slate-400 dark:text-slate-500">${c.turns ?? 0} turns</div>
                </div>
              </div>
            `;
          })
          .join('');
      }
    }
  } catch {
    // Ignore transient network errors
  }
}

const titleCase = (s) => s.replace(/[-_]/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase());

/* ---------- Event Listeners & Init ---------- */

els.call.addEventListener('click', startCall);
els.end.addEventListener('click', () => hangup(true));
els.mute.addEventListener('click', toggleMute);
els.interrupt?.addEventListener('click', triggerBargeIn);
els.clearTranscript?.addEventListener('click', clearTranscriptView);
els.exportTranscript?.addEventListener('click', exportTranscriptFile);
els.refreshStatsBtn?.addEventListener('click', () => {
  fetchHealth();
  fetchTelemetry();
});
els.themeToggle?.addEventListener('click', toggleTheme);

// Modal listeners
els.settingsBtn?.addEventListener('click', openSettingsModal);
els.settingsCloseBtn?.addEventListener('click', closeSettingsModal);
els.settingsBackdrop?.addEventListener('click', closeSettingsModal);
document.querySelectorAll('.modal-tab').forEach((btn) => {
  btn.addEventListener('click', () => switchTab(btn.dataset.tab));
});

document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape' && els.settingsModal && !els.settingsModal.classList.contains('hidden')) {
    closeSettingsModal();
    return;
  }
  if (e.code === 'Space' && !e.repeat && state.inCall && document.activeElement?.tagName !== 'BUTTON') {
    e.preventDefault();
    toggleMute();
  }
});

window.addEventListener('resize', () => {
  if (!state.inCall) renderIdleCanvas();
});

initTheme();
renderIdleCanvas();
fetchHealth();
fetchTelemetry();
