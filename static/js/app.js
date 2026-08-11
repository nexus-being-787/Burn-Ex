/**
 * Burn-Ex Dashboard — Improved JS
 * Connects to SSE /sse, drives two Chart.js instances, shows auto-detection
 * confidence bar, session timer, EMA smoothing, activity button states.
 */

'use strict';

// ---- Config ----
const MAX_CHART_PTS  = 180;   // 3 min of 1-s history
const MAX_RETRY_WAIT = 8000;  // max SSE reconnect backoff (ms)

// ---- Activity color map ----
const ACT_COLOR = {
  idle:          '#8891b0',
  walking:       '#00b4ff',
  jogging:       '#00e5a0',
  jumping_jacks: '#ff9d00',
  squats:        '#a855f7',
  unlabeled:     '#444',
};

// ---- State ----
let rateChart   = null;
let totalChart  = null;
let sessionStart = Date.now();
let timerHandle  = null;

const rateLabels  = [];
const rateData    = [];
const totalLabels = [];
const totalData   = [];

// ---- Helpers ----
const $  = id => document.getElementById(id);
const pop = el => { el.classList.remove('pop'); void el.offsetWidth; el.classList.add('pop'); };

function setVal(id, text) {
  const el = $(id);
  if (!el) return;
  if (el.textContent !== String(text)) { el.textContent = text; pop(el); }
}

function setBar(id, value, maxVal = 15) {
  const el = $(id);
  if (el) el.style.width = Math.min(100, (value / maxVal) * 100) + '%';
}

// ---- Session timer ----
function startTimer() {
  sessionStart = Date.now();
  clearInterval(timerHandle);
  timerHandle = setInterval(() => {
    const s   = Math.floor((Date.now() - sessionStart) / 1000);
    const mm  = String(Math.floor(s / 60)).padStart(2, '0');
    const ss  = String(s % 60).padStart(2, '0');
    const el  = $('session-timer');
    if (el) el.textContent = `${mm}:${ss}`;
  }, 1000);
}

// ---- Init Charts ----
function makeChart(canvasId, label, borderColor, fillColor, yLabel) {
  const ctx = $(canvasId).getContext('2d');
  return new Chart(ctx, {
    type: 'line',
    data: { labels: [], datasets: [{ label, data: [], borderColor, backgroundColor: fillColor,
      borderWidth: 2, pointRadius: 0, pointHoverRadius: 3, tension: 0.4, fill: true }] },
    options: {
      responsive: true, maintainAspectRatio: false,
      animation: { duration: 180 },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: 'rgba(8,10,20,.92)',
          titleColor: '#6b7799', bodyColor: borderColor,
          borderColor: borderColor + '44', borderWidth: 1,
          callbacks: { label: c => `${c.parsed.y.toFixed(3)} ${yLabel}` }
        }
      },
      scales: {
        x: { display: true, ticks: { color: '#6b7799', maxTicksLimit: 5, font: { size: 9 } },
             grid: { color: 'rgba(255,255,255,0.04)' } },
        y: { display: true, min: 0,
             ticks: { color: '#6b7799', font: { size: 9 } },
             grid: { color: 'rgba(255,255,255,0.04)' } }
      }
    }
  });
}

function initCharts() {
  rateChart  = makeChart('chart-rate',  'kcal/min', '#00b4ff', 'rgba(0,180,255,.07)', 'kcal/min');
  totalChart = makeChart('chart-total', 'kcal',     '#00e5a0', 'rgba(0,229,160,.07)', 'kcal');
}

function pushToChart(chart, labelArr, dataArr, timeStr, value) {
  labelArr.push(timeStr);
  dataArr.push(value);
  if (labelArr.length > MAX_CHART_PTS) { labelArr.shift(); dataArr.shift(); }
  chart.data.labels  = labelArr;
  chart.data.datasets[0].data = dataArr;
  chart.update('none');
}

function updateChartsFromHistory(history) {
  if (!history || history.length < 2) return;
  const slice = history.slice(-MAX_CHART_PTS);
  rateLabels.length  = 0; rateData.length  = 0;
  totalLabels.length = 0; totalData.length = 0;

  let cumulative = 0;
  slice.forEach(([t, v], i) => {
    const dt  = i > 0 ? t - slice[i-1][0] : 1;
    cumulative += v * dt / 60;
    const ts  = `${t.toFixed(0)}s`;
    rateLabels.push(ts);  rateData.push(v);
    totalLabels.push(ts); totalData.push(parseFloat(cumulative.toFixed(3)));
  });

  rateChart.data.labels  = rateLabels;
  rateChart.data.datasets[0].data = rateData;
  rateChart.update('none');

  totalChart.data.labels = totalLabels;
  totalChart.data.datasets[0].data = totalData;
  totalChart.update('none');
}

// ---- Apply SSE state ----
function applyState(d) {
  // Connection badge
  const badge = $('connection-badge');
  if (badge) { badge.textContent = '● LIVE'; badge.className = 'badge badge-live'; }

  // FPS
  const fps = $('fps-display');
  if (fps) fps.textContent = `${d.fps} FPS`;

  // Auto-detected activity
  const actColor  = ACT_COLOR[d.auto_label || d.label] || '#888';
  const autoBadge = $('auto-badge');
  if (autoBadge) {
    autoBadge.textContent    = (d.auto_label || d.label || 'AUTO').toUpperCase().replace('_', ' ');
    autoBadge.style.color       = actColor;
    autoBadge.style.borderColor = actColor + '44';
    autoBadge.style.background  = actColor + '12';
  }

  // Bio row — activity
  const actEl = $('val-activity');
  if (actEl) { actEl.textContent = (d.auto_label || d.label || '—'); actEl.style.color = actColor; }

  // Confidence bar
  const conf = d.confidence ?? 0;
  const fill = $('conf-bar-fill');
  if (fill) { fill.style.width = (conf * 100).toFixed(0) + '%'; fill.style.background = actColor; fill.style.boxShadow = `0 0 8px ${actColor}`; }
  const confPct = $('conf-pct');
  if (confPct) confPct.textContent = (conf * 100).toFixed(0) + '%';

  // AI burn rate
  const predStr = d.pred_kcal.toFixed(3);
  setVal('val-pred', predStr);
  setBar('bar-pred', d.pred_kcal);
  const hudPred = $('hud-pred');
  if (hudPred) hudPred.textContent = predStr + ' kcal/min';

  // MET reference
  setVal('val-met', d.gt_kcal.toFixed(3));
  setBar('bar-met', d.gt_kcal);

  // Session total
  const totalStr = d.total_kcal.toFixed(2);
  setVal('val-total', totalStr);
  const hudTotal = $('hud-total');
  if (hudTotal) hudTotal.textContent = totalStr + ' kcal';

  // Reps
  setVal('val-reps', d.reps);
  const hudReps = $('hud-reps');
  if (hudReps) hudReps.textContent = d.reps;

  // Bio
  const intEl = $('val-intensity');
  if (intEl) intEl.textContent = d.intensity.toFixed(4);
  const lk = $('val-lknee'), rk = $('val-rknee');
  if (lk) lk.textContent = d.left_knee  > 0 ? d.left_knee.toFixed(1)  + '°' : '—°';
  if (rk) rk.textContent = d.right_knee > 0 ? d.right_knee.toFixed(1) + '°' : '—°';

  // Charts
  if (d.cal_history && d.cal_history.length > 0) {
    updateChartsFromHistory(d.cal_history);
  }
}

// ---- SSE connection with exponential backoff ----
let _retryWait = 500;

function connectSSE() {
  const es = new EventSource('/sse');
  _retryWait = 500;

  es.onmessage = e => {
    try { applyState(JSON.parse(e.data)); } catch (_) {}
  };

  es.onerror = () => {
    es.close();
    const badge = $('connection-badge');
    if (badge) { badge.textContent = '● Reconnecting…'; badge.className = 'badge badge-connecting'; }
    setTimeout(connectSSE, _retryWait);
    _retryWait = Math.min(_retryWait * 1.8, MAX_RETRY_WAIT);
  };
}

// ---- Activity buttons ----
function initButtons() {
  document.querySelectorAll('.act-btn[data-label]').forEach(btn => {
    btn.addEventListener('click', async () => {
      const label = btn.dataset.label;
      const res   = await fetch(`/api/label/${label}`, { method: 'POST' }).catch(() => null);
      if (res && res.ok) {
        document.querySelectorAll('.act-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
      }
    });
  });

  const autoBtn = $('btn-auto');
  if (autoBtn) {
    autoBtn.addEventListener('click', async () => {
      await fetch('/api/label/unlabeled', { method: 'POST' }).catch(() => null);
      document.querySelectorAll('.act-btn').forEach(b => b.classList.remove('active'));
      autoBtn.classList.add('active');
    });
    // Start with auto selected
    autoBtn.classList.add('active');
  }

  const resetBtn = $('btn-reset');
  if (resetBtn) {
    resetBtn.addEventListener('click', async () => {
      await fetch('/api/reset', { method: 'POST' }).catch(() => null);
      // Reset charts
      rateLabels.length = 0; rateData.length = 0;
      totalLabels.length = 0; totalData.length = 0;
      if (rateChart)  { rateChart.data.labels  = []; rateChart.data.datasets[0].data  = []; rateChart.update('none'); }
      if (totalChart) { totalChart.data.labels = []; totalChart.data.datasets[0].data = []; totalChart.update('none'); }
      setVal('val-total', '0.00');
      setVal('val-reps',  '0');
      // Reset timer
      startTimer();
    });
  }
}

// ---- Boot ----
document.addEventListener('DOMContentLoaded', () => {
  initCharts();
  initButtons();
  startTimer();
  connectSSE();

  // Fallback poll if EventSource unavailable
  if (!window.EventSource) {
    console.warn('[Burn-Ex] EventSource not supported — falling back to polling');
    setInterval(async () => {
      try { applyState(await (await fetch('/api/state')).json()); } catch (_) {}
    }, 1200);
  }
});
