/**
 * planner.js — Burn-Ex AI Workout Planner
 * Uses OpenRouter API (server-side proxy) with nvidia/nemotron-3.5-lightning:free
 * Generates weekly & monthly workout plans and renders interactive checklists
 */

'use strict';

// ─── State ───────────────────────────────────────────────────
const PlannerState = {
  currentPlan:   null,   // { type, goal, level, days, duration, weeks, weekPlan, monthPlan }
  viewMode:      'week', // 'week' | 'month'
  savedPlans:    JSON.parse(localStorage.getItem('burnex_plans') || '[]'),
  completedSets: JSON.parse(localStorage.getItem('burnex_completed') || '{}'),
};

// ─── DOM refs ────────────────────────────────────────────────
const $ = id => document.getElementById(id);
const plannerForm    = $('planner-form');
const generateBtn    = $('generate-btn');
const plansOutput    = $('plans-output');
const weekTabBtn     = $('week-tab-btn');
const monthTabBtn    = $('month-tab-btn');
const weekView       = $('week-view');
const monthView      = $('month-view');
const savedPlansWrap = $('saved-plans-wrap');

// ─── Day pills ───────────────────────────────────────────────
const dayPills = document.querySelectorAll('.day-pill');
let selectedDays = ['Mon', 'Wed', 'Fri'];

dayPills.forEach(pill => {
  if (selectedDays.includes(pill.dataset.day)) pill.classList.add('selected');
  pill.addEventListener('click', () => {
    const day = pill.dataset.day;
    if (selectedDays.includes(day)) {
      if (selectedDays.length === 1) return; // keep at least 1
      selectedDays = selectedDays.filter(d => d !== day);
      pill.classList.remove('selected');
    } else {
      selectedDays.push(day);
      pill.classList.add('selected');
    }
  });
});

// ─── Plan tab switcher ───────────────────────────────────────
weekTabBtn?.addEventListener('click', () => {
  weekTabBtn.classList.add('active');
  monthTabBtn?.classList.remove('active');
  if (weekView)  weekView.style.display  = '';
  if (monthView) monthView.style.display = 'none';
  PlannerState.viewMode = 'week';
});

monthTabBtn?.addEventListener('click', () => {
  monthTabBtn.classList.add('active');
  weekTabBtn?.classList.remove('active');
  if (monthView) monthView.style.display = '';
  if (weekView)  weekView.style.display  = 'none';
  PlannerState.viewMode = 'month';
  if (PlannerState.currentPlan?.monthPlan) renderMonthPlan(PlannerState.currentPlan.monthPlan);
});

// ─── Generate Plan ───────────────────────────────────────────
plannerForm?.addEventListener('submit', async (e) => {
  e.preventDefault();
  const goal     = $('plan-goal')?.value     || 'weight_loss';
  const level    = $('plan-level')?.value    || 'beginner';
  const duration = parseInt($('plan-duration')?.value || '30', 10);
  const weeks    = parseInt($('plan-weeks')?.value    || '4', 10);

  if (selectedDays.length === 0) {
    showPlannerAlert('Please select at least one workout day.', 'error');
    return;
  }

  setGenerating(true);
  showThinkingIndicator(true);

  try {
    const plan = await generatePlanFromAI({ goal, level, duration, weeks, days: selectedDays });
    PlannerState.currentPlan = plan;

    // Persist
    PlannerState.savedPlans.unshift({ id: Date.now(), createdAt: new Date().toISOString(), ...plan });
    if (PlannerState.savedPlans.length > 10) PlannerState.savedPlans.pop();
    localStorage.setItem('burnex_plans', JSON.stringify(PlannerState.savedPlans));

    showThinkingIndicator(false);
    renderWeekPlan(plan.weekPlan, plan);
    renderSavedPlans();
    showPlannerAlert('✅ AI workout plan generated successfully!', 'success');
  } catch (err) {
    showThinkingIndicator(false);
    console.error('[Planner]', err);
    showPlannerAlert(`Failed to generate plan: ${err.message}`, 'error');
  } finally {
    setGenerating(false);
  }
});

// ─── API Call via Flask proxy ────────────────────────────────
async function generatePlanFromAI({ goal, level, duration, weeks, days }) {
  const daysStr = days.join(', ');
  const goalLabels = {
    weight_loss:   'weight loss and fat burning',
    muscle_gain:   'muscle gain and strength building',
    endurance:     'cardiovascular endurance and stamina',
    flexibility:   'flexibility, mobility and stretching',
    general:       'general fitness and overall wellness',
  };

  const prompt = `You are an expert certified personal trainer. Create a detailed ${weeks}-week workout plan for someone with the following profile:
- Goal: ${goalLabels[goal] || goal}
- Fitness Level: ${level}
- Workout Days: ${daysStr} (${days.length} days per week)
- Session Duration: ${duration} minutes each

Return ONLY valid JSON in this exact structure, no markdown, no extra text:
{
  "planName": "string (creative plan name)",
  "summary": "string (2-sentence summary)",
  "weekPlan": [
    {
      "day": "Monday",
      "type": "workout",
      "focus": "string (e.g. Upper Body Strength)",
      "duration": 30,
      "warmup": "string (2-3 warmup exercises)",
      "exercises": [
        { "name": "string", "sets": "string (e.g. 3x12)", "rest": "string (e.g. 60s)", "tip": "string (form tip)" }
      ],
      "cooldown": "string"
    }
  ],
  "restDays": ["list of rest day names"],
  "weeklyNotes": "string (training notes for the week)",
  "progressionTip": "string (how to progress over ${weeks} weeks)"
}

Important:
- Include ALL 7 days of the week. Non-workout days should have type: "rest"
- For rest days: { "day": "...", "type": "rest", "focus": "Active Recovery", "activities": "string" }
- Workout days must have 4-7 exercises
- Make exercises appropriate for ${level} level
- Include specific sets/reps for each exercise`;

  const response = await fetch('/api/generate-plan', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ prompt, goal, level, duration, weeks, days }),
  });

  if (!response.ok) {
    const err = await response.json().catch(() => ({}));
    throw new Error(err.error || `Server error ${response.status}`);
  }

  const data = await response.json();

  let parsed;
  try {
    // Strip markdown code fences if present
    let raw = data.content || data.plan || '';
    raw = raw.replace(/```json\n?/g, '').replace(/```\n?/g, '').trim();
    // Find JSON object
    const startIdx = raw.indexOf('{');
    const endIdx   = raw.lastIndexOf('}');
    if (startIdx !== -1 && endIdx !== -1) {
      raw = raw.slice(startIdx, endIdx + 1);
    }
    parsed = JSON.parse(raw);
  } catch (parseErr) {
    console.error('[Planner] JSON parse error:', parseErr, data);
    throw new Error('AI returned invalid JSON. Please try again.');
  }

  // Build month plan from weekly plan repeated
  const monthPlan = buildMonthPlan(parsed.weekPlan, weeks);

  return {
    type:      'ai',
    goal,
    level,
    duration,
    weeks,
    days,
    planName:  parsed.planName || 'My Fitness Plan',
    summary:   parsed.summary || '',
    weekPlan:  parsed.weekPlan || [],
    restDays:  parsed.restDays || [],
    weeklyNotes:     parsed.weeklyNotes || '',
    progressionTip:  parsed.progressionTip || '',
    monthPlan,
  };
}

// ─── Month Plan Builder ──────────────────────────────────────
function buildMonthPlan(weekPlan, weeks) {
  const days = [];
  const dayOrder = ['Sunday','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday'];

  for (let w = 0; w < weeks; w++) {
    weekPlan.forEach((day, idx) => {
      const dayNum = w * weekPlan.length + idx + 1;
      days.push({ ...day, weekNum: w + 1, dayNum });
    });
  }
  return days;
}

// ─── Render Week Plan ────────────────────────────────────────
function renderWeekPlan(weekPlan, plan) {
  if (!weekView) return;

  // Show the output section
  plansOutput?.classList.remove('hidden');

  // Update plan header
  const nameEl = $('plan-name-display');
  if (nameEl) nameEl.textContent = plan.planName || 'Your Plan';
  const summaryEl = $('plan-summary-display');
  if (summaryEl) summaryEl.textContent = plan.summary || '';

  weekView.innerHTML = '';
  const grid = document.createElement('div');
  grid.className = 'week-plan-grid';

  const allDays = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'];
  const planMap = {};
  (weekPlan || []).forEach(d => { planMap[d.day] = d; });

  allDays.forEach(dayName => {
    const day = planMap[dayName] || { day: dayName, type: 'rest', focus: 'Rest Day', activities: 'Light stretching, walking, or complete rest.' };
    grid.appendChild(renderDayCard(day));
  });

  weekView.appendChild(grid);

  // Show notes
  if (plan.weeklyNotes) {
    const notes = document.createElement('div');
    notes.className = 'dash-card';
    notes.style.marginTop = '12px';
    notes.innerHTML = `
      <div class="dash-card-header"><span class="dash-card-title">📋 Weekly Notes</span></div>
      <p style="font-size:13.5px;color:var(--text-sub);line-height:1.6">${plan.weeklyNotes}</p>
      ${plan.progressionTip ? `<p style="margin-top:12px;font-size:13.5px;color:var(--accent);font-weight:600;">💡 Progression: ${plan.progressionTip}</p>` : ''}
    `;
    weekView.appendChild(notes);
  }
}

function renderDayCard(day) {
  const isRest = day.type === 'rest';
  const card = document.createElement('div');
  card.className = 'day-plan-card';
  card.innerHTML = `
    <div class="day-plan-header">
      <div class="day-num ${isRest ? 'rest' : 'workout'}">${day.day.slice(0,3)}</div>
      <div class="day-plan-info">
        <div class="day-plan-name">${day.focus || (isRest ? 'Rest Day' : 'Workout')}</div>
        <div class="day-plan-meta">${isRest ? '😌 Recovery' : `⏱ ${day.duration || 30} min · ${(day.exercises||[]).length} exercises`}</div>
      </div>
      <span class="day-plan-toggle">▾</span>
    </div>
    <div class="day-plan-body">
      ${isRest
        ? `<div style="font-size:13.5px;color:var(--text-sub);padding:4px 0">${day.activities || 'Light stretching or walking.'}</div>`
        : `
          ${day.warmup ? `<div style="font-size:12.5px;color:var(--text-muted);margin-bottom:6px">🔆 Warmup: ${day.warmup}</div>` : ''}
          ${(day.exercises || []).map(ex => renderExercise(ex, day.day)).join('')}
          ${day.cooldown ? `<div style="font-size:12.5px;color:var(--text-muted);margin-top:6px">❄️ Cooldown: ${day.cooldown}</div>` : ''}
        `
      }
    </div>
  `;

  // Toggle accordion
  const header = card.querySelector('.day-plan-header');
  header.addEventListener('click', () => {
    card.classList.toggle('open');
  });

  return card;
}

function renderExercise(ex, dayName) {
  const key = `${dayName}_${ex.name}`.replace(/\s+/g,'_');
  const isDone = PlannerState.completedSets[key] === true;
  return `
    <div class="exercise-item ${isDone ? 'done' : ''}" data-key="${key}">
      <div class="exercise-check ${isDone ? 'checked' : ''}" onclick="toggleExercise('${key}', this)" role="checkbox" aria-checked="${isDone}" tabindex="0"></div>
      <div class="exercise-name">${ex.name}</div>
      <div class="exercise-detail">${ex.sets || ''}${ex.rest ? ' · ' + ex.rest : ''}</div>
    </div>
    ${ex.tip ? `<div style="font-size:11.5px;color:var(--text-muted);padding:2px 32px 4px;font-style:italic">💡 ${ex.tip}</div>` : ''}
  `;
}

// ─── Toggle Exercise Complete ────────────────────────────────
window.toggleExercise = function(key, checkEl) {
  const isDone = !PlannerState.completedSets[key];
  PlannerState.completedSets[key] = isDone;
  localStorage.setItem('burnex_completed', JSON.stringify(PlannerState.completedSets));

  checkEl.classList.toggle('checked', isDone);
  checkEl.setAttribute('aria-checked', isDone);
  const item = checkEl.closest('.exercise-item');
  if (item) item.classList.toggle('done', isDone);
};

// ─── Render Month Plan ───────────────────────────────────────
function renderMonthPlan(monthPlan) {
  if (!monthView) return;
  monthView.innerHTML = '';

  const DAYS = ['Sun','Mon','Tue','Wed','Thu','Fri','Sat'];
  const weekMap = {};
  monthPlan.forEach(day => {
    const w = day.weekNum || 1;
    if (!weekMap[w]) weekMap[w] = [];
    weekMap[w].push(day);
  });

  Object.keys(weekMap).sort((a,b)=>a-b).forEach(wNum => {
    const weekDays = weekMap[wNum];
    const weekTitle = document.createElement('div');
    weekTitle.style.cssText = 'font-size:13px;font-weight:700;color:var(--text-sub);margin:12px 0 6px;text-transform:uppercase;letter-spacing:.8px';
    weekTitle.textContent = `Week ${wNum}`;
    monthView.appendChild(weekTitle);

    const grid = document.createElement('div');
    grid.className = 'month-plan-grid';

    // Headers
    DAYS.forEach(d => {
      const h = document.createElement('div');
      h.className = 'month-day-header';
      h.textContent = d;
      grid.appendChild(h);
    });

    weekDays.forEach(day => {
      const cell = document.createElement('div');
      const isRest = day.type === 'rest';
      cell.className = `month-day-cell ${isRest ? 'rest-day' : ''}`;
      cell.innerHTML = `
        <div class="month-day-num">${day.day?.slice(0,3) || ''}</div>
        <span class="month-day-tag ${isRest ? 'tag-rest' : 'tag-workout'}">${isRest ? 'Rest' : 'Workout'}</span>
        <div class="month-exercise-preview">${isRest ? day.activities || 'Recovery' : (day.exercises||[]).slice(0,2).map(e=>e.name).join(', ') || day.focus || ''}</div>
      `;
      grid.appendChild(cell);
    });

    monthView.appendChild(grid);
  });
}

// ─── Saved Plans ─────────────────────────────────────────────
function renderSavedPlans() {
  if (!savedPlansWrap) return;
  if (PlannerState.savedPlans.length === 0) {
    savedPlansWrap.innerHTML = '<p style="font-size:13px;color:var(--text-muted)">No saved plans yet.</p>';
    return;
  }

  savedPlansWrap.innerHTML = PlannerState.savedPlans.map((p, i) => `
    <div class="activity-item" style="cursor:pointer" onclick="loadSavedPlan(${i})">
      <div class="act-dot" style="background:var(--accent)"></div>
      <div class="act-name">${p.planName || 'Plan ' + (i+1)}</div>
      <div class="act-time">${new Date(p.createdAt).toLocaleDateString()}</div>
      <div class="act-kcal">${p.weeks}w · ${(p.days||[]).length}d/wk</div>
    </div>
  `).join('');
}

window.loadSavedPlan = function(idx) {
  const plan = PlannerState.savedPlans[idx];
  if (!plan) return;
  PlannerState.currentPlan = plan;
  renderWeekPlan(plan.weekPlan, plan);
  if (plan.monthPlan) renderMonthPlan(plan.monthPlan);
  plansOutput?.classList.remove('hidden');
};

// ─── Helpers ─────────────────────────────────────────────────
function setGenerating(loading) {
  if (!generateBtn) return;
  generateBtn.disabled = loading;
  generateBtn.classList.toggle('loading', loading);
  const spinner = generateBtn.querySelector('.generate-spinner');
  if (spinner) spinner.style.display = loading ? 'block' : 'none';
  const label = generateBtn.querySelector('.gen-label');
  if (label) label.textContent = loading ? 'Generating…' : '✨ Generate Plan';
}

function showThinkingIndicator(show) {
  const el = $('ai-thinking');
  if (el) el.style.display = show ? 'flex' : 'none';
}

function showPlannerAlert(msg, type = 'error') {
  const el = $('planner-alert');
  if (!el) return;
  el.textContent = msg;
  el.className = `auth-alert ${type}`;
  setTimeout(() => { el.className = 'auth-alert'; }, 5000);
}

// ─── Init ─────────────────────────────────────────────────────
renderSavedPlans();
