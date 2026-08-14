/**
 * auth.js — Burn-Ex Authentication Logic
 * Handles: Login, Sign-up, Forgot password, localStorage session
 */

'use strict';

// ─── Constants ──────────────────────────────────────────────
const STORAGE_KEY = 'burnex_user';
const USERS_KEY   = 'burnex_users';

// ─── Helpers ────────────────────────────────────────────────
const qs  = (sel, ctx = document) => ctx.querySelector(sel);
const qsa = (sel, ctx = document) => [...ctx.querySelectorAll(sel)];

function showAlert(msg, type = 'error') {
  const el = qs('#auth-alert');
  el.textContent = msg;
  el.className = `auth-alert ${type}`;
  setTimeout(() => { el.className = 'auth-alert'; }, 5000);
}

function setLoading(btn, loading) {
  btn.disabled = loading;
  btn.classList.toggle('loading', loading);
}

function isValidEmail(email) {
  return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email.trim());
}

function showField(id, show) {
  const el = qs(id);
  if (el) el.classList.toggle('visible', show);
}

function markError(inputId, errId, show) {
  const input = qs(inputId);
  const err   = qs(errId);
  if (input) input.classList.toggle('error', show);
  if (err)   err.classList.toggle('visible', show);
}

// ─── Panel Switching ─────────────────────────────────────────
const panels = {
  login:  qs('#login-panel'),
  signup: qs('#signup-panel'),
  forgot: qs('#forgot-panel'),
};

function showPanel(name) {
  Object.entries(panels).forEach(([k, el]) => {
    if (el) el.style.display = k === name ? '' : 'none';
  });
  qs('#auth-alert').className = 'auth-alert'; // clear alerts
}

// ─── Password Toggle ─────────────────────────────────────────
function setupToggle(toggleId, inputId) {
  const btn   = qs(toggleId);
  const input = qs(inputId);
  if (!btn || !input) return;
  btn.addEventListener('click', () => {
    const visible = input.type === 'text';
    input.type = visible ? 'password' : 'text';
    btn.textContent = visible ? '👁' : '🙈';
  });
}
setupToggle('#toggle-login-pw',  '#login-password');
setupToggle('#toggle-signup-pw', '#signup-password');

// ─── Password Strength ──────────────────────────────────────
const pwInput = qs('#signup-password');
if (pwInput) {
  pwInput.addEventListener('input', () => {
    const pw = pwInput.value;
    const fill = qs('#strength-fill');
    if (!fill) return;
    let score = 0;
    if (pw.length >= 8)  score++;
    if (/[A-Z]/.test(pw))score++;
    if (/[0-9]/.test(pw))score++;
    if (/[^a-zA-Z0-9]/.test(pw)) score++;
    const colors = ['', '#EF4444', '#F97316', '#EAB308', '#10B981'];
    const widths  = ['0%', '25%', '50%', '75%', '100%'];
    fill.style.width = widths[score] || '0%';
    fill.style.background = colors[score] || '';
  });
}

// ─── Navigation Links ────────────────────────────────────────
qs('#go-signup')?.addEventListener('click',    e => { e.preventDefault(); showPanel('signup'); });
qs('#go-login')?.addEventListener('click',     e => { e.preventDefault(); showPanel('login'); });
qs('#forgot-link')?.addEventListener('click',  e => { e.preventDefault(); showPanel('forgot'); });
qs('#back-to-login')?.addEventListener('click',e => { e.preventDefault(); showPanel('login'); });

// ─── Users Store ─────────────────────────────────────────────
function getUsers() {
  try { return JSON.parse(localStorage.getItem(USERS_KEY)) || {}; }
  catch { return {}; }
}
function saveUsers(users) {
  localStorage.setItem(USERS_KEY, JSON.stringify(users));
}
function saveSession(user) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(user));
}

// ─── Auth Guard ──────────────────────────────────────────────
// If already logged in, redirect to main app
(function checkAuth() {
  try {
    const u = JSON.parse(localStorage.getItem(STORAGE_KEY));
    if (u && u.email) {
      window.location.href = '/';
    }
  } catch { /* ignore */ }
})();

// ─── LOGIN ───────────────────────────────────────────────────
const loginForm = qs('#login-form');
loginForm?.addEventListener('submit', async (e) => {
  e.preventDefault();

  const email  = qs('#login-email').value.trim();
  const pw     = qs('#login-password').value;
  let valid    = true;

  markError('#login-email', '#login-email-err', !isValidEmail(email));
  if (!isValidEmail(email)) valid = false;

  markError('#login-password', '#login-pw-err', pw.length < 6);
  if (pw.length < 6) valid = false;

  if (!valid) return;

  const btn = qs('#login-btn');
  setLoading(btn, true);

  // Simulate network delay
  await new Promise(r => setTimeout(r, 900));

  const users = getUsers();
  const user  = users[email.toLowerCase()];

  if (!user) {
    showAlert('No account found with that email. Please sign up first.');
    setLoading(btn, false);
    return;
  }
  if (user.password !== btoa(pw)) {
    showAlert('Incorrect password. Please try again.');
    setLoading(btn, false);
    return;
  }

  saveSession({ email: user.email, name: user.name, avatar: user.avatar || '' });
  showAlert('Login successful! Redirecting…', 'success');
  setTimeout(() => { window.location.href = '/'; }, 900);
});

// Google login (mock)
qs('#google-login-btn')?.addEventListener('click', async () => {
  const mockUser = {
    email: 'demo@burnex.app',
    name: 'Demo User',
    avatar: '',
  };
  const users = getUsers();
  if (!users[mockUser.email]) {
    users[mockUser.email] = { ...mockUser, password: btoa('demo123') };
    saveUsers(users);
  }
  saveSession(mockUser);
  showAlert('Google login successful! Redirecting…', 'success');
  setTimeout(() => { window.location.href = '/'; }, 900);
});

// ─── SIGN UP ─────────────────────────────────────────────────
const signupForm = qs('#signup-form');
signupForm?.addEventListener('submit', async (e) => {
  e.preventDefault();

  const fname = qs('#signup-fname').value.trim();
  const lname = qs('#signup-lname').value.trim();
  const email = qs('#signup-email').value.trim();
  const pw    = qs('#signup-password').value;
  let valid   = true;

  markError('#signup-fname', '#fname-err', !fname);
  if (!fname) valid = false;

  markError('#signup-lname', '#lname-err', !lname);
  if (!lname) valid = false;

  markError('#signup-email', '#signup-email-err', !isValidEmail(email));
  if (!isValidEmail(email)) valid = false;

  markError('#signup-password', '#signup-pw-err', pw.length < 8);
  if (pw.length < 8) valid = false;

  if (!valid) return;

  const btn = qs('#signup-btn');
  setLoading(btn, true);
  await new Promise(r => setTimeout(r, 900));

  const users = getUsers();
  const key   = email.toLowerCase();

  if (users[key]) {
    showAlert('An account with this email already exists. Please log in.');
    setLoading(btn, false);
    return;
  }

  const newUser = {
    email: key,
    name: `${fname} ${lname}`,
    firstName: fname,
    lastName:  lname,
    password:  btoa(pw),
    avatar:    '',
    joinedAt:  new Date().toISOString(),
  };
  users[key] = newUser;
  saveUsers(users);
  saveSession({ email: newUser.email, name: newUser.name });

  showAlert('Account created! Welcome to Burn-Ex 🔥', 'success');
  setTimeout(() => { window.location.href = '/'; }, 1000);
});

// Google signup (mock)
qs('#google-signup-btn')?.addEventListener('click', async () => {
  qs('#google-login-btn')?.click();
});

// ─── FORGOT PASSWORD ─────────────────────────────────────────
const forgotForm = qs('#forgot-form');
forgotForm?.addEventListener('submit', async (e) => {
  e.preventDefault();

  const email = qs('#forgot-email').value.trim();
  markError('#forgot-email', '#forgot-email-err', !isValidEmail(email));
  if (!isValidEmail(email)) return;

  const btn = qs('#forgot-btn');
  setLoading(btn, true);
  await new Promise(r => setTimeout(r, 1000));
  setLoading(btn, false);

  showAlert(`If an account exists for ${email}, a reset link has been sent.`, 'success');
});

// ─── Clear errors on input ───────────────────────────────────
qsa('.form-input').forEach(input => {
  input.addEventListener('input', () => {
    input.classList.remove('error');
    const errId = input.id.replace(/-/g, '_') + '_err';
    // try to find adjacent error span
    const err = input.closest('.form-group')?.querySelector('.field-error');
    if (err) err.classList.remove('visible');
  });
});
