/**
 * main.js - UI behaviour for APMS. No inline handlers are used anywhere (Content-Security-Policy),
 * so everything is wired up here through data-* attributes.
 */

// Categorical order for the light theme (checked for colour-blind separation and contrast on white).
const PALETTE = ['#2e5e9e', '#00937c', '#b7791f', '#8a55b0', '#c0392b', '#5c8ad6'];
// Fixed colours for outcomes, so "Failed" is always red and never just "series 2".
const OUTCOME = { Passed: '#1d7a52', Failed: '#b02a20', Withdrawn: '#8a94a3' };
const FONT = "'Segoe UI', system-ui, -apple-system, 'Helvetica Neue', Arial, sans-serif";

document.addEventListener('DOMContentLoaded', () => {
  const el = document.getElementById('currentDate');
  if (el) {
    el.textContent = new Date().toLocaleDateString('en-US', {
      weekday: 'short', year: 'numeric', month: 'short', day: 'numeric',
    });
  }

  // Mobile menu toggle
  const toggle = document.getElementById('navToggle');
  const nav = document.getElementById('siteNav');
  if (toggle && nav) {
    toggle.addEventListener('click', () => {
      const open = nav.classList.toggle('open');
      toggle.setAttribute('aria-expanded', open ? 'true' : 'false');
    });
  }

  // Auto-dismiss flash messages after 6 s
  document.querySelectorAll('.alert-dismissible').forEach(alert => {
    setTimeout(() => bootstrap.Alert.getOrCreateInstance(alert).close(), 6000);
  });

  // Confirm before destructive actions:  <form data-confirm="Delete this student?">
  document.querySelectorAll('form[data-confirm]').forEach(form => {
    form.addEventListener('submit', (e) => { if (!confirm(form.dataset.confirm)) e.preventDefault(); });
  });

  // Print button:  <button data-print>
  document.querySelectorAll('[data-print]').forEach(b => b.addEventListener('click', () => window.print()));

  // Copy text:  <button data-copy="#elementId">
  document.querySelectorAll('[data-copy]').forEach(b => b.addEventListener('click', async () => {
    const target = document.querySelector(b.dataset.copy);
    try {
      await navigator.clipboard.writeText(target.value || target.textContent);
      const old = b.innerHTML; b.textContent = 'Copied'; setTimeout(() => { b.innerHTML = old; }, 1500);
    } catch (err) { target.select && target.select(); }
  }));

  document.querySelectorAll('[data-bs-toggle="tooltip"]').forEach(t => new bootstrap.Tooltip(t));
  buildCharts();
});

/**
 * Charts are described by data-* attributes on a <canvas>:
 *   data-chart="bar|line|doughnut|scatter"  data-labels='[...]'  data-values='[...]'  data-label="Series"
 *   data-datasets='[{"label":..,"data":[..]}]' for several series, data-points for scatter,
 *   data-horizontal, data-max, data-unit
 */
function buildCharts() {
  if (typeof Chart === 'undefined') return;
  Chart.defaults.color = '#5d6979';
  Chart.defaults.borderColor = '#e4e8ee';
  Chart.defaults.font.family = FONT;
  Chart.defaults.font.size = 12;
  Chart.defaults.plugins.legend.labels.boxWidth = 12;
  Chart.defaults.plugins.tooltip.backgroundColor = '#0f2341';
  Chart.defaults.plugins.tooltip.padding = 8;
  Chart.defaults.plugins.tooltip.cornerRadius = 3;

  document.querySelectorAll('canvas[data-chart]').forEach(canvas => {
    const d = canvas.dataset;
    const type = d.chart;
    const labels = d.labels ? JSON.parse(d.labels) : [];
    let datasets;
    if (d.datasets) datasets = JSON.parse(d.datasets);
    else if (d.points) datasets = JSON.parse(d.points);
    else datasets = [{ label: d.label || '', data: JSON.parse(d.values), ...(d.color ? { color: d.color } : {}) }];

    datasets = datasets.map((s, i) => {
      const color = s.color || PALETTE[i % PALETTE.length];
      if (type === 'doughnut') {
        const colors = labels.map((l, j) => OUTCOME[l] || PALETTE[j % PALETTE.length]);
        return { ...s, backgroundColor: s.backgroundColor || colors, borderColor: '#fff', borderWidth: 2 };
      }
      if (type === 'line') return { borderColor: color, backgroundColor: color, borderWidth: 2, tension: .3, fill: false, pointRadius: 4, ...s };
      if (type === 'scatter') return { pointRadius: 4, backgroundColor: color + 'b3', borderColor: '#fff', borderWidth: 1, ...s };
      return { backgroundColor: color, borderRadius: 4, borderSkipped: 'start', maxBarThickness: 48, ...s };
    });

    const options = { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: datasets.length > 1 || type === 'doughnut' } } };
    if (type !== 'doughnut') {
      options.scales = { y: { beginAtZero: true, title: { display: !!d.unit, text: d.unit || '' } }, x: { grid: { display: false } } };
      if (d.max) options.scales.y.max = Number(d.max);
      if (d.xtitle) options.scales.x.title = { display: true, text: d.xtitle };
      if (d.horizontal !== undefined) {
        options.indexAxis = 'y';
        options.scales.x.grid = { display: true };
        options.scales.y.grid = { display: false };
      }
    }
    new Chart(canvas, { type, data: { labels, datasets }, options });
  });
}
