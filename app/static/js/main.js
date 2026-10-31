/**
 * main.js - UI behaviour for APMS. No inline handlers are used anywhere (Content-Security-Policy),
 * so everything is wired up here through data-* attributes.
 */

const PALETTE = ['#0d6efd', '#20c997', '#ffc107', '#dc3545', '#6f42c1', '#fd7e14', '#0dcaf0', '#adb5bd'];

document.addEventListener('DOMContentLoaded', () => {
  const el = document.getElementById('currentDate');
  if (el) {
    el.textContent = new Date().toLocaleDateString('en-US', {
      weekday: 'short', year: 'numeric', month: 'short', day: 'numeric',
    });
  }

  // Mobile sidebar toggle
  const toggle = document.getElementById('sidebarToggle');
  const sidebar = document.querySelector('.sidebar');
  if (toggle && sidebar) {
    toggle.addEventListener('click', () => sidebar.classList.toggle('open'));
    document.addEventListener('click', (e) => {
      if (!sidebar.contains(e.target) && !toggle.contains(e.target)) sidebar.classList.remove('open');
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
  Chart.defaults.color = '#8a9099';
  Chart.defaults.borderColor = 'rgba(255,255,255,0.07)';
  Chart.defaults.font.family = "'Segoe UI', system-ui, sans-serif";
  Chart.defaults.font.size = 12;

  document.querySelectorAll('canvas[data-chart]').forEach(canvas => {
    const d = canvas.dataset;
    const type = d.chart;
    const labels = d.labels ? JSON.parse(d.labels) : [];
    let datasets;
    if (d.datasets) datasets = JSON.parse(d.datasets);
    else if (d.points) datasets = JSON.parse(d.points);
    else datasets = [{ label: d.label || '', data: JSON.parse(d.values) }];

    datasets = datasets.map((s, i) => {
      const color = s.color || PALETTE[i % PALETTE.length];
      if (type === 'doughnut') return { ...s, backgroundColor: s.backgroundColor || PALETTE, borderWidth: 0 };
      if (type === 'line') return { borderColor: color, backgroundColor: color + '33', tension: .3, fill: false, pointRadius: 3, ...s };
      if (type === 'scatter') return { pointRadius: 3, backgroundColor: color + 'aa', ...s };
      return { backgroundColor: color, borderRadius: 4, ...s };
    });

    const options = { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: datasets.length > 1 || type === 'doughnut' } } };
    if (type !== 'doughnut') {
      options.scales = { y: { beginAtZero: true, title: { display: !!d.unit, text: d.unit || '' } }, x: {} };
      if (d.max) options.scales.y.max = Number(d.max);
      if (d.xtitle) options.scales.x.title = { display: true, text: d.xtitle };
      if (d.horizontal !== undefined) options.indexAxis = 'y';
    }
    new Chart(canvas, { type, data: { labels, datasets }, options });
  });
}
