// WorkforceAnalytics - Chart.js Configuration & Helpers
Chart.defaults.color = '#94a3b8';
Chart.defaults.font.family = "'Inter', system-ui, sans-serif";
Chart.defaults.plugins.tooltip.backgroundColor = '#0f172a';
Chart.defaults.plugins.tooltip.borderColor = '#334155';
Chart.defaults.plugins.tooltip.borderWidth = 1;
Chart.defaults.plugins.tooltip.padding = 10;
Chart.defaults.plugins.tooltip.cornerRadius = 8;

const CHART_COLORS = {
  blue: '#38bdf8',
  purple: '#8b5cf6',
  green: '#10b981',
  yellow: '#f59e0b',
  red: '#ef4444',
  teal: '#14b8a6',
  pink: '#ec4899',
  border: '#334155',
  grid: 'rgba(51, 65, 85, 0.4)'
};

function createDonutChart(canvasId, labels, data, colors) {
  const ctx = document.getElementById(canvasId);
  if (!ctx) return null;
  return new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels: labels,
      datasets: [{
        data: data,
        backgroundColor: colors || [CHART_COLORS.green, CHART_COLORS.yellow, CHART_COLORS.red],
        borderWidth: 2,
        borderColor: '#1e293b'
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      cutout: '70%',
      plugins: {
        legend: { position: 'bottom', labels: { boxWidth: 12, padding: 16 } }
      }
    }
  });
}

function createBarChart(canvasId, labels, data, labelName, color) {
  const ctx = document.getElementById(canvasId);
  if (!ctx) return null;
  return new Chart(ctx, {
    type: 'bar',
    data: {
      labels: labels,
      datasets: [{
        label: labelName,
        data: data,
        backgroundColor: color || CHART_COLORS.blue,
        borderRadius: 6,
        borderSkipped: false
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false }
      },
      scales: {
        x: { grid: { display: false } },
        y: { grid: { color: CHART_COLORS.grid }, beginAtZero: true }
      }
    }
  });
}

function createRadarChart(canvasId, labels, actualData, targetData) {
  const ctx = document.getElementById(canvasId);
  if (!ctx) return null;
  return new Chart(ctx, {
    type: 'radar',
    data: {
      labels: labels,
      datasets: [
        {
          label: 'Current Proficiency',
          data: actualData,
          backgroundColor: 'rgba(56, 189, 248, 0.25)',
          borderColor: '#38bdf8',
          pointBackgroundColor: '#38bdf8',
          borderWidth: 2
        },
        {
          label: 'Target Benchmark',
          data: targetData,
          backgroundColor: 'rgba(139, 92, 246, 0.15)',
          borderColor: '#8b5cf6',
          pointBackgroundColor: '#8b5cf6',
          borderWidth: 2,
          borderDash: [4, 4]
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        r: {
          angleLines: { color: CHART_COLORS.border },
          grid: { color: CHART_COLORS.grid },
          pointLabels: { color: '#cbd5e1', font: { size: 12, weight: 600 } },
          suggestedMin: 0,
          suggestedMax: 5
        }
      },
      plugins: {
        legend: { position: 'top', labels: { boxWidth: 12 } }
      }
    }
  });
}
