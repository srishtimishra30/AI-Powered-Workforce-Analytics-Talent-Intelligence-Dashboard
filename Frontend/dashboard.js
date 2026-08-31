// Dashboard Interactivity and Chart Initializers
document.addEventListener('DOMContentLoaded', () => {
  if (window.dashboardData) {
    const data = window.dashboardData;
    
    createDonutChart('riskDonutChart', 
      ['Low Risk (<35%)', 'Medium Risk (35-65%)', 'High Risk (>65%)'],
      [data.risk_counts.Low || 0, data.risk_counts.Medium || 0, data.risk_counts.High || 0],
      ['#10b981', '#f59e0b', '#ef4444']
    );
    
    createBarChart('deptAttritionChart',
      Object.keys(data.dept_attrition),
      Object.values(data.dept_attrition),
      'Attrition Rate (%)',
      '#38bdf8'
    );
    
    createDonutChart('readinessDonutChart',
      ['Ready', 'Developing', 'Needs Upskilling'],
      [data.readiness_counts.Ready || 0, data.readiness_counts.Developing || 0, data.readiness_counts['Needs Upskilling'] || 0],
      ['#10b981', '#38bdf8', '#ef4444']
    );
    
    const satCtx = document.getElementById('satisfactionChart');
    if (satCtx && data.sat_turnover) {
      new Chart(satCtx, {
        type: 'line',
        data: {
          labels: ['Rating 1 (Poor)', 'Rating 2 (Fair)', 'Rating 3 (Good)', 'Rating 4 (High)'],
          datasets: [{
            label: 'Attrition Probability %',
            data: data.sat_turnover,
            borderColor: '#ef4444',
            backgroundColor: 'rgba(239, 68, 68, 0.1)',
            fill: true,
            tension: 0.35,
            pointBackgroundColor: '#ef4444'
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          scales: {
            y: { grid: { color: 'rgba(51, 65, 85, 0.4)' }, beginAtZero: true }
          }
        }
      });
    }
  }
});
