// Employees directory filtering and search
document.addEventListener('DOMContentLoaded', () => {
  const searchInput = document.getElementById('searchEmployee');
  const deptFilter = document.getElementById('filterDept');
  const riskFilter = document.getElementById('filterRisk');
  const tableRows = document.querySelectorAll('#employeeTableBody tr');

  function filterTable() {
    const query = (searchInput ? searchInput.value : '').toLowerCase();
    const dept = deptFilter ? deptFilter.value : '';
    const risk = riskFilter ? riskFilter.value : '';

    tableRows.forEach(row => {
      const name = row.dataset.name ? row.dataset.name.toLowerCase() : '';
      const rowDept = row.dataset.dept || '';
      const rowRisk = row.dataset.risk || '';

      const matchesSearch = name.includes(query) || row.dataset.id.includes(query);
      const matchesDept = !dept || rowDept === dept;
      const matchesRisk = !risk || rowRisk === risk;

      if (matchesSearch && matchesDept && matchesRisk) {
        row.style.display = '';
      } else {
        row.style.display = 'none';
      }
    });
  }

  if (searchInput) searchInput.addEventListener('input', filterTable);
  if (deptFilter) deptFilter.addEventListener('change', filterTable);
  if (riskFilter) riskFilter.addEventListener('change', filterTable);
});
