/**
 * Dashboard functionality for the Mycology Research Pipeline
 */

// Initialize the dashboard
function initDashboard() {
  // Set up event listeners
  setupEventListeners();
  
  // Initialize any dashboard components
  initDataRefresh();
  
  // Initialize tooltips and popovers
  initTooltips();
}

// Set up event listeners for dashboard controls
function setupEventListeners() {
  // Sample filter controls
  const sampleFilterForm = document.getElementById('sampleFilterForm');
  if (sampleFilterForm) {
    sampleFilterForm.addEventListener('submit', function(e) {
      e.preventDefault();
      // Implement filtering logic
      const species = document.getElementById('speciesFilter').value;
      const dateRange = document.getElementById('dateRangeFilter').value;
      filterSamples(species, dateRange);
    });
  }
  
  // Analysis date range selector
  const dateRangeSelector = document.getElementById('analysisDateRange');
  if (dateRangeSelector) {
    dateRangeSelector.addEventListener('change', function() {
      updateAnalysisCharts(this.value);
    });
  }
  
  // Export results button
  const exportButton = document.getElementById('exportResults');
  if (exportButton) {
    exportButton.addEventListener('click', exportDashboardData);
  }
  
  // Refresh dashboard button
  const refreshButton = document.getElementById('refreshDashboard');
  if (refreshButton) {
    refreshButton.addEventListener('click', refreshDashboardData);
  }
}

// Initialize data refresh for dashboard components
function initDataRefresh() {
  // Auto-refresh dashboard data every 5 minutes
  setInterval(function() {
    refreshDashboardData();
  }, 5 * 60 * 1000);
}

// Filter samples based on criteria
function filterSamples(species, dateRange) {
  // Show loading indicator
  showLoading();
  
  // In a real application, this would make a request to the server
  // For now, we'll just simulate filtering by updating the UI after a delay
  setTimeout(function() {
    // This would be replaced by logic to handle the filtered data
    console.log(`Filtering by species: ${species}, date range: ${dateRange}`);
    
    // Hide loading indicator
    hideLoading();
    
    // Show toast notification
    showNotification('Samples filtered successfully');
  }, 500);
}

// Update analysis charts based on date range
function updateAnalysisCharts(dateRange) {
  // Show loading indicator
  showLoading();
  
  // In a real application, this would make a request to the server
  // For now, we'll just simulate updating by refreshing the UI after a delay
  setTimeout(function() {
    console.log(`Updating charts for date range: ${dateRange}`);
    
    // This would be replaced by logic to update the charts with new data
    
    // Hide loading indicator
    hideLoading();
    
    // Show toast notification
    showNotification('Analysis charts updated');
  }, 500);
}

// Export dashboard data
function exportDashboardData() {
  // Show loading indicator
  showLoading();
  
  // In a real application, this would generate and download a file
  // For now, we'll just simulate export by showing a notification after a delay
  setTimeout(function() {
    console.log('Exporting dashboard data');
    
    // Hide loading indicator
    hideLoading();
    
    // Show toast notification
    showNotification('Dashboard data exported successfully');
  }, 1000);
}

// Refresh all dashboard data
function refreshDashboardData() {
  // Show loading indicator
  showLoading();
  
  // In a real application, this would make requests to refresh data
  // For now, we'll just simulate refresh by updating the UI after a delay
  setTimeout(function() {
    console.log('Refreshing dashboard data');
    
    // Here we would reload all dashboard components
    
    // Hide loading indicator
    hideLoading();
    
    // Show toast notification
    showNotification('Dashboard refreshed');
  }, 1000);
}

// Show the loading spinner
function showLoading() {
  // Create the spinner overlay if it doesn't exist
  let spinnerOverlay = document.getElementById('spinnerOverlay');
  if (!spinnerOverlay) {
    spinnerOverlay = document.createElement('div');
    spinnerOverlay.id = 'spinnerOverlay';
    spinnerOverlay.className = 'spinner-overlay';
    spinnerOverlay.innerHTML = `
      <div class="spinner-border" role="status">
        <span class="sr-only">Loading...</span>
      </div>
    `;
    document.body.appendChild(spinnerOverlay);
  }
  
  // Show the spinner
  spinnerOverlay.style.display = 'flex';
}

// Hide the loading spinner
function hideLoading() {
  const spinnerOverlay = document.getElementById('spinnerOverlay');
  if (spinnerOverlay) {
    spinnerOverlay.style.display = 'none';
  }
}

// Show a notification toast
function showNotification(message, type = 'success') {
  // Create toast container if it doesn't exist
  let toastContainer = document.getElementById('toastContainer');
  if (!toastContainer) {
    toastContainer = document.createElement('div');
    toastContainer.id = 'toastContainer';
    toastContainer.className = 'toast-container position-fixed top-0 end-0 p-3';
    document.body.appendChild(toastContainer);
  }
  
  // Create a unique ID for this toast
  const toastId = 'toast-' + Date.now();
  
  // Create the toast element
  const toast = document.createElement('div');
  toast.id = toastId;
  toast.className = `toast align-items-center text-white bg-${type} border-0`;
  toast.setAttribute('role', 'alert');
  toast.setAttribute('aria-live', 'assertive');
  toast.setAttribute('aria-atomic', 'true');
  
  // Set the toast content
  toast.innerHTML = `
    <div class="d-flex">
      <div class="toast-body">
        ${message}
      </div>
      <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast" aria-label="Close"></button>
    </div>
  `;
  
  // Add the toast to the container
  toastContainer.appendChild(toast);
  
  // Initialize and show the toast
  const toastInstance = new bootstrap.Toast(toast, {
    autohide: true,
    delay: 3000
  });
  toastInstance.show();
  
  // Remove the toast after it's hidden
  toast.addEventListener('hidden.bs.toast', function() {
    toast.remove();
  });
}

// Initialize tooltips and popovers
function initTooltips() {
  // Initialize all tooltips
  const tooltips = document.querySelectorAll('[data-bs-toggle="tooltip"]');
  tooltips.forEach(tooltip => {
    new bootstrap.Tooltip(tooltip);
  });
  
  // Initialize all popovers
  const popovers = document.querySelectorAll('[data-bs-toggle="popover"]');
  popovers.forEach(popover => {
    new bootstrap.Popover(popover);
  });
}

// Monitor visibility of dashboard sections
function monitorSectionVisibility() {
  // Use Intersection Observer to detect when sections are visible
  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        // The section is now visible
        entry.target.classList.add('section-visible');
        
        // Animate any charts in this section if they haven't been animated yet
        if (!entry.target.dataset.animated) {
          const charts = entry.target.querySelectorAll('.chart-container');
          charts.forEach(chart => {
            // Add animation class
            chart.classList.add('chart-animate');
          });
          
          entry.target.dataset.animated = 'true';
        }
      }
    });
  }, {
    threshold: 0.1 // 10% of the element must be visible
  });
  
  // Observe all dashboard sections
  const sections = document.querySelectorAll('.dashboard-section');
  sections.forEach(section => {
    observer.observe(section);
  });
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
  initDashboard();
  monitorSectionVisibility();
  
  // If we're on the dashboard page, initialize the charts
  if (document.querySelector('.dashboard-content')) {
    // Check if charts.js is loaded and initialize charts
    if (typeof initDashboardCharts === 'function') {
      initDashboardCharts();
    }
  }
});
