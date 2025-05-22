// Main JavaScript for Mycology Research Pipeline

document.addEventListener('DOMContentLoaded', function() {
    // Initialize all tooltips
    var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });

    // Initialize all popovers
    var popoverTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="popover"]'));
    var popoverList = popoverTriggerList.map(function (popoverTriggerEl) {
        return new bootstrap.Popover(popoverTriggerEl);
    });
    
    // Auto-hide flash messages after 5 seconds
    setTimeout(function() {
        var flashMessages = document.querySelectorAll('.alert');
        flashMessages.forEach(function(alert) {
            var bsAlert = new bootstrap.Alert(alert);
            bsAlert.close();
        });
    }, 5000);
    
    // Handle image uploads preview
    var imageUploadInputs = document.querySelectorAll('input[type="file"][data-preview]');
    imageUploadInputs.forEach(function(input) {
        var previewElement = document.getElementById(input.dataset.preview);
        if (previewElement) {
            input.addEventListener('change', function(e) {
                if (input.files && input.files[0]) {
                    var reader = new FileReader();
                    reader.onload = function(e) {
                        previewElement.src = e.target.result;
                        previewElement.style.display = 'block';
                    }
                    reader.readAsDataURL(input.files[0]);
                }
            });
        }
    });
    
    // Add loading overlay for forms with data-loading attribute
    var loadingForms = document.querySelectorAll('form[data-loading]');
    loadingForms.forEach(function(form) {
        form.addEventListener('submit', function(e) {
            // Don't show loading overlay for invalid forms
            if (!form.checkValidity()) {
                return;
            }
            
            // Create and append the loading overlay
            var loadingOverlay = document.createElement('div');
            loadingOverlay.className = 'loading-overlay';
            
            var spinner = document.createElement('div');
            spinner.className = 'loading-spinner';
            loadingOverlay.appendChild(spinner);
            
            // Add message if specified
            if (form.dataset.loadingMessage) {
                var message = document.createElement('div');
                message.className = 'text-white mt-3';
                message.textContent = form.dataset.loadingMessage;
                loadingOverlay.appendChild(message);
            }
            
            document.body.appendChild(loadingOverlay);
        });
    });
    
    // Initialize any charts if Chart.js is loaded
    if (typeof Chart !== 'undefined') {
        initializeCharts();
    }
});

// Function to initialize charts
function initializeCharts() {
    // Sample charts initialization code
    var chartElements = document.querySelectorAll('[data-chart]');
    chartElements.forEach(function(element) {
        var chartType = element.dataset.chart;
        var chartData = JSON.parse(element.dataset.chartData || '{}');
        var chartOptions = JSON.parse(element.dataset.chartOptions || '{}');
        
        new Chart(element, {
            type: chartType,
            data: chartData,
            options: chartOptions
        });
    });
}