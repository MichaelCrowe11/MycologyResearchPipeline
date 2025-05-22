/**
 * Main JavaScript functions for Mycology Research Pipeline
 */

document.addEventListener('DOMContentLoaded', function() {
    // Initialize all tooltips
    var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
    
    // Initialize all popovers
    var popoverTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="popover"]'));
    popoverTriggerList.map(function (popoverTriggerEl) {
        return new bootstrap.Popover(popoverTriggerEl);
    });
    
    // Add animation to cards
    const cards = document.querySelectorAll('.card');
    cards.forEach(card => {
        card.addEventListener('mouseenter', function() {
            this.classList.add('card-hover');
        });
        
        card.addEventListener('mouseleave', function() {
            this.classList.remove('card-hover');
        });
    });
    
    // Password strength meter
    const passwordInput = document.getElementById('password');
    const passwordStrength = document.getElementById('password-strength');
    
    if (passwordInput && passwordStrength) {
        passwordInput.addEventListener('input', function() {
            const password = this.value;
            let strength = 0;
            
            // Check password length
            if (password.length >= 8) {
                strength += 1;
            }
            
            // Check for lowercase letters
            if (password.match(/[a-z]/)) {
                strength += 1;
            }
            
            // Check for uppercase letters
            if (password.match(/[A-Z]/)) {
                strength += 1;
            }
            
            // Check for numbers
            if (password.match(/[0-9]/)) {
                strength += 1;
            }
            
            // Check for special characters
            if (password.match(/[^a-zA-Z0-9]/)) {
                strength += 1;
            }
            
            // Update strength meter
            switch (strength) {
                case 0:
                case 1:
                    passwordStrength.className = 'progress-bar bg-danger';
                    passwordStrength.style.width = '20%';
                    passwordStrength.textContent = 'Weak';
                    break;
                case 2:
                    passwordStrength.className = 'progress-bar bg-warning';
                    passwordStrength.style.width = '40%';
                    passwordStrength.textContent = 'Fair';
                    break;
                case 3:
                    passwordStrength.className = 'progress-bar bg-info';
                    passwordStrength.style.width = '60%';
                    passwordStrength.textContent = 'Good';
                    break;
                case 4:
                    passwordStrength.className = 'progress-bar bg-primary';
                    passwordStrength.style.width = '80%';
                    passwordStrength.textContent = 'Strong';
                    break;
                case 5:
                    passwordStrength.className = 'progress-bar bg-success';
                    passwordStrength.style.width = '100%';
                    passwordStrength.textContent = 'Very Strong';
                    break;
            }
        });
    }
    
    // Confirm password validation
    const confirmPasswordInput = document.getElementById('confirm_password');
    const passwordMatchFeedback = document.getElementById('password-match-feedback');
    
    if (passwordInput && confirmPasswordInput && passwordMatchFeedback) {
        confirmPasswordInput.addEventListener('input', function() {
            if (this.value === passwordInput.value) {
                passwordMatchFeedback.textContent = 'Passwords match';
                passwordMatchFeedback.className = 'form-text text-success';
                this.classList.remove('is-invalid');
                this.classList.add('is-valid');
            } else {
                passwordMatchFeedback.textContent = 'Passwords do not match';
                passwordMatchFeedback.className = 'form-text text-danger';
                this.classList.remove('is-valid');
                this.classList.add('is-invalid');
            }
        });
    }
    
    // Membership plan selection
    const planOptions = document.querySelectorAll('.plan-option');
    if (planOptions.length > 0) {
        planOptions.forEach(option => {
            option.addEventListener('click', function() {
                // Remove active class from all options
                planOptions.forEach(opt => opt.classList.remove('active'));
                
                // Add active class to selected option
                this.classList.add('active');
                
                // Update hidden input value
                const planInput = document.getElementById('selected_plan');
                if (planInput) {
                    planInput.value = this.dataset.plan;
                }
                
                // Update summary section if it exists
                const planTitle = document.getElementById('summary-plan-title');
                const planPrice = document.getElementById('summary-plan-price');
                const planBillingPeriod = document.getElementById('summary-billing-period');
                
                if (planTitle && planPrice && planBillingPeriod) {
                    planTitle.textContent = this.dataset.planTitle;
                    planPrice.textContent = this.dataset.planPrice;
                    planBillingPeriod.textContent = this.dataset.planPeriod;
                }
            });
        });
    }
    
    // API key copy functionality
    const apiKeyCopyButtons = document.querySelectorAll('.copy-api-key');
    if (apiKeyCopyButtons.length > 0) {
        apiKeyCopyButtons.forEach(button => {
            button.addEventListener('click', function() {
                const apiKey = this.dataset.apiKey;
                
                // Copy to clipboard
                navigator.clipboard.writeText(apiKey).then(() => {
                    // Change button text temporarily
                    const originalText = this.innerHTML;
                    this.innerHTML = '<i class="fas fa-check me-2"></i>Copied!';
                    
                    // Revert back after 2 seconds
                    setTimeout(() => {
                        this.innerHTML = originalText;
                    }, 2000);
                });
            });
        });
    }
    
    // Show/hide API key
    const toggleApiKeyButtons = document.querySelectorAll('.toggle-api-key');
    if (toggleApiKeyButtons.length > 0) {
        toggleApiKeyButtons.forEach(button => {
            button.addEventListener('click', function() {
                const keySpan = document.getElementById(this.dataset.targetId);
                const isHidden = keySpan.classList.contains('text-secret');
                
                if (isHidden) {
                    keySpan.classList.remove('text-secret');
                    keySpan.textContent = this.dataset.apiKey;
                    this.innerHTML = '<i class="fas fa-eye-slash me-2"></i>Hide';
                } else {
                    keySpan.classList.add('text-secret');
                    keySpan.textContent = '••••••••••••••••••••••••••';
                    this.innerHTML = '<i class="fas fa-eye me-2"></i>Show';
                }
            });
        });
    }
});