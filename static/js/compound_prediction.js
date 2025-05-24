/**
 * Compound Prediction JavaScript Module
 * 
 * This module handles the user interaction for compound prediction functionality
 * in the Mycology Research Pipeline.
 */

document.addEventListener('DOMContentLoaded', function() {
    // Form submission handler for prediction
    const predictionForm = document.getElementById('prediction-form');
    if (predictionForm) {
        predictionForm.addEventListener('submit', function(e) {
            e.preventDefault();
            runPrediction();
        });
    }
    
    // Sample selection handler
    const sampleSelect = document.getElementById('sampleSelect');
    if (sampleSelect) {
        sampleSelect.addEventListener('change', function() {
            updatePredictionForm();
        });
    }
    
    // Analysis method toggles
    const analysisCheckboxes = document.querySelectorAll('.analysis-method-checkbox');
    analysisCheckboxes.forEach(checkbox => {
        checkbox.addEventListener('change', function() {
            updateMethodologySection();
        });
    });
});

/**
 * Run a compound prediction based on form inputs
 */
function runPrediction() {
    // Show loading state
    document.getElementById('prediction-loading').classList.remove('d-none');
    document.getElementById('prediction-results').classList.add('d-none');
    
    // Get form data
    const formData = new FormData(document.getElementById('prediction-form'));
    
    // Send the prediction request
    fetch('/ml-prediction', {
        method: 'POST',
        body: formData
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return response.json();
    })
    .then(data => {
        displayPredictionResults(data);
    })
    .catch(error => {
        console.error('Error running prediction:', error);
        displayError(error);
    })
    .finally(() => {
        document.getElementById('prediction-loading').classList.add('d-none');
    });
}

/**
 * Update the prediction form based on selected sample
 */
function updatePredictionForm() {
    const sampleId = document.getElementById('sampleSelect').value;
    const speciesElement = document.getElementById('speciesSelect');
    
    if (!sampleId) return;
    
    // In a production environment, this would fetch sample details
    // For demo, we'll just update some fields
    
    if (sampleId === "1") { // Lion's Mane
        if (speciesElement) speciesElement.value = "Hericium erinaceus";
        setDefaultAnalysisMethods('hplc', 'ms', 'cv');
    } else if (sampleId === "2") { // Reishi
        if (speciesElement) speciesElement.value = "Ganoderma lucidum";
        setDefaultAnalysisMethods('hplc', 'ms', 'spectral');
    } else if (sampleId === "3") { // Turkey Tail
        if (speciesElement) speciesElement.value = "Trametes versicolor";
        setDefaultAnalysisMethods('hplc', 'ms', 'cv');
    } else if (sampleId === "4") { // Cordyceps
        if (speciesElement) speciesElement.value = "Cordyceps militaris";
        setDefaultAnalysisMethods('ms', 'spectral', 'nmr');
    }
    
    updateMethodologySection();
}

/**
 * Set default analysis methods based on sample type
 */
function setDefaultAnalysisMethods(...methods) {
    // Reset all checkboxes
    document.querySelectorAll('.analysis-method-checkbox').forEach(checkbox => {
        checkbox.checked = false;
    });
    
    // Set the specified methods
    methods.forEach(method => {
        const checkbox = document.getElementById(method + 'Analysis');
        if (checkbox) checkbox.checked = true;
    });
}

/**
 * Update the methodology section based on selected analysis methods
 */
function updateMethodologySection() {
    // In a real implementation, this could update methodology descriptions
    // based on selected analysis methods
    
    const methodCount = document.querySelectorAll('.analysis-method-checkbox:checked').length;
    const confidenceElement = document.getElementById('prediction-confidence');
    
    if (confidenceElement) {
        if (methodCount >= 3) {
            confidenceElement.textContent = "High";
            confidenceElement.className = "badge bg-success";
        } else if (methodCount >= 2) {
            confidenceElement.textContent = "Medium";
            confidenceElement.className = "badge bg-warning";
        } else {
            confidenceElement.textContent = "Low";
            confidenceElement.className = "badge bg-danger";
        }
    }
}

/**
 * Display prediction results in the UI
 */
function displayPredictionResults(data) {
    const resultsContainer = document.getElementById('prediction-results');
    if (!resultsContainer) return;
    
    resultsContainer.classList.remove('d-none');
    
    // Display compounds
    const compoundsContainer = document.getElementById('predicted-compounds');
    if (compoundsContainer && data.compounds) {
        compoundsContainer.innerHTML = '';
        
        data.compounds.forEach(compound => {
            const compoundCard = createCompoundCard(compound);
            compoundsContainer.appendChild(compoundCard);
        });
    }
    
    // Update model metrics
    updateModelMetrics(data.model_metrics);
    
    // Update bioactivity info
    if (data.bioactivity_type) {
        const bioactivityElement = document.getElementById('bioactivity-type');
        if (bioactivityElement) {
            bioactivityElement.textContent = data.bioactivity_type;
        }
    }
    
    // Display feature importance
    displayFeatureImportance(data.model_metrics?.feature_importance);
}

/**
 * Create a compound card element
 */
function createCompoundCard(compound) {
    const card = document.createElement('div');
    card.className = 'col-md-6 col-lg-4 mb-4';
    
    const confidenceClass = compound.confidence >= 85 ? 'high-score' : 
                           (compound.confidence >= 70 ? 'medium-score' : 'low-score');
    
    card.innerHTML = `
        <div class="card prediction-card h-100">
            <div class="card-body">
                <div class="d-flex justify-content-between align-items-center mb-3">
                    <h5 class="mb-0">${compound.name}</h5>
                    <div class="score-circle ${confidenceClass}">${compound.confidence}%</div>
                </div>
                <p class="text-muted">${compound.type || 'Bioactive compound'}</p>
                <div class="mt-3">
                    <small>Prediction Confidence</small>
                    <div class="progress prediction-progress">
                        <div class="progress-bar ${compound.confidence >= 85 ? 'bg-success' : 'bg-warning'}" 
                             role="progressbar" 
                             style="width: ${compound.confidence}%" 
                             aria-valuenow="${compound.confidence}" 
                             aria-valuemin="0" 
                             aria-valuemax="100"></div>
                    </div>
                </div>
                <div class="mt-3">
                    <div class="d-flex align-items-center mb-2">
                        <small class="me-auto">Bioactivity Score</small>
                        <small>${compound.bioactivity_score}/100</small>
                    </div>
                    <div class="progress prediction-progress">
                        <div class="progress-bar bg-info" 
                             role="progressbar" 
                             style="width: ${compound.bioactivity_score}%" 
                             aria-valuenow="${compound.bioactivity_score}" 
                             aria-valuemin="0" 
                             aria-valuemax="100"></div>
                    </div>
                </div>
                <hr>
                <div class="d-flex justify-content-between">
                    <button class="btn btn-sm btn-outline-primary view-compound-btn" 
                            data-compound-id="${compound.structure_id || ''}">View Details</button>
                    <button class="btn btn-sm btn-outline-secondary">Compare</button>
                </div>
            </div>
        </div>
    `;
    
    // Add event listener to view button
    card.querySelector('.view-compound-btn').addEventListener('click', function() {
        viewCompoundDetails(compound);
    });
    
    return card;
}

/**
 * Update model metrics display
 */
function updateModelMetrics(metrics) {
    if (!metrics) return;
    
    if (metrics.accuracy) {
        const accuracyElement = document.getElementById('model-accuracy');
        if (accuracyElement) {
            accuracyElement.textContent = (metrics.accuracy * 100).toFixed(1) + '%';
        }
    }
    
    if (metrics.precision) {
        const precisionElement = document.getElementById('model-precision');
        if (precisionElement) {
            precisionElement.textContent = metrics.precision.toFixed(2);
        }
    }
    
    if (metrics.recall) {
        const recallElement = document.getElementById('model-recall');
        if (recallElement) {
            recallElement.textContent = metrics.recall.toFixed(2);
        }
    }
    
    if (metrics.f1_score) {
        const f1Element = document.getElementById('model-f1');
        if (f1Element) {
            f1Element.textContent = metrics.f1_score.toFixed(2);
        }
    }
}

/**
 * Display feature importance chart
 */
function displayFeatureImportance(featureImportance) {
    if (!featureImportance) return;
    
    const featuresContainer = document.getElementById('feature-importance');
    if (!featuresContainer) return;
    
    featuresContainer.innerHTML = '';
    
    // Convert object to array of [key, value] pairs and sort by importance
    const features = Object.entries(featureImportance)
        .sort((a, b) => b[1] - a[1]);
    
    // Create feature bars
    features.forEach(([feature, importance]) => {
        const featureBar = document.createElement('div');
        featureBar.className = 'd-flex align-items-center mb-2';
        featureBar.innerHTML = `
            <div style="width: 150px;">${feature}</div>
            <div class="flex-grow-1">
                <div class="progress feature-importance">
                    <div class="progress-bar bg-primary" 
                         role="progressbar" 
                         style="width: ${importance}%" 
                         aria-valuenow="${importance}" 
                         aria-valuemin="0" 
                         aria-valuemax="100"></div>
                </div>
            </div>
            <div style="width: 50px; text-align: right;">${importance}%</div>
        `;
        
        featuresContainer.appendChild(featureBar);
    });
}

/**
 * Display error message
 */
function displayError(error) {
    const resultsContainer = document.getElementById('prediction-results');
    if (!resultsContainer) return;
    
    resultsContainer.classList.remove('d-none');
    resultsContainer.innerHTML = `
        <div class="alert alert-danger">
            <i class="fas fa-exclamation-triangle me-2"></i>
            <strong>Error:</strong> ${error.message || 'An error occurred during prediction. Please try again.'}
        </div>
    `;
}

/**
 * View compound details
 */
function viewCompoundDetails(compound) {
    // In a production environment, this would navigate to a compound details page
    // or open a modal with more information
    
    // For demo purposes, we'll create a simple modal
    
    // Create modal if it doesn't exist
    let modal = document.getElementById('compound-detail-modal');
    if (!modal) {
        modal = document.createElement('div');
        modal.className = 'modal fade';
        modal.id = 'compound-detail-modal';
        modal.setAttribute('tabindex', '-1');
        modal.setAttribute('aria-labelledby', 'compound-detail-modal-label');
        modal.setAttribute('aria-hidden', 'true');
        
        modal.innerHTML = `
            <div class="modal-dialog modal-lg">
                <div class="modal-content">
                    <div class="modal-header">
                        <h5 class="modal-title" id="compound-detail-modal-label">Compound Details</h5>
                        <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
                    </div>
                    <div class="modal-body" id="compound-detail-content">
                        <div class="text-center py-5">
                            <div class="spinner-border text-primary" role="status">
                                <span class="visually-hidden">Loading...</span>
                            </div>
                        </div>
                    </div>
                    <div class="modal-footer">
                        <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Close</button>
                        <a href="#" class="btn btn-primary" id="compound-view-3d-btn">View 3D Structure</a>
                    </div>
                </div>
            </div>
        `;
        
        document.body.appendChild(modal);
    }
    
    // Get content container
    const contentContainer = modal.querySelector('#compound-detail-content');
    
    // Update content
    contentContainer.innerHTML = `
        <div class="row">
            <div class="col-md-6">
                <h3>${compound.name}</h3>
                <p class="text-muted">${compound.type || 'Bioactive compound'}</p>
                
                <div class="mt-4">
                    <h5>Properties</h5>
                    <table class="table">
                        <tbody>
                            <tr>
                                <th>Prediction Confidence</th>
                                <td>${compound.confidence}%</td>
                            </tr>
                            <tr>
                                <th>Bioactivity Score</th>
                                <td>${compound.bioactivity_score}/100</td>
                            </tr>
                            <tr>
                                <th>Structure ID</th>
                                <td>${compound.structure_id || 'Not available'}</td>
                            </tr>
                        </tbody>
                    </table>
                </div>
                
                <div class="mt-4">
                    <h5>Biological Activities</h5>
                    <ul>
                        <li>Neuroprotective properties</li>
                        <li>Anti-inflammatory effects</li>
                        <li>Antioxidant activity</li>
                    </ul>
                    <p class="fst-italic text-muted small">* Based on literature analysis and predictive models</p>
                </div>
            </div>
            <div class="col-md-6">
                <div class="text-center mb-3">
                    <div style="height: 200px; background-color: #f8f9fa; display: flex; align-items: center; justify-content: center;">
                        <i class="fas fa-cube fa-5x text-secondary"></i>
                    </div>
                    <p class="text-muted small mt-2">Molecular structure visualization</p>
                </div>
                
                <div class="mt-4">
                    <h5>Related Research</h5>
                    <div class="list-group list-group-flush">
                        <a href="#" class="list-group-item list-group-item-action">
                            <div class="d-flex w-100 justify-content-between">
                                <h6 class="mb-1">Isolation and characterization of ${compound.name}</h6>
                                <small>2023</small>
                            </div>
                            <p class="mb-1">Journal of Natural Products</p>
                        </a>
                        <a href="#" class="list-group-item list-group-item-action">
                            <div class="d-flex w-100 justify-content-between">
                                <h6 class="mb-1">Bioactivity assessment of ${compound.name}</h6>
                                <small>2022</small>
                            </div>
                            <p class="mb-1">Phytochemistry</p>
                        </a>
                    </div>
                </div>
            </div>
        </div>
    `;
    
    // Update 3D view button
    const viewButton = modal.querySelector('#compound-view-3d-btn');
    viewButton.href = `/molecular-viewer?compound=${compound.structure_id || ''}`;
    
    // Show the modal
    const modalInstance = new bootstrap.Modal(modal);
    modalInstance.show();
}