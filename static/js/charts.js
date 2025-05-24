/**
 * Charts and visualization utilities for the Mycology Research Pipeline
 */

// Initialize all charts on the dashboard
function initDashboardCharts() {
  // Sample Distribution Chart
  if (document.getElementById('sampleDistributionChart')) {
    createSampleDistributionChart();
  }
  
  // Bioactivity Histogram
  if (document.getElementById('bioactivityHistogram')) {
    createBioactivityHistogram();
  }
  
  // Analysis Success Rate Chart
  if (document.getElementById('analysisSuccessChart')) {
    createAnalysisSuccessChart();
  }
  
  // Feature Importance Chart
  if (document.getElementById('featureImportanceChart')) {
    createFeatureImportanceChart();
  }
}

// Create a doughnut chart showing sample distribution by species
function createSampleDistributionChart() {
  // Get the chart canvas element
  const ctx = document.getElementById('sampleDistributionChart').getContext('2d');
  
  // Sample data (in production this would come from the server)
  const sampleData = {
    species: [
      'Agaricus bisporus',
      'Lentinula edodes',
      'Ganoderma lucidum',
      'Cordyceps militaris',
      'Other'
    ],
    counts: [15, 12, 9, 7, 5]
  };
  
  // Create the chart
  new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels: sampleData.species,
      datasets: [{
        data: sampleData.counts,
        backgroundColor: [
          '#4a8f6e',
          '#6c757d',
          '#17a2b8',
          '#ffc107',
          '#dc3545'
        ],
        borderWidth: 1
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      legend: {
        position: 'right',
        labels: {
          fontSize: 12,
          boxWidth: 15
        }
      },
      title: {
        display: true,
        text: 'Sample Distribution by Species',
        fontSize: 16
      },
      animation: {
        animateScale: true
      },
      tooltips: {
        callbacks: {
          label: function(tooltipItem, data) {
            const dataset = data.datasets[tooltipItem.datasetIndex];
            const total = dataset.data.reduce((acc, val) => acc + val, 0);
            const value = dataset.data[tooltipItem.index];
            const percentage = Math.round((value / total) * 100);
            return `${data.labels[tooltipItem.index]}: ${value} (${percentage}%)`;
          }
        }
      }
    }
  });
}

// Create a histogram of bioactivity scores
function createBioactivityHistogram() {
  const ctx = document.getElementById('bioactivityHistogram').getContext('2d');
  
  // Sample data (in production this would come from the server)
  const bioactivityData = {
    ranges: ['0.0 - 0.2', '0.2 - 0.4', '0.4 - 0.6', '0.6 - 0.8', '0.8 - 1.0'],
    counts: [7, 12, 25, 18, 8]
  };
  
  // Create the chart
  new Chart(ctx, {
    type: 'bar',
    data: {
      labels: bioactivityData.ranges,
      datasets: [{
        label: 'Number of Compounds',
        data: bioactivityData.counts,
        backgroundColor: 'rgba(74, 143, 110, 0.7)',
        borderColor: 'rgba(74, 143, 110, 1)',
        borderWidth: 1
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        xAxes: [{
          scaleLabel: {
            display: true,
            labelString: 'Bioactivity Score Range'
          }
        }],
        yAxes: [{
          ticks: {
            beginAtZero: true
          },
          scaleLabel: {
            display: true,
            labelString: 'Count'
          }
        }]
      },
      title: {
        display: true,
        text: 'Distribution of Bioactivity Scores',
        fontSize: 16
      }
    }
  });
}

// Create a pie chart showing analysis success rates
function createAnalysisSuccessChart() {
  const ctx = document.getElementById('analysisSuccessChart').getContext('2d');
  
  // Get success rate from the data attribute
  const successRate = parseFloat(document.getElementById('analysisSuccessChart').dataset.successRate || 0);
  const failureRate = 100 - successRate;
  
  // Create the chart
  new Chart(ctx, {
    type: 'pie',
    data: {
      labels: ['Successful', 'Failed'],
      datasets: [{
        data: [successRate, failureRate],
        backgroundColor: [
          'rgba(40, 167, 69, 0.7)',
          'rgba(220, 53, 69, 0.7)'
        ],
        borderColor: [
          'rgba(40, 167, 69, 1)',
          'rgba(220, 53, 69, 1)'
        ],
        borderWidth: 1
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      title: {
        display: true,
        text: 'Analysis Success Rate',
        fontSize: 16
      },
      tooltips: {
        callbacks: {
          label: function(tooltipItem, data) {
            const value = data.datasets[0].data[tooltipItem.index];
            return `${data.labels[tooltipItem.index]}: ${value.toFixed(1)}%`;
          }
        }
      }
    }
  });
}

// Create a horizontal bar chart showing feature importance
function createFeatureImportanceChart() {
  const ctx = document.getElementById('featureImportanceChart').getContext('2d');
  
  // Sample data (in production this would come from the server)
  const featureData = {
    features: [
      'Molecular Weight',
      'pH Level',
      'Temperature',
      'Concentration',
      'Incubation Time'
    ],
    importance: [0.35, 0.25, 0.20, 0.15, 0.05]
  };
  
  // Create the chart
  new Chart(ctx, {
    type: 'horizontalBar',
    data: {
      labels: featureData.features,
      datasets: [{
        label: 'Importance',
        data: featureData.importance,
        backgroundColor: 'rgba(23, 162, 184, 0.7)',
        borderColor: 'rgba(23, 162, 184, 1)',
        borderWidth: 1
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        xAxes: [{
          ticks: {
            beginAtZero: true,
            max: 0.5
          },
          scaleLabel: {
            display: true,
            labelString: 'Importance Score'
          }
        }],
        yAxes: [{
          scaleLabel: {
            display: true,
            labelString: 'Feature'
          }
        }]
      },
      title: {
        display: true,
        text: 'Feature Importance',
        fontSize: 16
      }
    }
  });
}

// Create a visualization of analysis results
function createAnalysisResultVisualization(resultsData, canvasId) {
  if (!resultsData || !canvasId || !document.getElementById(canvasId)) {
    console.error('Missing data or canvas element for visualization');
    return;
  }
  
  const ctx = document.getElementById(canvasId).getContext('2d');
  
  // Determine chart type based on results
  if (resultsData.hasOwnProperty('bioactivity_scores')) {
    // Regressor results - scatter plot with error bars
    createRegressionResultsChart(ctx, resultsData);
  } else if (resultsData.hasOwnProperty('categories')) {
    // Classifier results - bar chart of categories
    createClassificationResultsChart(ctx, resultsData);
  } else {
    console.warn('Unknown result type for visualization');
  }
}

// Create a scatter plot for regression results
function createRegressionResultsChart(ctx, resultsData) {
  // Extract data from results
  const scores = resultsData.bioactivity_scores;
  const lowerCI = resultsData.confidence_intervals.map(ci => ci[0]);
  const upperCI = resultsData.confidence_intervals.map(ci => ci[1]);
  
  // Create labels for X axis
  const labels = Array.from({length: scores.length}, (_, i) => `Sample ${i+1}`);
  
  // Prepare error bar dataset
  const errorBarData = [];
  for (let i = 0; i < scores.length; i++) {
    errorBarData.push({
      x: i,
      y: scores[i],
      yMin: lowerCI[i],
      yMax: upperCI[i]
    });
  }
  
  // Create the chart
  new Chart(ctx, {
    type: 'scatter',
    data: {
      datasets: [{
        label: 'Bioactivity Score',
        data: errorBarData,
        backgroundColor: 'rgba(74, 143, 110, 0.7)',
        borderColor: 'rgba(74, 143, 110, 1)',
        pointRadius: 5,
        pointHoverRadius: 7
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      title: {
        display: true,
        text: 'Bioactivity Analysis Results',
        fontSize: 16
      },
      scales: {
        xAxes: [{
          type: 'linear',
          position: 'bottom',
          ticks: {
            min: -0.5,
            max: scores.length - 0.5,
            stepSize: 1,
            callback: function(value) {
              if (Number.isInteger(value) && value >= 0 && value < labels.length) {
                return labels[value];
              }
              return null;
            }
          }
        }],
        yAxes: [{
          ticks: {
            min: 0,
            max: 1.0
          },
          scaleLabel: {
            display: true,
            labelString: 'Bioactivity Score'
          }
        }]
      },
      tooltips: {
        callbacks: {
          title: function(tooltipItems) {
            const index = tooltipItems[0].index;
            return labels[index];
          },
          label: function(tooltipItem) {
            const index = tooltipItem.index;
            return [
              `Score: ${scores[index].toFixed(3)}`,
              `CI: [${lowerCI[index].toFixed(3)}, ${upperCI[index].toFixed(3)}]`
            ];
          }
        }
      }
    }
  });
}

// Create a bar chart for classification results
function createClassificationResultsChart(ctx, resultsData) {
  // Extract data from results
  const categories = resultsData.categories;
  const probabilities = resultsData.probabilities;
  
  // Create labels for X axis
  const labels = Array.from({length: categories.length}, (_, i) => `Sample ${i+1}`);
  
  // Color mapping for categories
  const colorMap = {
    'active': 'rgba(40, 167, 69, 0.7)',
    'moderate': 'rgba(255, 193, 7, 0.7)',
    'inactive': 'rgba(220, 53, 69, 0.7)'
  };
  
  // Create dataset
  const backgroundColor = categories.map(cat => colorMap[cat] || 'rgba(108, 117, 125, 0.7)');
  
  // Create the chart
  new Chart(ctx, {
    type: 'bar',
    data: {
      labels: labels,
      datasets: [{
        label: 'Classification Probability',
        data: probabilities,
        backgroundColor: backgroundColor,
        borderColor: backgroundColor.map(color => color.replace('0.7', '1')),
        borderWidth: 1
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      title: {
        display: true,
        text: 'Compound Classification Results',
        fontSize: 16
      },
      scales: {
        xAxes: [{
          scaleLabel: {
            display: true,
            labelString: 'Sample'
          }
        }],
        yAxes: [{
          ticks: {
            min: 0,
            max: 1.0
          },
          scaleLabel: {
            display: true,
            labelString: 'Probability'
          }
        }]
      },
      tooltips: {
        callbacks: {
          label: function(tooltipItem, data) {
            const index = tooltipItem.index;
            const value = tooltipItem.value;
            return [
              `Category: ${categories[index]}`,
              `Probability: ${parseFloat(value).toFixed(3)}`
            ];
          }
        }
      }
    }
  });
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
  initDashboardCharts();
  
  // If we have result data in a data attribute, visualize it
  const resultElements = document.querySelectorAll('[data-results]');
  resultElements.forEach(element => {
    try {
      const resultsData = JSON.parse(element.dataset.results);
      const canvasId = element.dataset.canvas;
      if (resultsData && canvasId) {
        createAnalysisResultVisualization(resultsData, canvasId);
      }
    } catch (e) {
      console.error('Error parsing results data:', e);
    }
  });
});
