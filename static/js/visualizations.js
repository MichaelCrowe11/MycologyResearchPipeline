/**
 * Advanced visualizations for the Mycology Research Pipeline
 */

// D3.js-based visualizations for analysis results
class AnalysisVisualizer {
  constructor(containerId) {
    this.container = document.getElementById(containerId);
    if (!this.container) {
      console.error(`Container element with ID ${containerId} not found`);
      return;
    }
    
    // Set dimensions based on container size
    this.width = this.container.clientWidth;
    this.height = 400;
    this.margin = {top: 40, right: 30, bottom: 60, left: 60};
    
    // Initialize the SVG container
    this.svg = d3.select(this.container)
      .append('svg')
      .attr('width', this.width)
      .attr('height', this.height)
      .append('g')
      .attr('transform', `translate(${this.margin.left},${this.margin.top})`);
    
    // Calculate actual dimensions accounting for margins
    this.innerWidth = this.width - this.margin.left - this.margin.right;
    this.innerHeight = this.height - this.margin.top - this.margin.bottom;
  }
  
  // Create a heatmap of compound bioactivity
  createBioactivityHeatmap(data) {
    if (!data || !data.compounds || !data.targets || !data.values) {
      console.error('Invalid data format for bioactivity heatmap');
      return;
    }
    
    // Clear previous content
    this.svg.selectAll('*').remove();
    
    // Create scales
    const xScale = d3.scaleBand()
      .domain(data.targets)
      .range([0, this.innerWidth])
      .padding(0.05);
    
    const yScale = d3.scaleBand()
      .domain(data.compounds)
      .range([0, this.innerHeight])
      .padding(0.05);
    
    // Color scale for bioactivity values (0 to 1)
    const colorScale = d3.scaleSequential()
      .interpolator(d3.interpolateViridis)
      .domain([0, 1]);
    
    // Add X axis
    this.svg.append('g')
      .attr('transform', `translate(0,${this.innerHeight})`)
      .call(d3.axisBottom(xScale))
      .selectAll('text')
      .attr('transform', 'translate(-10,0)rotate(-45)')
      .style('text-anchor', 'end');
    
    // Add Y axis
    this.svg.append('g')
      .call(d3.axisLeft(yScale));
    
    // Create the heatmap cells
    const cells = this.svg.selectAll('rect')
      .data(data.values)
      .enter()
      .append('rect')
      .attr('x', d => xScale(d.target))
      .attr('y', d => yScale(d.compound))
      .attr('width', xScale.bandwidth())
      .attr('height', yScale.bandwidth())
      .style('fill', d => colorScale(d.value))
      .style('stroke', '#fff')
      .style('stroke-width', 0.5);
    
    // Add tooltip functionality
    const tooltip = d3.select('body')
      .append('div')
      .style('position', 'absolute')
      .style('background', 'rgba(0, 0, 0, 0.7)')
      .style('color', 'white')
      .style('padding', '5px')
      .style('border-radius', '5px')
      .style('visibility', 'hidden');
    
    cells.on('mouseover', function(event, d) {
      tooltip.style('visibility', 'visible')
        .html(`Compound: ${d.compound}<br>Target: ${d.target}<br>Bioactivity: ${d.value.toFixed(3)}`)
        .style('top', (event.pageY - 10) + 'px')
        .style('left', (event.pageX + 10) + 'px');
    })
    .on('mouseout', function() {
      tooltip.style('visibility', 'hidden');
    });
    
    // Add title
    this.svg.append('text')
      .attr('x', this.innerWidth / 2)
      .attr('y', -this.margin.top / 2)
      .attr('text-anchor', 'middle')
      .style('font-size', '16px')
      .text('Compound Bioactivity Heatmap');
    
    // Add X axis label
    this.svg.append('text')
      .attr('x', this.innerWidth / 2)
      .attr('y', this.innerHeight + this.margin.bottom - 10)
      .attr('text-anchor', 'middle')
      .text('Targets');
    
    // Add Y axis label
    this.svg.append('text')
      .attr('transform', 'rotate(-90)')
      .attr('x', -this.innerHeight / 2)
      .attr('y', -this.margin.left + 15)
      .attr('text-anchor', 'middle')
      .text('Compounds');
  }
  
  // Create a network graph of compound relationships
  createCompoundNetworkGraph(data) {
    if (!data || !data.nodes || !data.links) {
      console.error('Invalid data format for network graph');
      return;
    }
    
    // Clear previous content
    this.svg.selectAll('*').remove();
    
    // Create a force simulation
    const simulation = d3.forceSimulation(data.nodes)
      .force('link', d3.forceLink(data.links).id(d => d.id).distance(100))
      .force('charge', d3.forceManyBody().strength(-200))
      .force('center', d3.forceCenter(this.innerWidth / 2, this.innerHeight / 2))
      .force('collision', d3.forceCollide().radius(30));
    
    // Add links
    const link = this.svg.append('g')
      .attr('class', 'links')
      .selectAll('line')
      .data(data.links)
      .enter()
      .append('line')
      .attr('stroke-width', d => Math.sqrt(d.value) * 2)
      .attr('stroke', '#999')
      .attr('stroke-opacity', 0.6);
    
    // Define a color scale for node types
    const colorScale = d3.scaleOrdinal()
      .domain(['compound', 'effect', 'property'])
      .range(['#4a8f6e', '#17a2b8', '#ffc107']);
    
    // Add nodes
    const node = this.svg.append('g')
      .attr('class', 'nodes')
      .selectAll('circle')
      .data(data.nodes)
      .enter()
      .append('circle')
      .attr('r', d => d.size || 10)
      .attr('fill', d => colorScale(d.type))
      .attr('stroke', '#fff')
      .attr('stroke-width', 1.5)
      .call(d3.drag()
        .on('start', dragstarted)
        .on('drag', dragged)
        .on('end', dragended));
    
    // Add node labels
    const labels = this.svg.append('g')
      .attr('class', 'labels')
      .selectAll('text')
      .data(data.nodes)
      .enter()
      .append('text')
      .text(d => d.name)
      .attr('font-size', '10px')
      .attr('dx', 12)
      .attr('dy', 4);
    
    // Add tooltip functionality
    const tooltip = d3.select('body')
      .append('div')
      .attr('class', 'tooltip')
      .style('position', 'absolute')
      .style('background', 'rgba(0, 0, 0, 0.7)')
      .style('color', 'white')
      .style('padding', '5px')
      .style('border-radius', '5px')
      .style('visibility', 'hidden');
    
    node.on('mouseover', function(event, d) {
      tooltip.style('visibility', 'visible')
        .html(`Name: ${d.name}<br>Type: ${d.type}<br>${d.description || ''}`)
        .style('top', (event.pageY - 10) + 'px')
        .style('left', (event.pageX + 10) + 'px');
    })
    .on('mouseout', function() {
      tooltip.style('visibility', 'hidden');
    });
    
    // Update positions on simulation tick
    simulation.on('tick', () => {
      link
        .attr('x1', d => d.source.x)
        .attr('y1', d => d.source.y)
        .attr('x2', d => d.target.x)
        .attr('y2', d => d.target.y);
      
      node
        .attr('cx', d => d.x = Math.max(20, Math.min(this.innerWidth - 20, d.x)))
        .attr('cy', d => d.y = Math.max(20, Math.min(this.innerHeight - 20, d.y)));
      
      labels
        .attr('x', d => d.x)
        .attr('y', d => d.y);
    });
    
    // Add title
    this.svg.append('text')
      .attr('x', this.innerWidth / 2)
      .attr('y', -this.margin.top / 2)
      .attr('text-anchor', 'middle')
      .style('font-size', '16px')
      .text('Compound Relationship Network');
    
    // Add legend
    const legend = this.svg.append('g')
      .attr('class', 'legend')
      .attr('transform', `translate(${this.innerWidth - 120}, 20)`);
    
    const legendItems = ['compound', 'effect', 'property'];
    legendItems.forEach((item, i) => {
      const legendItem = legend.append('g')
        .attr('transform', `translate(0, ${i * 20})`);
      
      legendItem.append('circle')
        .attr('r', 6)
        .attr('fill', colorScale(item));
      
      legendItem.append('text')
        .attr('x', 12)
        .attr('y', 4)
        .attr('font-size', '12px')
        .text(item);
    });
    
    // Define drag behavior functions
    function dragstarted(event, d) {
      if (!event.active) simulation.alphaTarget(0.3).restart();
      d.fx = d.x;
      d.fy = d.y;
    }
    
    function dragged(event, d) {
      d.fx = event.x;
      d.fy = event.y;
    }
    
    function dragended(event, d) {
      if (!event.active) simulation.alphaTarget(0);
      d.fx = null;
      d.fy = null;
    }
  }
  
  // Create a scatter plot of bioactivity vs molecular weight
  createBioactivityScatterPlot(data) {
    if (!data || !data.compounds) {
      console.error('Invalid data format for scatter plot');
      return;
    }
    
    // Clear previous content
    this.svg.selectAll('*').remove();
    
    // Create scales
    const xScale = d3.scaleLinear()
      .domain([0, d3.max(data.compounds, d => d.molecular_weight) * 1.1])
      .range([0, this.innerWidth]);
    
    const yScale = d3.scaleLinear()
      .domain([0, 1])
      .range([this.innerHeight, 0]);
    
    // Add X axis
    this.svg.append('g')
      .attr('transform', `translate(0,${this.innerHeight})`)
      .call(d3.axisBottom(xScale));
    
    // Add Y axis
    this.svg.append('g')
      .call(d3.axisLeft(yScale));
    
    // Color scale for different species
    const colorScale = d3.scaleOrdinal()
      .domain([...new Set(data.compounds.map(d => d.species))])
      .range(d3.schemeCategory10);
    
    // Add data points
    const dots = this.svg.append('g')
      .selectAll('circle')
      .data(data.compounds)
      .enter()
      .append('circle')
      .attr('cx', d => xScale(d.molecular_weight))
      .attr('cy', d => yScale(d.bioactivity_index))
      .attr('r', 5)
      .style('fill', d => colorScale(d.species))
      .style('opacity', 0.7)
      .style('stroke', '#fff')
      .style('stroke-width', 0.5);
    
    // Add tooltip functionality
    const tooltip = d3.select('body')
      .append('div')
      .style('position', 'absolute')
      .style('background', 'rgba(0, 0, 0, 0.7)')
      .style('color', 'white')
      .style('padding', '5px')
      .style('border-radius', '5px')
      .style('visibility', 'hidden');
    
    dots.on('mouseover', function(event, d) {
      tooltip.style('visibility', 'visible')
        .html(`Compound: ${d.name}<br>Molecular Weight: ${d.molecular_weight.toFixed(2)}<br>Bioactivity: ${d.bioactivity_index.toFixed(3)}<br>Species: ${d.species}`)
        .style('top', (event.pageY - 10) + 'px')
        .style('left', (event.pageX + 10) + 'px');
      
      d3.select(this)
        .attr('r', 8)
        .style('opacity', 1);
    })
    .on('mouseout', function() {
      tooltip.style('visibility', 'hidden');
      
      d3.select(this)
        .attr('r', 5)
        .style('opacity', 0.7);
    });
    
    // Add title
    this.svg.append('text')
      .attr('x', this.innerWidth / 2)
      .attr('y', -this.margin.top / 2)
      .attr('text-anchor', 'middle')
      .style('font-size', '16px')
      .text('Bioactivity vs Molecular Weight');
    
    // Add X axis label
    this.svg.append('text')
      .attr('x', this.innerWidth / 2)
      .attr('y', this.innerHeight + this.margin.bottom - 10)
      .attr('text-anchor', 'middle')
      .text('Molecular Weight (Da)');
    
    // Add Y axis label
    this.svg.append('text')
      .attr('transform', 'rotate(-90)')
      .attr('x', -this.innerHeight / 2)
      .attr('y', -this.margin.left + 15)
      .attr('text-anchor', 'middle')
      .text('Bioactivity Index');
    
    // Add trend line
    if (data.compounds.length > 1) {
      // Calculate regression line
      const xValues = data.compounds.map(d => d.molecular_weight);
      const yValues = data.compounds.map(d => d.bioactivity_index);
      
      const { slope, intercept } = this.linearRegression(xValues, yValues);
      
      // Create line function
      const line = d3.line()
        .x(d => xScale(d))
        .y(d => yScale(slope * d + intercept));
      
      // Draw line
      const xDomain = xScale.domain();
      const trendLinePoints = [xDomain[0], xDomain[1]];
      
      this.svg.append('path')
        .datum(trendLinePoints)
        .attr('fill', 'none')
        .attr('stroke', '#ff7f0e')
        .attr('stroke-width', 1.5)
        .attr('stroke-dasharray', '5,5')
        .attr('d', line);
    }
    
    // Add legend
    const species = [...new Set(data.compounds.map(d => d.species))];
    const legend = this.svg.append('g')
      .attr('class', 'legend')
      .attr('transform', `translate(${this.innerWidth - 150}, 20)`);
    
    species.forEach((s, i) => {
      const legendItem = legend.append('g')
        .attr('transform', `translate(0, ${i * 20})`);
      
      legendItem.append('circle')
        .attr('r', 6)
        .attr('fill', colorScale(s))
        .style('opacity', 0.7);
      
      legendItem.append('text')
        .attr('x', 12)
        .attr('y', 4)
        .attr('font-size', '12px')
        .text(s);
    });
  }
  
  // Helper function to calculate linear regression
  linearRegression(x, y) {
    const n = x.length;
    let sumX = 0;
    let sumY = 0;
    let sumXY = 0;
    let sumXX = 0;
    
    for (let i = 0; i < n; i++) {
      sumX += x[i];
      sumY += y[i];
      sumXY += x[i] * y[i];
      sumXX += x[i] * x[i];
    }
    
    const slope = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);
    const intercept = (sumY - slope * sumX) / n;
    
    return { slope, intercept };
  }
}

// Create a molecular structure viewer
class MolecularViewer {
  constructor(containerId) {
    this.container = document.getElementById(containerId);
    if (!this.container) {
      console.error(`Container element with ID ${containerId} not found`);
      return;
    }
    
    // Initialize container
    this.container.style.height = '400px';
    this.container.style.width = '100%';
    this.container.style.position = 'relative';
    this.container.style.backgroundColor = '#f8f9fa';
    this.container.style.borderRadius = '10px';
    this.container.style.overflow = 'hidden';
    
    // Create canvas for 3D rendering
    this.canvas = document.createElement('canvas');
    this.canvas.style.width = '100%';
    this.canvas.style.height = '100%';
    this.container.appendChild(this.canvas);
    
    // Create status overlay
    this.statusOverlay = document.createElement('div');
    this.statusOverlay.style.position = 'absolute';
    this.statusOverlay.style.top = '10px';
    this.statusOverlay.style.left = '10px';
    this.statusOverlay.style.padding = '5px 10px';
    this.statusOverlay.style.backgroundColor = 'rgba(0,0,0,0.7)';
    this.statusOverlay.style.color = 'white';
    this.statusOverlay.style.borderRadius = '5px';
    this.statusOverlay.style.fontFamily = 'monospace';
    this.statusOverlay.style.fontSize = '12px';
    this.statusOverlay.style.zIndex = '10';
    this.statusOverlay.textContent = 'Initializing...';
    this.container.appendChild(this.statusOverlay);
    
    // Create loading indicator
    this.loadingIndicator = document.createElement('div');
    this.loadingIndicator.className = 'spinner-border text-primary';
    this.loadingIndicator.style.position = 'absolute';
    this.loadingIndicator.style.top = '50%';
    this.loadingIndicator.style.left = '50%';
    this.loadingIndicator.style.transform = 'translate(-50%, -50%)';
    this.loadingIndicator.style.zIndex = '5';
    this.container.appendChild(this.loadingIndicator);
    
    // Initialize viewer
    this.initViewer();
  }
  
  async initViewer() {
    try {
      // This would load a molecular visualization library in a real application
      // For now, we'll simulate it
      this.statusOverlay.textContent = 'Viewer ready';
      this.loadingIndicator.style.display = 'none';
      
      // Display placeholder info
      const placeholder = document.createElement('div');
      placeholder.style.position = 'absolute';
      placeholder.style.top = '50%';
      placeholder.style.left = '50%';
      placeholder.style.transform = 'translate(-50%, -50%)';
      placeholder.style.textAlign = 'center';
      placeholder.innerHTML = `
        <div>
          <p>Molecular structure viewer ready</p>
          <p>Load a compound to view its structure</p>
        </div>
      `;
      this.container.appendChild(placeholder);
      this.placeholder = placeholder;
      
      // Add control buttons
      this.addControls();
      
    } catch (error) {
      console.error('Error initializing molecular viewer:', error);
      this.statusOverlay.textContent = 'Error initializing viewer';
      this.loadingIndicator.style.display = 'none';
    }
  }
  
  addControls() {
    // Create control panel
    const controls = document.createElement('div');
    controls.style.position = 'absolute';
    controls.style.bottom = '10px';
    controls.style.right = '10px';
    controls.style.zIndex = '10';
    
    // Add buttons
    const rotateButton = document.createElement('button');
    rotateButton.className = 'btn btn-sm btn-primary mr-2';
    rotateButton.textContent = 'Rotate';
    rotateButton.addEventListener('click', () => this.toggleRotation());
    controls.appendChild(rotateButton);
    
    const zoomInButton = document.createElement('button');
    zoomInButton.className = 'btn btn-sm btn-secondary mx-2';
    zoomInButton.textContent = 'Zoom In';
    zoomInButton.addEventListener('click', () => this.zoomIn());
    controls.appendChild(zoomInButton);
    
    const zoomOutButton = document.createElement('button');
    zoomOutButton.className = 'btn btn-sm btn-secondary ml-2';
    zoomOutButton.textContent = 'Zoom Out';
    zoomOutButton.addEventListener('click', () => this.zoomOut());
    controls.appendChild(zoomOutButton);
    
    this.container.appendChild(controls);
  }
  
  loadMolecule(molData) {
    // Simulate loading a molecule
    this.placeholder.style.display = 'none';
    this.loadingIndicator.style.display = 'block';
    
    setTimeout(() => {
      this.loadingIndicator.style.display = 'none';
      this.statusOverlay.textContent = `Loaded: ${molData.name}`;
      
      // Draw placeholder molecule
      this.drawPlaceholderMolecule(molData);
    }, 1000);
  }
  
  drawPlaceholderMolecule(molData) {
    // Create a placeholder for the molecule visualization
    const ctx = this.canvas.getContext('2d');
    const width = this.canvas.width;
    const height = this.canvas.height;
    
    // Clear canvas
    ctx.clearRect(0, 0, width, height);
    
    // Draw placeholder
    ctx.fillStyle = '#f8f9fa';
    ctx.fillRect(0, 0, width, height);
    
    // Draw molecule info
    ctx.fillStyle = '#333';
    ctx.font = '16px Arial';
    ctx.textAlign = 'center';
    ctx.fillText(`Molecule: ${molData.name}`, width/2, height/2 - 30);
    ctx.fillText(`Formula: ${molData.formula}`, width/2, height/2);
    ctx.fillText(`Weight: ${molData.weight}`, width/2, height/2 + 30);
  }
  
  toggleRotation() {
    // Toggle molecule rotation (simulated)
    this.statusOverlay.textContent = 'Rotation toggled';
  }
  
  zoomIn() {
    // Zoom in on molecule (simulated)
    this.statusOverlay.textContent = 'Zooming in';
  }
  
  zoomOut() {
    // Zoom out on molecule (simulated)
    this.statusOverlay.textContent = 'Zooming out';
  }
}

// Initialize visualizations when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
  // Check if we're on a page that needs advanced visualizations
  const bioactivityHeatmapContainer = document.getElementById('bioactivityHeatmap');
  if (bioactivityHeatmapContainer) {
    const visualizer = new AnalysisVisualizer('bioactivityHeatmap');
    
    // Try to get data from the container's data attributes
    try {
      const heatmapData = JSON.parse(bioactivityHeatmapContainer.dataset.heatmapData || '{}');
      if (heatmapData.compounds) {
        visualizer.createBioactivityHeatmap(heatmapData);
      }
    } catch (e) {
      console.error('Error parsing heatmap data:', e);
    }
  }
  
  const networkGraphContainer = document.getElementById('compoundNetworkGraph');
  if (networkGraphContainer) {
    const visualizer = new AnalysisVisualizer('compoundNetworkGraph');
    
    // Try to get data from the container's data attributes
    try {
      const networkData = JSON.parse(networkGraphContainer.dataset.networkData || '{}');
      if (networkData.nodes) {
        visualizer.createCompoundNetworkGraph(networkData);
      }
    } catch (e) {
      console.error('Error parsing network data:', e);
    }
  }
  
  const scatterPlotContainer = document.getElementById('bioactivityScatterPlot');
  if (scatterPlotContainer) {
    const visualizer = new AnalysisVisualizer('bioactivityScatterPlot');
    
    // Try to get data from the container's data attributes
    try {
      const scatterData = JSON.parse(scatterPlotContainer.dataset.scatterData || '{}');
      if (scatterData.compounds) {
        visualizer.createBioactivityScatterPlot(scatterData);
      }
    } catch (e) {
      console.error('Error parsing scatter plot data:', e);
    }
  }
  
  // Initialize molecular viewer if container exists
  const molecularViewerContainer = document.getElementById('molecularViewer');
  if (molecularViewerContainer) {
    const viewer = new MolecularViewer('molecularViewer');
    
    // Check if there's data to load
    try {
      const moleculeData = JSON.parse(molecularViewerContainer.dataset.moleculeData || '{}');
      if (moleculeData.name) {
        viewer.loadMolecule(moleculeData);
      }
    } catch (e) {
      console.error('Error parsing molecule data:', e);
    }
    
    // Store viewer in global scope for interaction
    window.molecularViewer = viewer;
  }
});
