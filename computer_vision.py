"""
Computer Vision Module for Mycology Research Pipeline

This module provides functions for processing and analyzing images of
mushroom specimens using computer vision techniques. It includes:

1. Species identification
2. Feature extraction
3. Growth stage analysis
4. Color analysis
5. Morphological measurements
"""

import os
import cv2
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Optional, Any
import logging
from scientific_databases import ScientificDataIntegrator, search_all_databases

# Setup logging
logger = logging.getLogger(__name__)

# Define constants
SUPPORTED_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
MAX_IMAGE_SIZE = 1024  # Max dimension for processing
MODEL_PATH = os.path.join('models', 'vision')
FEATURE_EXTRACTION_MODEL = os.path.join(MODEL_PATH, 'feature_extractor.pkl')
SPECIES_CLASSIFIER_MODEL = os.path.join(MODEL_PATH, 'species_classifier.pkl')
COLOR_REFERENCE_FILE = os.path.join(MODEL_PATH, 'color_references.json')


class ImageProcessor:
    """Base class for image processing operations."""
    
    def __init__(self):
        """Initialize the image processor."""
        self.image = None
        self.original_image = None
        self.image_path = None
        self.height = 0
        self.width = 0
        self.channels = 0
    
    def load_image(self, image_path: str) -> bool:
        """
        Load an image from the specified path.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            bool: True if the image was loaded successfully, False otherwise
        """
        if not os.path.exists(image_path):
            logger.error(f"Image path does not exist: {image_path}")
            return False
        
        _, ext = os.path.splitext(image_path)
        if ext.lower() not in SUPPORTED_EXTENSIONS:
            logger.error(f"Unsupported image format: {ext}")
            return False
        
        try:
            self.image_path = image_path
            self.original_image = cv2.imread(image_path)
            if self.original_image is None:
                logger.error(f"Failed to load image: {image_path}")
                return False
            
            # Make a copy for processing
            self.image = self.original_image.copy()
            self.height, self.width, self.channels = self.image.shape
            
            # Resize if too large
            if max(self.height, self.width) > MAX_IMAGE_SIZE:
                self._resize_image()
            
            return True
        except Exception as e:
            logger.error(f"Error loading image: {str(e)}")
            return False
    
    def _resize_image(self) -> None:
        """Resize the image to maintain the aspect ratio with max dimension."""
        scale = MAX_IMAGE_SIZE / max(self.height, self.width)
        new_width = int(self.width * scale)
        new_height = int(self.height * scale)
        self.image = cv2.resize(self.image, (new_width, new_height))
        self.height, self.width, self.channels = self.image.shape
    
    def preprocess_image(self) -> np.ndarray:
        """
        Preprocess the image for analysis.
        
        Returns:
            np.ndarray: Preprocessed image
        """
        if self.image is None:
            logger.error("No image loaded")
            return None
        
        try:
            # Convert to RGB (OpenCV uses BGR)
            rgb_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            
            # Apply basic preprocessing
            # - Normalize pixel values
            normalized = rgb_image / 255.0
            
            return normalized
        except Exception as e:
            logger.error(f"Error preprocessing image: {str(e)}")
            return None
    
    def save_processed_image(self, output_path: str) -> bool:
        """
        Save the current processed image.
        
        Args:
            output_path: Path to save the processed image
            
        Returns:
            bool: True if the image was saved successfully, False otherwise
        """
        if self.image is None:
            logger.error("No image to save")
            return False
        
        try:
            directory = os.path.dirname(output_path)
            if not os.path.exists(directory):
                os.makedirs(directory)
            
            result = cv2.imwrite(output_path, self.image)
            if not result:
                logger.error(f"Failed to save image: {output_path}")
                return False
            
            return True
        except Exception as e:
            logger.error(f"Error saving image: {str(e)}")
            return False


class SpeciesIdentifier(ImageProcessor):
    """Class for identifying mushroom species from images."""
    
    def __init__(self):
        """Initialize the species identifier."""
        super().__init__()
        self.model = None
        self.species_names = []
        self.confidence_threshold = 0.7
    
    def load_model(self) -> bool:
        """
        Load the species identification model.
        
        Returns:
            bool: True if the model was loaded successfully, False otherwise
        """
        try:
            # Mock model loading for now
            # In a real implementation, this would load a trained model
            # self.model = joblib.load(SPECIES_CLASSIFIER_MODEL)
            self.model = True  # Mock model loaded
            
            # Sample species list
            self.species_names = [
                "Hericium erinaceus",       # Lion's Mane
                "Ganoderma lucidum",        # Reishi
                "Trametes versicolor",      # Turkey Tail
                "Lentinula edodes",         # Shiitake
                "Pleurotus ostreatus",      # Oyster
                "Grifola frondosa",         # Maitake
                "Cordyceps militaris",      # Cordyceps
                "Inonotus obliquus",        # Chaga
                "Agaricus bisporus",        # Button Mushroom
                "Flammulina velutipes",     # Enoki
            ]
            
            logger.info("Species identification model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Error loading species model: {str(e)}")
            return False
    
    def identify_species(self) -> Dict[str, Any]:
        """
        Identify the mushroom species in the image.
        
        Returns:
            Dict: Results containing species and confidence scores
        """
        if self.image is None:
            logger.error("No image loaded")
            return {"error": "No image loaded"}
        
        if self.model is None:
            success = self.load_model()
            if not success:
                return {"error": "Failed to load species identification model"}
        
        try:
            # Preprocess the image
            preprocessed = self.preprocess_image()
            if preprocessed is None:
                return {"error": "Failed to preprocess image"}
            
            # In a real implementation, this would use the model to predict
            # For demonstration purposes, we'll return mock results
            
            # Mock prediction logic
            # For demo purposes, use a simple heuristic based on image properties
            avg_color = np.mean(self.image, axis=(0, 1))
            hue_value = (avg_color[0] + avg_color[1] * 2 + avg_color[2]) / 4
            
            # Use hue to select from species list
            species_index = int(hue_value) % len(self.species_names)
            primary_species = self.species_names[species_index]
            
            # Generate some alternative species with lower confidence
            secondary_species = []
            for i in range(3):
                alt_index = (species_index + i + 1) % len(self.species_names)
                confidence = max(0.3, 0.9 - (i * 0.2))
                secondary_species.append({
                    "name": self.species_names[alt_index],
                    "confidence": round(confidence, 2)
                })
            
            return {
                "primary_species": {
                    "name": primary_species,
                    "confidence": round(0.85 + (np.random.random() * 0.1), 2)
                },
                "alternatives": secondary_species,
                "analysis_time": round(np.random.random() * 2 + 0.5, 2),  # Mock analysis time
                "image_quality": self._assess_image_quality()
            }
        except Exception as e:
            logger.error(f"Error identifying species: {str(e)}")
            return {"error": f"Error identifying species: {str(e)}"}
    
    def _assess_image_quality(self) -> Dict[str, Any]:
        """
        Assess the quality of the image for species identification.
        
        Returns:
            Dict: Image quality metrics
        """
        if self.image is None:
            return {"score": 0, "issues": ["No image loaded"]}
        
        issues = []
        quality_score = 1.0
        
        # Check brightness
        brightness = np.mean(self.image)
        if brightness < 40:
            issues.append("Image is too dark")
            quality_score -= 0.2
        elif brightness > 215:
            issues.append("Image is too bright")
            quality_score -= 0.2
        
        # Check contrast
        contrast = np.std(self.image)
        if contrast < 20:
            issues.append("Low contrast")
            quality_score -= 0.15
        
        # Check blur
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if lap_var < 50:
            issues.append("Image appears blurry")
            quality_score -= 0.3
        
        # Check resolution
        if self.width * self.height < 250000:  # Less than ~500x500
            issues.append("Low resolution")
            quality_score -= 0.2
        
        # Ensure quality score is in valid range
        quality_score = max(0, min(1, quality_score))
        
        return {
            "score": round(quality_score, 2),
            "issues": issues
        }


class MorphologicalAnalyzer(ImageProcessor):
    """Class for analyzing morphological features of mushrooms."""
    
    def __init__(self):
        """Initialize the morphological analyzer."""
        super().__init__()
        self.contours = None
        self.mask = None
    
    def segment_mushroom(self) -> bool:
        """
        Segment the mushroom from the background.
        
        Returns:
            bool: True if segmentation was successful, False otherwise
        """
        if self.image is None:
            logger.error("No image loaded")
            return False
        
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Apply Otsu's thresholding
            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                logger.warning("No contours found in the image")
                return False
            
            # Find the largest contour (assumed to be the mushroom)
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Create a mask for the mushroom
            self.mask = np.zeros_like(gray)
            cv2.drawContours(self.mask, [largest_contour], 0, 255, -1)
            
            # Store the contour for future use
            self.contours = [largest_contour]
            
            return True
        except Exception as e:
            logger.error(f"Error segmenting mushroom: {str(e)}")
            return False
    
    def measure_features(self) -> Dict[str, Any]:
        """
        Measure morphological features of the mushroom.
        
        Returns:
            Dict: Morphological measurements and features
        """
        if self.image is None:
            logger.error("No image loaded")
            return {"error": "No image loaded"}
        
        if self.contours is None:
            success = self.segment_mushroom()
            if not success:
                return {"error": "Failed to segment mushroom from background"}
        
        try:
            # Get the largest contour
            contour = self.contours[0]
            
            # Calculate basic properties
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            
            # Calculate shape descriptors
            circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
            
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h if h > 0 else 0
            
            # Get minimum enclosing circle
            (center_x, center_y), radius = cv2.minEnclosingCircle(contour)
            
            # Get rotated bounding rectangle
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int32(box)
            rect_width, rect_height = rect[1]
            
            # Calculate elongation
            elongation = min(rect_width, rect_height) / max(rect_width, rect_height) if max(rect_width, rect_height) > 0 else 0
            
            # Calculate convexity
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            convexity = area / hull_area if hull_area > 0 else 0
            
            # Draw results on image (for visualization)
            result_image = self.image.copy()
            cv2.drawContours(result_image, [contour], 0, (0, 255, 0), 2)
            cv2.drawContours(result_image, [box], 0, (255, 0, 0), 2)
            cv2.circle(result_image, (int(center_x), int(center_y)), int(radius), (0, 0, 255), 2)
            
            # Save the result image
            self.image = result_image
            
            return {
                "measurements": {
                    "area": round(area, 2),
                    "perimeter": round(perimeter, 2),
                    "width": round(w, 2),
                    "height": round(h, 2),
                    "radius": round(radius, 2)
                },
                "shape_features": {
                    "circularity": round(circularity, 3),
                    "aspect_ratio": round(aspect_ratio, 3),
                    "elongation": round(elongation, 3),
                    "convexity": round(convexity, 3)
                },
                "position": {
                    "center_x": round(center_x, 2),
                    "center_y": round(center_y, 2),
                    "bounding_box": {
                        "x": x,
                        "y": y,
                        "width": w,
                        "height": h
                    }
                }
            }
        except Exception as e:
            logger.error(f"Error measuring features: {str(e)}")
            return {"error": f"Error measuring features: {str(e)}"}


class ColorAnalyzer(ImageProcessor):
    """Class for analyzing color properties of mushrooms."""
    
    def __init__(self):
        """Initialize the color analyzer."""
        super().__init__()
        self.mask = None
        self.color_spaces = {
            'rgb': cv2.COLOR_BGR2RGB,
            'hsv': cv2.COLOR_BGR2HSV,
            'lab': cv2.COLOR_BGR2LAB
        }
    
    def set_mask(self, mask: np.ndarray) -> None:
        """
        Set the mask for color analysis.
        
        Args:
            mask: Binary mask indicating the region to analyze
        """
        self.mask = mask
    
    def analyze_colors(self) -> Dict[str, Any]:
        """
        Analyze color properties of the mushroom.
        
        Returns:
            Dict: Color analysis results
        """
        if self.image is None:
            logger.error("No image loaded")
            return {"error": "No image loaded"}
        
        if self.mask is None:
            # If no mask provided, create a simple one with the entire image
            self.mask = np.ones((self.height, self.width), dtype=np.uint8) * 255
        
        try:
            # Convert to various color spaces
            rgb_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            hsv_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
            lab_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2LAB)
            
            # Apply mask to all color spaces
            rgb_masked = cv2.bitwise_and(rgb_image, 
                                          cv2.cvtColor(self.mask, cv2.COLOR_GRAY2RGB))
            hsv_masked = cv2.bitwise_and(hsv_image, 
                                          cv2.cvtColor(self.mask, cv2.COLOR_GRAY2RGB))
            lab_masked = cv2.bitwise_and(lab_image, 
                                          cv2.cvtColor(self.mask, cv2.COLOR_GRAY2RGB))
            
            # Calculate color statistics
            rgb_mean = cv2.mean(rgb_masked, mask=self.mask)[:3]
            hsv_mean = cv2.mean(hsv_masked, mask=self.mask)[:3]
            lab_mean = cv2.mean(lab_masked, mask=self.mask)[:3]
            
            # Calculate color histograms
            rgb_hist = self._calculate_color_histogram(rgb_masked, 'rgb')
            hsv_hist = self._calculate_color_histogram(hsv_masked, 'hsv')
            
            # Determine dominant colors
            dominant_colors = self._extract_dominant_colors(rgb_masked, n_colors=5)
            
            # Create color visualization
            color_vis = self._create_color_visualization(dominant_colors)
            
            return {
                "means": {
                    "rgb": {
                        "r": round(rgb_mean[0], 2),
                        "g": round(rgb_mean[1], 2),
                        "b": round(rgb_mean[2], 2)
                    },
                    "hsv": {
                        "h": round(hsv_mean[0], 2),
                        "s": round(hsv_mean[1], 2),
                        "v": round(hsv_mean[2], 2)
                    },
                    "lab": {
                        "l": round(lab_mean[0], 2),
                        "a": round(lab_mean[1], 2),
                        "b": round(lab_mean[2], 2)
                    }
                },
                "dominant_colors": [
                    {
                        "rgb": color.tolist(),
                        "percentage": round(percentage * 100, 2)
                    } for color, percentage in dominant_colors
                ],
                "histograms": {
                    "rgb": rgb_hist,
                    "hsv": hsv_hist
                }
            }
        except Exception as e:
            logger.error(f"Error analyzing colors: {str(e)}")
            return {"error": f"Error analyzing colors: {str(e)}"}
    
    def _calculate_color_histogram(self, image: np.ndarray, color_space: str) -> Dict[str, List[int]]:
        """
        Calculate color histogram for the given image.
        
        Args:
            image: Image to calculate histogram for
            color_space: Color space of the image ('rgb' or 'hsv')
            
        Returns:
            Dict: Histogram values for each channel
        """
        channels = [0, 1, 2]
        hist_size = [32, 32, 32]
        
        if color_space == 'rgb':
            ranges = [0, 256, 0, 256, 0, 256]
            channel_names = ['r', 'g', 'b']
        elif color_space == 'hsv':
            ranges = [0, 180, 0, 256, 0, 256]
            channel_names = ['h', 's', 'v']
        else:
            ranges = [0, 256, 0, 256, 0, 256]
            channel_names = ['c1', 'c2', 'c3']
        
        histograms = {}
        
        for i, channel in enumerate(channels):
            hist = cv2.calcHist([image], [channel], self.mask, [hist_size[i]], 
                                [ranges[i*2], ranges[i*2+1]])
            hist = cv2.normalize(hist, hist).flatten().tolist()
            histograms[channel_names[i]] = hist
        
        return histograms
    
    def _extract_dominant_colors(self, image: np.ndarray, n_colors: int = 5) -> List[Tuple[np.ndarray, float]]:
        """
        Extract dominant colors from an image.
        
        Args:
            image: Image to extract colors from
            n_colors: Number of dominant colors to extract
            
        Returns:
            List: List of tuples (color, percentage)
        """
        # Reshape the image to be a list of pixels
        pixels = image.reshape(-1, 3)
        
        # Remove black pixels (background)
        pixels = pixels[np.all(pixels > 5, axis=1)]
        
        if len(pixels) == 0:
            return []
        
        # Use K-means clustering to find dominant colors
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.1)
        _, labels, centers = cv2.kmeans(pixels.astype(np.float32), n_colors, None, 
                                        criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # Count percentage of pixels in each cluster
        count = np.bincount(labels.flatten())
        percentage = count / len(pixels)
        
        # Sort by percentage
        indices = np.argsort(percentage)[::-1]
        centers = centers[indices]
        percentage = percentage[indices]
        
        # Return colors and their percentages
        return [(centers[i], percentage[i]) for i in range(min(n_colors, len(centers)))]
    
    def _create_color_visualization(self, colors: List[Tuple[np.ndarray, float]]) -> np.ndarray:
        """
        Create a visualization image of dominant colors.
        
        Args:
            colors: List of tuples (color, percentage)
            
        Returns:
            np.ndarray: Visualization image
        """
        # Create a 100x300 image to show colors
        vis = np.zeros((100, 300, 3), dtype=np.uint8)
        
        if not colors:
            return vis
        
        # Calculate widths based on percentages
        start_x = 0
        for color, percentage in colors:
            width = int(300 * percentage)
            if width <= 0:
                continue
            
            end_x = min(start_x + width, 300)
            
            # Fill rectangle with the color
            vis[:, start_x:end_x] = color
            
            start_x = end_x
            if start_x >= 300:
                break
        
        return vis


class GrowthStageAnalyzer(ImageProcessor):
    """Class for analyzing growth stage of mushrooms."""
    
    def __init__(self):
        """Initialize the growth stage analyzer."""
        super().__init__()
    
    def analyze_growth_stage(self) -> Dict[str, Any]:
        """
        Analyze the growth stage of the mushroom.
        
        Returns:
            Dict: Growth stage analysis results
        """
        if self.image is None:
            logger.error("No image loaded")
            return {"error": "No image loaded"}
        
        try:
            # In a real implementation, this would use trained models for growth stage analysis
            # For demonstration purposes, we'll provide mock results
            
            # Extract basic features
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            
            # Compute average brightness as a simple feature
            avg_brightness = np.mean(gray)
            
            # Compute simple texture features
            texture_entropy = self._compute_entropy(gray)
            
            # Mock growth stage estimation based on simple image features
            if texture_entropy > 5.0:
                primary_stage = "mature"
                stage_confidence = 0.85
            elif texture_entropy > 4.0:
                primary_stage = "mid-growth"
                stage_confidence = 0.75
            else:
                primary_stage = "early-growth"
                stage_confidence = 0.80
            
            # Mock growth progress
            growth_progress = min(1.0, max(0.0, texture_entropy / 6.0))
            
            # Create mock stage estimates
            stages = [
                {"name": "early-growth", "probability": 0.0},
                {"name": "mid-growth", "probability": 0.0},
                {"name": "mature", "probability": 0.0},
                {"name": "late-stage", "probability": 0.0}
            ]
            
            # Set probabilities based on primary stage
            for stage in stages:
                if stage["name"] == primary_stage:
                    stage["probability"] = stage_confidence
                elif stage["name"] == "mid-growth" and primary_stage == "early-growth":
                    stage["probability"] = (1.0 - stage_confidence) * 0.8
                elif stage["name"] == "mature" and primary_stage == "mid-growth":
                    stage["probability"] = (1.0 - stage_confidence) * 0.8
                elif stage["name"] == "early-growth" and primary_stage == "mid-growth":
                    stage["probability"] = (1.0 - stage_confidence) * 0.6
                elif stage["name"] == "late-stage" and primary_stage == "mature":
                    stage["probability"] = (1.0 - stage_confidence) * 0.7
                else:
                    stage["probability"] = (1.0 - stage_confidence) * 0.2
            
            # Ensure probabilities sum to 1.0
            total_prob = sum(s["probability"] for s in stages)
            if total_prob > 0:
                for stage in stages:
                    stage["probability"] = round(stage["probability"] / total_prob, 2)
            
            # Sort stages by probability (descending)
            stages.sort(key=lambda x: x["probability"], reverse=True)
            
            return {
                "primary_stage": primary_stage,
                "confidence": round(stage_confidence, 2),
                "growth_progress": round(growth_progress, 2),
                "estimated_days_to_harvest": self._estimate_days_to_harvest(primary_stage, growth_progress),
                "all_stages": stages,
                "features": {
                    "texture_entropy": round(texture_entropy, 2),
                    "avg_brightness": round(avg_brightness, 2)
                }
            }
        except Exception as e:
            logger.error(f"Error analyzing growth stage: {str(e)}")
            return {"error": f"Error analyzing growth stage: {str(e)}"}
    
    def _compute_entropy(self, image: np.ndarray) -> float:
        """
        Compute entropy of an image as a texture feature.
        
        Args:
            image: Grayscale image
            
        Returns:
            float: Entropy value
        """
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        hist = hist / hist.sum()
        hist = hist[hist > 0]
        return -np.sum(hist * np.log2(hist))
    
    def _estimate_days_to_harvest(self, stage: str, progress: float) -> Optional[int]:
        """
        Estimate days to harvest based on growth stage.
        
        Args:
            stage: Growth stage
            progress: Growth progress (0.0 to 1.0)
            
        Returns:
            int: Estimated days to harvest, or None if already harvested
        """
        # Mock estimates based on stage and progress
        if stage == "early-growth":
            return 10 - int(progress * 5)
        elif stage == "mid-growth":
            return 5 - int(progress * 3)
        elif stage == "mature":
            return 1 if progress < 0.8 else 0
        else:  # late-stage
            return 0


def process_sample_image(
    image_path: str,
    output_dir: str = None,
    analyze_species: bool = True,
    analyze_morphology: bool = True,
    analyze_color: bool = True,
    analyze_growth: bool = True
) -> Dict[str, Any]:
    """
    Process a sample image with multiple analysis types.
    
    Args:
        image_path: Path to the image file
        output_dir: Directory to save output files
        analyze_species: Whether to perform species identification
        analyze_morphology: Whether to perform morphological analysis
        analyze_color: Whether to perform color analysis
        analyze_growth: Whether to perform growth stage analysis
        
    Returns:
        Dict: Combined analysis results
    """
    results = {
        "image_path": image_path,
        "timestamp": pd.Timestamp.now().isoformat(),
        "success": False
    }
    
    # Create output directory if needed
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Initialize analysis results
    species_results = None
    morphology_results = None
    color_results = None
    growth_results = None
    
    try:
        # Species identification
        if analyze_species:
            identifier = SpeciesIdentifier()
            if identifier.load_image(image_path):
                species_results = identifier.identify_species()
                
                # Save visualized species result if output directory provided
                if output_dir:
                    species_output = os.path.join(output_dir, "species_result.jpg")
                    identifier.save_processed_image(species_output)
                    results["species_output_image"] = species_output
        
        # Morphological analysis
        if analyze_morphology:
            morpho_analyzer = MorphologicalAnalyzer()
            if morpho_analyzer.load_image(image_path):
                success = morpho_analyzer.segment_mushroom()
                if success:
                    morphology_results = morpho_analyzer.measure_features()
                    
                    # Save morphology visualization if output directory provided
                    if output_dir:
                        morpho_output = os.path.join(output_dir, "morphology_result.jpg")
                        morpho_analyzer.save_processed_image(morpho_output)
                        results["morphology_output_image"] = morpho_output
                    
                    # Use the mask for color analysis
                    if analyze_color:
                        color_analyzer = ColorAnalyzer()
                        if color_analyzer.load_image(image_path):
                            color_analyzer.set_mask(morpho_analyzer.mask)
                            color_results = color_analyzer.analyze_colors()
        
        # Color analysis (if not already done with morphology)
        if analyze_color and color_results is None:
            color_analyzer = ColorAnalyzer()
            if color_analyzer.load_image(image_path):
                color_results = color_analyzer.analyze_colors()
        
        # Growth stage analysis
        if analyze_growth:
            growth_analyzer = GrowthStageAnalyzer()
            if growth_analyzer.load_image(image_path):
                growth_results = growth_analyzer.analyze_growth_stage()
        
        # Combine all results
        if species_results and "error" not in species_results:
            results["species"] = species_results
        
        if morphology_results and "error" not in morphology_results:
            results["morphology"] = morphology_results
        
        if color_results and "error" not in color_results:
            results["color"] = color_results
        
        if growth_results and "error" not in growth_results:
            results["growth_stage"] = growth_results
        
        results["success"] = True
        
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        results["error"] = f"Error processing image: {str(e)}"
    
    return results


def save_analysis_results(results: Dict[str, Any], output_path: str, format: str = "json") -> bool:
    """
    Save analysis results to a file.
    
    Args:
        results: Analysis results
        output_path: Path to save the results
        format: Format to save the results in ('json' or 'csv')
        
    Returns:
        bool: True if the results were saved successfully, False otherwise
    """
    try:
        directory = os.path.dirname(output_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        
        if format.lower() == "json":
            with open(output_path, 'w') as f:
                import json
                json.dump(results, f, indent=2)
        elif format.lower() == "csv":
            # For CSV, flatten the dictionary
            flat_dict = {}
            
            def flatten(obj, prefix=""):
                if isinstance(obj, dict):
                    for key, value in obj.items():
                        flatten(value, f"{prefix}_{key}" if prefix else key)
                elif isinstance(obj, list):
                    for i, item in enumerate(obj):
                        flatten(item, f"{prefix}_{i}")
                else:
                    flat_dict[prefix] = obj
            
            flatten(results)
            
            df = pd.DataFrame([flat_dict])
            df.to_csv(output_path, index=False)
        else:
            logger.error(f"Unsupported format: {format}")
            return False
        
        return True
    except Exception as e:
        logger.error(f"Error saving results: {str(e)}")
        return False


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Example usage
    test_image = "path/to/test_image.jpg"
    output_dir = "results/vision_analysis"
    
    if os.path.exists(test_image):
        results = process_sample_image(test_image, output_dir)
        print(f"Analysis completed successfully: {results['success']}")
        
        if results["success"]:
            # Save results
            save_analysis_results(results, os.path.join(output_dir, "analysis_results.json"))
    else:
        print(f"Test image not found: {test_image}")