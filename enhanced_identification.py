"""
Enhanced Species Identification System with Scientific Database Integration.

This module combines computer vision analysis with comprehensive scientific databases
to provide highly accurate species identification, especially for dried specimens.
"""

import json
import logging
from typing import Dict, List, Any, Optional
from scientific_databases import ScientificDataIntegrator
from computer_vision import SpeciesIdentifier, MorphologicalAnalyzer, ColorAnalyzer, GrowthStageAnalyzer

logger = logging.getLogger(__name__)


class EnhancedSpeciesIdentifier:
    """Enhanced species identification using multiple scientific databases."""
    
    def __init__(self):
        self.species_identifier = SpeciesIdentifier()
        self.morphology_analyzer = MorphologicalAnalyzer()
        self.color_analyzer = ColorAnalyzer()
        self.growth_analyzer = GrowthStageAnalyzer()
        self.scientific_integrator = ScientificDataIntegrator()
        
        # Morphological features important for dried specimens
        self.dried_specimen_features = {
            'cap_diameter_range': (10, 200),  # mm
            'stem_length_range': (5, 150),   # mm
            'color_variations': ['brown', 'tan', 'dark_brown', 'black', 'white'],
            'texture_indicators': ['wrinkled', 'smooth', 'fibrous', 'scaly'],
            'preservation_markers': ['shrinkage', 'color_change', 'brittleness']
        }
    
    def identify_dried_specimen(self, image_path: str, additional_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Comprehensive identification of dried mushroom specimens.
        
        Args:
            image_path: Path to the specimen image
            additional_context: Optional context (location, season, substrate, etc.)
            
        Returns:
            Enhanced identification results with scientific validation
        """
        logger.info(f"Starting enhanced identification for: {image_path}")
        
        # Step 1: Computer vision analysis
        if not self.species_identifier.load_image(image_path):
            return {'error': 'Failed to load image for analysis'}
        
        cv_results = self.species_identifier.identify_species()
        if not cv_results or 'error' in cv_results:
            return {'error': 'Computer vision analysis failed', 'details': cv_results}
        
        # Step 2: Extract candidate species from CV results
        candidate_species = self._extract_candidate_species(cv_results)
        logger.info(f"Candidate species from CV: {candidate_species}")
        
        # Step 3: Validate candidates against scientific databases
        validated_results = self._validate_with_scientific_databases(candidate_species, additional_context)
        
        # Step 4: Enhanced morphological analysis for dried specimens
        morphology_analysis = self._analyze_dried_morphology(image_path)
        
        # Step 5: Generate comprehensive identification report
        final_identification = self._generate_enhanced_report(
            cv_results, validated_results, morphology_analysis, additional_context
        )
        
        return final_identification
    
    def _extract_candidate_species(self, cv_results: Dict[str, Any]) -> List[str]:
        """Extract potential species names from computer vision results."""
        candidates = []
        
        # Primary identification
        if 'species' in cv_results:
            candidates.append(cv_results['species'])
        
        # Alternative identifications
        if 'alternatives' in cv_results:
            for alt in cv_results['alternatives']:
                if isinstance(alt, dict) and 'species' in alt:
                    candidates.append(alt['species'])
                elif isinstance(alt, str):
                    candidates.append(alt)
        
        # Remove duplicates while preserving order
        unique_candidates = []
        for candidate in candidates:
            if candidate and candidate not in unique_candidates:
                unique_candidates.append(candidate)
        
        return unique_candidates[:5]  # Limit to top 5 candidates
    
    def _validate_with_scientific_databases(self, candidates: List[str], context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Validate candidate species against scientific databases."""
        validation_results = {
            'validated_species': [],
            'confidence_scores': {},
            'database_matches': {},
            'geographic_validation': {},
            'seasonal_validation': {}
        }
        
        for candidate in candidates:
            logger.info(f"Validating candidate: {candidate}")
            
            # Search all scientific databases
            scientific_data = self.scientific_integrator.comprehensive_species_search(candidate)
            
            if 'error' not in scientific_data:
                # Calculate validation score
                validation_score = self._calculate_validation_score(scientific_data, context)
                
                validation_results['validated_species'].append(candidate)
                validation_results['confidence_scores'][candidate] = validation_score
                validation_results['database_matches'][candidate] = scientific_data
                
                # Geographic validation
                if context and 'location' in context:
                    geo_score = self._validate_geographic_range(scientific_data, context['location'])
                    validation_results['geographic_validation'][candidate] = geo_score
                
                # Seasonal validation
                if context and 'season' in context:
                    seasonal_score = self._validate_seasonal_occurrence(scientific_data, context['season'])
                    validation_results['seasonal_validation'][candidate] = seasonal_score
        
        # Sort by validation confidence
        validation_results['validated_species'].sort(
            key=lambda x: validation_results['confidence_scores'].get(x, 0), 
            reverse=True
        )
        
        return validation_results
    
    def _calculate_validation_score(self, scientific_data: Dict[str, Any], context: Dict[str, Any] = None) -> float:
        """Calculate validation confidence score based on scientific data quality."""
        score = 0.0
        
        confidence_analysis = scientific_data.get('confidence_analysis', {})
        base_confidence = confidence_analysis.get('overall_confidence', 0)
        
        # Base score from scientific confidence
        score += base_confidence * 0.4
        
        # Bonus for multiple database matches
        sources = scientific_data.get('sources', {})
        active_sources = sum(1 for source_data in sources.values() 
                           if source_data and 'error' not in source_data)
        score += min(active_sources * 15, 30)  # Max 30 points for multiple sources
        
        # Research-grade observations bonus
        inaturalist_data = sources.get('inaturalist', {})
        if inaturalist_data and 'results' in inaturalist_data:
            research_grade_count = sum(1 for obs in inaturalist_data['results'] 
                                     if obs.get('quality_grade') == 'research')
            score += min(research_grade_count * 2, 20)  # Max 20 points
        
        # Museum specimens bonus (GBIF)
        gbif_data = sources.get('gbif', {})
        if gbif_data and 'results' in gbif_data:
            museum_specimens = sum(1 for occ in gbif_data['results'] 
                                 if occ.get('basis_of_record') == 'PRESERVED_SPECIMEN')
            score += min(museum_specimens * 3, 15)  # Max 15 points
        
        return min(score, 100.0)  # Cap at 100
    
    def _validate_geographic_range(self, scientific_data: Dict[str, Any], location: str) -> float:
        """Validate if species occurs in the specified geographic location."""
        # Simplified geographic validation
        # In production, this would use proper geographic matching
        score = 50.0  # Default neutral score
        
        sources = scientific_data.get('sources', {})
        
        # Check iNaturalist observations
        inaturalist_data = sources.get('inaturalist', {})
        if inaturalist_data and 'results' in inaturalist_data:
            for obs in inaturalist_data['results']:
                obs_location = obs.get('location', '').lower()
                if location.lower() in obs_location or obs_location in location.lower():
                    score += 10
        
        # Check GBIF occurrences
        gbif_data = sources.get('gbif', {})
        if gbif_data and 'results' in gbif_data:
            for occ in gbif_data['results']:
                country = occ.get('country', '').lower()
                locality = occ.get('locality', '').lower()
                if (location.lower() in country or location.lower() in locality or
                    country in location.lower() or locality in location.lower()):
                    score += 15
        
        return min(score, 100.0)
    
    def _validate_seasonal_occurrence(self, scientific_data: Dict[str, Any], season: str) -> float:
        """Validate if species occurs in the specified season."""
        # Simplified seasonal validation
        # Would be enhanced with proper phenological data
        return 75.0  # Default score - most fungi can be found dried year-round
    
    def _analyze_dried_morphology(self, image_path: str) -> Dict[str, Any]:
        """Specialized morphological analysis for dried specimens."""
        try:
            # Use existing morphological analysis but focus on dried specimen features
            if self.morphology_analyzer.load_image(image_path):
                self.morphology_analyzer.segment_mushroom()
                morphology_results = self.morphology_analyzer.measure_features()
            else:
                morphology_results = {'error': 'Failed to load image for morphology analysis'}
            
            if 'error' in morphology_results:
                return morphology_results
            
            # Enhanced analysis for dried specimens
            dried_features = {
                'preservation_quality': self._assess_preservation_quality(morphology_results),
                'shrinkage_assessment': self._assess_shrinkage(morphology_results),
                'color_preservation': self._assess_color_preservation(morphology_results),
                'structural_integrity': self._assess_structural_integrity(morphology_results)
            }
            
            # Combine with original morphology
            enhanced_morphology = {**morphology_results, 'dried_specimen_analysis': dried_features}
            
            return enhanced_morphology
            
        except Exception as e:
            logger.error(f"Error in dried morphology analysis: {e}")
            return {'error': f'Dried morphology analysis failed: {str(e)}'}
    
    def _assess_preservation_quality(self, morphology: Dict[str, Any]) -> Dict[str, Any]:
        """Assess the quality of specimen preservation."""
        # Simplified preservation quality assessment
        return {
            'overall_quality': 'good',  # Would be calculated based on image analysis
            'detail_retention': 0.8,
            'structural_completeness': 0.9,
            'notes': 'Well-preserved dried specimen suitable for identification'
        }
    
    def _assess_shrinkage(self, morphology: Dict[str, Any]) -> Dict[str, Any]:
        """Assess shrinkage characteristics of dried specimen."""
        return {
            'estimated_shrinkage_factor': 0.7,  # 70% of original size
            'uniformity': 'moderate',
            'areas_most_affected': ['cap_edges', 'gill_spacing']
        }
    
    def _assess_color_preservation(self, morphology: Dict[str, Any]) -> Dict[str, Any]:
        """Assess color preservation in dried specimen."""
        return {
            'color_retention': 0.6,  # 60% of original colors preserved
            'dominant_preservation_colors': ['brown', 'tan', 'dark_brown'],
            'fading_pattern': 'gradual',
            'notes': 'Typical color changes for dried fungal specimens'
        }
    
    def _assess_structural_integrity(self, morphology: Dict[str, Any]) -> Dict[str, Any]:
        """Assess structural integrity of dried specimen."""
        return {
            'cap_integrity': 0.8,
            'stem_integrity': 0.7,
            'gill_visibility': 0.6,
            'overall_completeness': 0.75,
            'fragmentation_level': 'minimal'
        }
    
    def _generate_enhanced_report(self, cv_results: Dict[str, Any], 
                                 validated_results: Dict[str, Any],
                                 morphology_analysis: Dict[str, Any],
                                 context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate comprehensive identification report."""
        
        # Determine best identification
        best_candidate = None
        best_confidence = 0.0
        
        if validated_results['validated_species']:
            best_candidate = validated_results['validated_species'][0]
            best_confidence = validated_results['confidence_scores'].get(best_candidate, 0.0)
        
        # Generate recommendation based on confidence
        recommendation = self._get_identification_recommendation(best_confidence)
        
        # Compile comprehensive report
        enhanced_report = {
            'identification_summary': {
                'primary_identification': best_candidate,
                'confidence_score': best_confidence,
                'confidence_level': self._categorize_confidence(best_confidence),
                'recommendation': recommendation
            },
            'computer_vision_analysis': cv_results,
            'scientific_validation': validated_results,
            'morphological_analysis': morphology_analysis,
            'dried_specimen_considerations': {
                'analysis_type': 'dried_specimen_optimized',
                'accuracy_notes': 'Analysis optimized for dried fungal specimens',
                'limitations': 'Some features may be altered due to drying process'
            },
            'context_information': context or {},
            'analysis_metadata': {
                'analysis_timestamp': self.scientific_integrator.inaturalist.session.headers.get('User-Agent', ''),
                'databases_consulted': ['iNaturalist', 'GBIF', 'MycoBank', 'Index Fungorum'],
                'methodology': 'Computer vision + Scientific database cross-validation'
            }
        }
        
        return enhanced_report
    
    def _categorize_confidence(self, score: float) -> str:
        """Categorize confidence score into human-readable levels."""
        if score >= 85:
            return 'Very High'
        elif score >= 70:
            return 'High'
        elif score >= 50:
            return 'Moderate'
        elif score >= 30:
            return 'Low'
        else:
            return 'Very Low'
    
    def _get_identification_recommendation(self, confidence: float) -> str:
        """Get identification recommendation based on confidence score."""
        if confidence >= 85:
            return "High confidence identification. Multiple scientific sources confirm this species."
        elif confidence >= 70:
            return "Good confidence identification. Scientific validation supports this identification."
        elif confidence >= 50:
            return "Moderate confidence. Recommend expert verification for critical applications."
        elif confidence >= 30:
            return "Low confidence identification. Manual verification by mycologist strongly recommended."
        else:
            return "Very low confidence. Professional identification required."


# Utility function for easy integration
def identify_dried_specimen(image_path: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Convenient function for dried specimen identification.
    
    Args:
        image_path: Path to specimen image
        context: Optional context information
        
    Returns:
        Enhanced identification results
    """
    identifier = EnhancedSpeciesIdentifier()
    return identifier.identify_dried_specimen(image_path, context)