"""
Smart Parameter Generator for Mycology Research Pipeline.

This module automatically generates JSON parameters for batch processing
based on simple user descriptions and goals.
"""

import json
import re
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)


class SmartParameterGenerator:
    """Generate analysis parameters from natural language descriptions."""
    
    def __init__(self):
        self.analysis_templates = {
            'species_identification': {
                'confidence_threshold': 0.7,
                'include_alternatives': True,
                'max_alternatives': 5,
                'use_scientific_validation': True
            },
            'morphological_analysis': {
                'measurement_units': 'pixels',
                'include_shape_features': True,
                'calculate_ratios': True,
                'contour_detection': 'auto'
            },
            'color_analysis': {
                'color_spaces': ['rgb', 'hsv', 'lab'],
                'dominant_colors_count': 5,
                'generate_histogram': True,
                'color_tolerance': 10
            },
            'growth_stage_analysis': {
                'stage_categories': ['early', 'mid', 'mature', 'late'],
                'include_harvest_timing': True,
                'confidence_threshold': 0.6
            },
            'dried_specimen_analysis': {
                'specimen_type': 'dried',
                'adjust_for_shrinkage': True,
                'color_correction': True,
                'enhanced_identification': True
            },
            'fresh_specimen_analysis': {
                'specimen_type': 'fresh',
                'full_color_analysis': True,
                'moisture_consideration': True
            },
            'batch_processing': {
                'parallel_processing': True,
                'save_intermediate_results': True,
                'generate_summary_report': True,
                'output_format': 'csv'
            }
        }
        
        self.keyword_mappings = {
            # Specimen types
            'dried': ['dried', 'dehydrated', 'preserved', 'desiccated'],
            'fresh': ['fresh', 'live', 'wet', 'moist', 'recently_harvested'],
            
            # Analysis goals
            'identification': ['identify', 'classify', 'determine_species', 'name', 'what_is'],
            'measurement': ['measure', 'size', 'dimensions', 'quantify', 'metrics'],
            'color': ['color', 'colour', 'hue', 'pigment', 'shade', 'appearance'],
            'growth': ['growth', 'stage', 'maturity', 'development', 'harvest'],
            'quality': ['quality', 'condition', 'health', 'assessment'],
            
            # Research purposes
            'cultivation': ['cultivation', 'growing', 'farming', 'production'],
            'research': ['research', 'study', 'analysis', 'investigation'],
            'commercial': ['commercial', 'business', 'sale', 'market'],
            'education': ['education', 'learning', 'teaching', 'academic'],
            
            # Accuracy levels
            'high_accuracy': ['precise', 'accurate', 'detailed', 'thorough', 'comprehensive'],
            'quick_analysis': ['quick', 'fast', 'rapid', 'simple', 'basic'],
            
            # Species groups
            'medicinal': ['medicinal', 'therapeutic', 'healing', 'cordyceps', 'reishi', 'chaga'],
            'edible': ['edible', 'culinary', 'food', 'cooking', 'shiitake', 'oyster'],
            'wild': ['wild', 'foraged', 'forest', 'natural', 'field_collected']
        }
    
    def generate_parameters(self, user_description: str, analysis_goals: List[str] = None) -> Dict[str, Any]:
        """
        Generate JSON parameters based on user description and goals.
        
        Args:
            user_description: Natural language description of what user wants to do
            analysis_goals: List of specific analysis types requested
            
        Returns:
            Dict containing generated parameters
        """
        logger.info(f"Generating parameters for: {user_description}")
        
        # Parse user description
        parsed_intent = self._parse_user_intent(user_description)
        
        # Start with base parameters
        parameters = {
            'analysis_type': 'comprehensive',
            'generated_from_description': True,
            'user_description': user_description,
            'confidence_level': 'standard'
        }
        
        # Add specimen-specific parameters
        if parsed_intent['specimen_type']:
            parameters.update(self.analysis_templates[f"{parsed_intent['specimen_type']}_specimen_analysis"])
        
        # Add analysis-specific parameters based on goals
        if parsed_intent['analysis_goals']:
            for goal in parsed_intent['analysis_goals']:
                if goal in self.analysis_templates:
                    parameters.update(self.analysis_templates[goal])
        
        # Add research context parameters
        if parsed_intent['research_purpose']:
            parameters['research_context'] = parsed_intent['research_purpose']
            parameters.update(self._get_purpose_specific_params(parsed_intent['research_purpose']))
        
        # Adjust accuracy based on user intent
        if parsed_intent['accuracy_preference']:
            parameters.update(self._get_accuracy_params(parsed_intent['accuracy_preference']))
        
        # Add species-specific parameters
        if parsed_intent['species_group']:
            parameters.update(self._get_species_group_params(parsed_intent['species_group']))
        
        # Add batch processing parameters if multiple samples mentioned
        if self._mentions_batch_processing(user_description):
            parameters.update(self.analysis_templates['batch_processing'])
        
        return parameters
    
    def _parse_user_intent(self, description: str) -> Dict[str, Any]:
        """Parse user description to extract intent."""
        description_lower = description.lower()
        
        intent = {
            'specimen_type': None,
            'analysis_goals': [],
            'research_purpose': None,
            'accuracy_preference': None,
            'species_group': None
        }
        
        # Detect specimen type
        for specimen_type, keywords in self.keyword_mappings.items():
            if specimen_type in ['dried', 'fresh']:
                if any(keyword in description_lower for keyword in keywords):
                    intent['specimen_type'] = specimen_type
                    break
        
        # Detect analysis goals
        for goal, keywords in self.keyword_mappings.items():
            if goal in ['identification', 'measurement', 'color', 'growth', 'quality']:
                if any(keyword in description_lower for keyword in keywords):
                    goal_mapping = {
                        'identification': 'species_identification',
                        'measurement': 'morphological_analysis',
                        'color': 'color_analysis',
                        'growth': 'growth_stage_analysis',
                        'quality': 'quality_assessment'
                    }
                    intent['analysis_goals'].append(goal_mapping.get(goal, goal))
        
        # Detect research purpose
        for purpose, keywords in self.keyword_mappings.items():
            if purpose in ['cultivation', 'research', 'commercial', 'education']:
                if any(keyword in description_lower for keyword in keywords):
                    intent['research_purpose'] = purpose
                    break
        
        # Detect accuracy preference
        for accuracy, keywords in self.keyword_mappings.items():
            if accuracy in ['high_accuracy', 'quick_analysis']:
                if any(keyword in description_lower for keyword in keywords):
                    intent['accuracy_preference'] = accuracy
                    break
        
        # Detect species group
        for group, keywords in self.keyword_mappings.items():
            if group in ['medicinal', 'edible', 'wild']:
                if any(keyword in description_lower for keyword in keywords):
                    intent['species_group'] = group
                    break
        
        return intent
    
    def _get_purpose_specific_params(self, purpose: str) -> Dict[str, Any]:
        """Get parameters specific to research purpose."""
        purpose_params = {
            'cultivation': {
                'focus_on_growth_stage': True,
                'include_harvest_recommendations': True,
                'track_development_patterns': True
            },
            'research': {
                'detailed_measurements': True,
                'scientific_nomenclature': True,
                'include_statistical_analysis': True,
                'generate_research_report': True
            },
            'commercial': {
                'quality_grading': True,
                'market_value_indicators': True,
                'standardized_measurements': True
            },
            'education': {
                'include_explanations': True,
                'visual_learning_aids': True,
                'simplified_terminology': True
            }
        }
        return purpose_params.get(purpose, {})
    
    def _get_accuracy_params(self, accuracy: str) -> Dict[str, Any]:
        """Get parameters based on accuracy preference."""
        accuracy_params = {
            'high_accuracy': {
                'confidence_level': 'high',
                'confidence_threshold': 0.85,
                'use_multiple_validation_sources': True,
                'detailed_feature_extraction': True,
                'cross_reference_databases': True
            },
            'quick_analysis': {
                'confidence_level': 'standard',
                'confidence_threshold': 0.65,
                'fast_processing_mode': True,
                'essential_features_only': True
            }
        }
        return accuracy_params.get(accuracy, {})
    
    def _get_species_group_params(self, group: str) -> Dict[str, Any]:
        """Get parameters specific to species groups."""
        group_params = {
            'medicinal': {
                'focus_on_bioactive_compounds': True,
                'therapeutic_properties_analysis': True,
                'authenticity_verification': True,
                'quality_standards': 'pharmaceutical'
            },
            'edible': {
                'safety_assessment': True,
                'nutritional_indicators': True,
                'freshness_evaluation': True,
                'culinary_quality_factors': True
            },
            'wild': {
                'habitat_correlation': True,
                'seasonal_factors': True,
                'geographic_validation': True,
                'conservation_status_check': True
            }
        }
        return group_params.get(group, {})
    
    def _mentions_batch_processing(self, description: str) -> bool:
        """Check if description mentions batch processing."""
        batch_keywords = [
            'multiple', 'batch', 'several', 'many', 'bulk', 'collection',
            'dataset', 'group', 'series', 'set of', 'large number'
        ]
        return any(keyword in description.lower() for keyword in batch_keywords)
    
    def generate_example_descriptions(self) -> List[Dict[str, str]]:
        """Generate example descriptions and their resulting parameters."""
        examples = [
            {
                'description': "I want to identify dried cordyceps specimens for my research",
                'use_case': "Species identification of dried medicinal mushrooms"
            },
            {
                'description': "Analyze fresh shiitake mushrooms for cultivation timing",
                'use_case': "Growth stage analysis for cultivation"
            },
            {
                'description': "Quick color analysis of multiple wild mushroom samples",
                'use_case': "Batch color analysis of wild specimens"
            },
            {
                'description': "Detailed morphological measurements for academic research",
                'use_case': "Comprehensive morphological analysis"
            },
            {
                'description': "Assess quality of dried reishi for commercial sale",
                'use_case': "Quality assessment for commercial purposes"
            }
        ]
        
        for example in examples:
            example['generated_parameters'] = self.generate_parameters(example['description'])
        
        return examples


def generate_smart_parameters(description: str, goals: List[str] = None) -> str:
    """
    Utility function to generate parameters JSON string from description.
    
    Args:
        description: User's natural language description
        goals: Optional list of specific analysis goals
        
    Returns:
        JSON string of generated parameters
    """
    generator = SmartParameterGenerator()
    parameters = generator.generate_parameters(description, goals)
    return json.dumps(parameters, indent=2)


def get_parameter_examples() -> List[Dict[str, Any]]:
    """Get example descriptions and their generated parameters."""
    generator = SmartParameterGenerator()
    return generator.generate_example_descriptions()


# Example usage for testing
if __name__ == "__main__":
    generator = SmartParameterGenerator()
    
    # Test descriptions
    test_descriptions = [
        "I need to identify dried cordyceps samples for my research project",
        "Quick analysis of fresh mushrooms for harvest timing",
        "Detailed measurements of wild mushrooms for academic study",
        "Color analysis of multiple edible mushroom varieties"
    ]
    
    for desc in test_descriptions:
        print(f"\nDescription: {desc}")
        params = generator.generate_parameters(desc)
        print(f"Generated Parameters: {json.dumps(params, indent=2)}")