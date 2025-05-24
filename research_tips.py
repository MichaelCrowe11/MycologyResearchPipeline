"""
Contextual Research Tip Generator for Mycology Research Pipeline.

This module provides contextual research tips with whimsical fungi illustrations
based on user activity and current research context.
"""

import random
from datetime import datetime
from typing import Dict, List, Any, Optional

class ResearchTipGenerator:
    """Generate contextual research tips with fungi illustrations."""
    
    def __init__(self):
        """Initialize the research tip generator."""
        self.tips_database = self._load_tips_database()
        self.fungi_illustrations = self._load_fungi_illustrations()
    
    def _load_tips_database(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load research tips organized by context."""
        return {
            'sample_collection': [
                {
                    'tip': 'Always collect samples in the early morning when spores are most concentrated and moisture levels are optimal.',
                    'category': 'field_work',
                    'difficulty': 'beginner',
                    'fungi_type': 'chanterelle'
                },
                {
                    'tip': 'Use sterile cotton swabs dampened with distilled water to collect spores from delicate specimens without damaging the fruiting body.',
                    'category': 'laboratory',
                    'difficulty': 'intermediate',
                    'fungi_type': 'oyster_mushroom'
                },
                {
                    'tip': 'Document GPS coordinates and environmental conditions - temperature, humidity, and substrate type can be crucial for species identification.',
                    'category': 'documentation',
                    'difficulty': 'beginner',
                    'fungi_type': 'shiitake'
                }
            ],
            'analysis': [
                {
                    'tip': 'When analyzing bioactive compounds, consider seasonal variations - many fungi produce different metabolite profiles throughout their life cycle.',
                    'category': 'biochemistry',
                    'difficulty': 'advanced',
                    'fungi_type': 'reishi'
                },
                {
                    'tip': 'Use multiple extraction methods (water, ethanol, methanol) to capture the full spectrum of bioactive compounds in your samples.',
                    'category': 'extraction',
                    'difficulty': 'intermediate',
                    'fungi_type': 'turkey_tail'
                },
                {
                    'tip': 'Always run controls alongside your samples - include both positive and negative controls to validate your analytical methods.',
                    'category': 'quality_control',
                    'difficulty': 'beginner',
                    'fungi_type': 'lions_mane'
                }
            ],
            'identification': [
                {
                    'tip': 'Spore prints are your best friend! Different species produce distinctly colored spores that are key to accurate identification.',
                    'category': 'microscopy',
                    'difficulty': 'beginner',
                    'fungi_type': 'amanita'
                },
                {
                    'tip': 'Pay attention to the gill attachment - free, adnate, or decurrent gills can help narrow down species identification significantly.',
                    'category': 'morphology',
                    'difficulty': 'intermediate',
                    'fungi_type': 'portobello'
                },
                {
                    'tip': 'DNA barcoding using ITS sequences is becoming the gold standard for species confirmation, especially for cryptic species.',
                    'category': 'molecular',
                    'difficulty': 'advanced',
                    'fungi_type': 'morel'
                }
            ],
            'cultivation': [
                {
                    'tip': 'Maintain sterile conditions throughout inoculation - even tiny contaminations can ruin entire cultivation batches.',
                    'category': 'sterility',
                    'difficulty': 'beginner',
                    'fungi_type': 'enoki'
                },
                {
                    'tip': 'Different fungi prefer different substrates - hardwood species love oak and beech, while others thrive on agricultural waste.',
                    'category': 'substrates',
                    'difficulty': 'intermediate',
                    'fungi_type': 'wine_cap'
                },
                {
                    'tip': 'Monitor CO2 levels carefully during fruiting - most mushrooms need fresh air exchange to develop properly.',
                    'category': 'environment',
                    'difficulty': 'intermediate',
                    'fungi_type': 'king_oyster'
                }
            ],
            'ai_research': [
                {
                    'tip': 'When using AI for hypothesis generation, provide detailed context about your research goals and existing findings for better suggestions.',
                    'category': 'ai_assistance',
                    'difficulty': 'beginner',
                    'fungi_type': 'porcini'
                },
                {
                    'tip': 'Cross-reference AI-generated insights with peer-reviewed literature - AI is a powerful tool but requires human validation.',
                    'category': 'validation',
                    'difficulty': 'intermediate',
                    'fungi_type': 'cordyceps'
                },
                {
                    'tip': 'Use AI to identify patterns in large datasets that might not be obvious to human analysis, especially in metabolomics data.',
                    'category': 'data_analysis',
                    'difficulty': 'advanced',
                    'fungi_type': 'chaga'
                }
            ]
        }
    
    def _load_fungi_illustrations(self) -> Dict[str, Dict[str, str]]:
        """Load fungi illustration data."""
        return {
            'chanterelle': {
                'svg_path': 'fungi/chanterelle.svg',
                'name': 'Golden Chanterelle',
                'color_scheme': '#FFD700',
                'whimsical_fact': 'Known as the "Golden Treasure of the Forest"'
            },
            'oyster_mushroom': {
                'svg_path': 'fungi/oyster.svg',
                'name': 'Oyster Mushroom',
                'color_scheme': '#E6E6FA',
                'whimsical_fact': 'Can grow in space and loves coffee grounds!'
            },
            'shiitake': {
                'svg_path': 'fungi/shiitake.svg',
                'name': 'Shiitake',
                'color_scheme': '#8B4513',
                'whimsical_fact': 'The "Elixir of Life" mushroom in ancient Japan'
            },
            'reishi': {
                'svg_path': 'fungi/reishi.svg',
                'name': 'Reishi',
                'color_scheme': '#B22222',
                'whimsical_fact': 'Called the "Mushroom of Immortality"'
            },
            'turkey_tail': {
                'svg_path': 'fungi/turkey_tail.svg',
                'name': 'Turkey Tail',
                'color_scheme': '#DEB887',
                'whimsical_fact': 'Nature\'s rainbow in fungal form'
            },
            'lions_mane': {
                'svg_path': 'fungi/lions_mane.svg',
                'name': "Lion's Mane",
                'color_scheme': '#F5F5DC',
                'whimsical_fact': 'The brain-boosting pom-pom of the forest'
            },
            'amanita': {
                'svg_path': 'fungi/amanita.svg',
                'name': 'Amanita',
                'color_scheme': '#FF6347',
                'whimsical_fact': 'The fairy tale mushroom with polka dots'
            },
            'portobello': {
                'svg_path': 'fungi/portobello.svg',
                'name': 'Portobello',
                'color_scheme': '#8B7D6B',
                'whimsical_fact': 'Just a button mushroom that grew up!'
            },
            'morel': {
                'svg_path': 'fungi/morel.svg',
                'name': 'Morel',
                'color_scheme': '#DAA520',
                'whimsical_fact': 'The honeycomb treasure hunters seek'
            },
            'enoki': {
                'svg_path': 'fungi/enoki.svg',
                'name': 'Enoki',
                'color_scheme': '#FFFACD',
                'whimsical_fact': 'Tiny white dancers in perfect formation'
            }
        }
    
    def get_contextual_tip(self, context: str, user_level: str = 'beginner') -> Dict[str, Any]:
        """
        Get a contextual research tip based on current activity.
        
        Args:
            context: The research context (sample_collection, analysis, etc.)
            user_level: User experience level (beginner, intermediate, advanced)
            
        Returns:
            Dict containing tip, illustration, and metadata
        """
        tips = self.tips_database.get(context, [])
        
        # Filter tips by user level
        suitable_tips = [tip for tip in tips if tip['difficulty'] == user_level]
        if not suitable_tips:
            suitable_tips = tips  # Fallback to all tips if none match level
        
        if not suitable_tips:
            return self._get_default_tip()
        
        selected_tip = random.choice(suitable_tips)
        fungi_type = selected_tip['fungi_type']
        illustration = self.fungi_illustrations.get(fungi_type, {})
        
        return {
            'tip_text': selected_tip['tip'],
            'category': selected_tip['category'],
            'difficulty': selected_tip['difficulty'],
            'fungi_name': illustration.get('name', 'Mystery Mushroom'),
            'fungi_illustration': illustration.get('svg_path', ''),
            'color_scheme': illustration.get('color_scheme', '#8B4513'),
            'whimsical_fact': illustration.get('whimsical_fact', ''),
            'timestamp': datetime.now().isoformat()
        }
    
    def get_random_tip(self) -> Dict[str, Any]:
        """Get a random research tip for general inspiration."""
        all_contexts = list(self.tips_database.keys())
        random_context = random.choice(all_contexts)
        return self.get_contextual_tip(random_context, 'beginner')
    
    def _get_default_tip(self) -> Dict[str, Any]:
        """Get a default tip when no specific tips are available."""
        return {
            'tip_text': 'Remember: every great discovery starts with careful observation and meticulous documentation!',
            'category': 'general',
            'difficulty': 'universal',
            'fungi_name': 'Wisdom Mushroom',
            'fungi_illustration': 'fungi/default.svg',
            'color_scheme': '#9370DB',
            'whimsical_fact': 'The most magical mushroom is the one you\'re studying right now!',
            'timestamp': datetime.now().isoformat()
        }
    
    def get_tip_by_activity(self, activity_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate tip based on user's current activity.
        
        Args:
            activity_data: Dict containing user's current activity information
            
        Returns:
            Contextual tip with illustration
        """
        # Determine context from activity
        if 'sample' in activity_data.get('current_page', '').lower():
            context = 'sample_collection'
        elif 'analysis' in activity_data.get('current_page', '').lower():
            context = 'analysis'
        elif 'ai' in activity_data.get('current_page', '').lower():
            context = 'ai_research'
        elif 'cultivation' in activity_data.get('current_page', '').lower():
            context = 'cultivation'
        else:
            context = 'identification'
        
        user_level = activity_data.get('user_level', 'beginner')
        return self.get_contextual_tip(context, user_level)


# Initialize global tip generator
tip_generator = ResearchTipGenerator()

def get_research_tip(context: str = 'general', user_level: str = 'beginner') -> Dict[str, Any]:
    """
    Utility function to get a research tip.
    
    Args:
        context: Research context
        user_level: User experience level
        
    Returns:
        Research tip with fungi illustration
    """
    if context == 'general':
        return tip_generator.get_random_tip()
    return tip_generator.get_contextual_tip(context, user_level)