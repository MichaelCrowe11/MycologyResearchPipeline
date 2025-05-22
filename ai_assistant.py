"""
AI Assistant Module for Mycology Research Pipeline.

This module provides integration with OpenAI's models to assist with research,
data analysis, and code generation for mycological applications.
"""

import os
import logging
import json
from typing import Dict, List, Any, Optional
import openai
from openai import OpenAI

logger = logging.getLogger(__name__)

class AIAssistant:
    """Class for interacting with OpenAI's models."""
    
    def __init__(self):
        """Initialize the AI assistant."""
        self.api_key = os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            logger.warning("OpenAI API key not found in environment variables.")
        
        self.client = OpenAI(api_key=self.api_key)
        self.default_model = "gpt-3.5-turbo"
    
    def analyze_sample_data(self, sample_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze sample data and provide insights.
        
        Args:
            sample_data: Dictionary containing sample information
            
        Returns:
            Dict: Analysis results and insights
        """
        prompt = self._construct_sample_analysis_prompt(sample_data)
        
        try:
            response = self.client.chat.completions.create(
                model=self.default_model,
                messages=[
                    {"role": "system", "content": "You are a mycology research assistant with expertise in analyzing fungal samples and compounds."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1000
            )
            
            analysis_text = response.choices[0].message.content
            
            # Extract structured data if possible
            try:
                # Try to find JSON in the response
                if analysis_text is not None:
                    json_start = analysis_text.find("{")
                    json_end = analysis_text.rfind("}") + 1
                    
                    if json_start >= 0 and json_end > json_start:
                        json_str = analysis_text[json_start:json_end]
                        structured_data = json.loads(json_str)
                        return {
                            "raw_response": analysis_text,
                            "structured_data": structured_data
                        }
            except json.JSONDecodeError:
                pass
            
            # Return unstructured response if JSON parsing failed
            return {
                "raw_response": analysis_text,
                "structured_data": None
            }
            
        except Exception as e:
            logger.error(f"Error analyzing sample data: {str(e)}")
            return {
                "error": str(e),
                "raw_response": None,
                "structured_data": None
            }
    
    def analyze_compounds(self, compound_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze compound data and provide insights about bioactivity and potential applications.
        
        Args:
            compound_data: List of dictionaries containing compound information
            
        Returns:
            Dict: Analysis results and insights about compounds
        """
        prompt = self._construct_compound_analysis_prompt(compound_data)
        
        try:
            response = self.client.chat.completions.create(
                model=self.default_model,
                messages=[
                    {"role": "system", "content": "You are a mycology research assistant with expertise in fungal compounds, bioactivity, and medicinal applications."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1000
            )
            
            analysis_text = response.choices[0].message.content
            
            return {
                "raw_response": analysis_text
            }
            
        except Exception as e:
            logger.error(f"Error analyzing compound data: {str(e)}")
            return {
                "error": str(e),
                "raw_response": None
            }
    
    def generate_research_hypothesis(self, context_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate research hypotheses based on existing data.
        
        Args:
            context_data: Dictionary containing research context
            
        Returns:
            Dict: Generated hypotheses and rationale
        """
        prompt = self._construct_hypothesis_prompt(context_data)
        
        try:
            response = self.client.chat.completions.create(
                model=self.default_model,
                messages=[
                    {"role": "system", "content": "You are a mycology research assistant specializing in generating novel research hypotheses based on existing data."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,  # Higher temperature for more creative responses
                max_tokens=1000
            )
            
            hypothesis_text = response.choices[0].message.content
            
            return {
                "raw_response": hypothesis_text
            }
            
        except Exception as e:
            logger.error(f"Error generating research hypothesis: {str(e)}")
            return {
                "error": str(e),
                "raw_response": None
            }
    
    def analyze_species_similarity(self, species_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze similarities between different fungal species.
        
        Args:
            species_data: List of dictionaries containing species information
            
        Returns:
            Dict: Analysis of similarities and differences
        """
        prompt = self._construct_species_comparison_prompt(species_data)
        
        try:
            response = self.client.chat.completions.create(
                model=self.default_model,
                messages=[
                    {"role": "system", "content": "You are a mycology research assistant with expertise in fungal taxonomy and comparative biology."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1000
            )
            
            analysis_text = response.choices[0].message.content
            
            return {
                "raw_response": analysis_text
            }
            
        except Exception as e:
            logger.error(f"Error analyzing species similarity: {str(e)}")
            return {
                "error": str(e),
                "raw_response": None
            }
    
    def generate_code_sample(self, task_description: str, language: str = "python") -> Dict[str, Any]:
        """
        Generate code samples for common research tasks.
        
        Args:
            task_description: Description of the coding task
            language: Programming language (default: python)
            
        Returns:
            Dict: Generated code and explanation
        """
        prompt = f"Generate {language} code for the following task in mycology research:\n\n{task_description}\n\nProvide code with comments explaining each step."
        
        try:
            response = self.client.chat.completions.create(
                model=self.default_model,
                messages=[
                    {"role": "system", "content": f"You are a mycology research assistant with expertise in programming, especially {language} for scientific data analysis."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,  # Lower temperature for more precise code
                max_tokens=1000
            )
            
            code_text = response.choices[0].message.content
            
            return {
                "raw_response": code_text
            }
            
        except Exception as e:
            logger.error(f"Error generating code sample: {str(e)}")
            return {
                "error": str(e),
                "raw_response": None
            }
    
    def analyze_research_literature(self, literature_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze research literature and provide a summary of findings.
        
        Args:
            literature_data: List of dictionaries containing literature information
            
        Returns:
            Dict: Analysis and summary of literature
        """
        prompt = self._construct_literature_analysis_prompt(literature_data)
        
        try:
            response = self.client.chat.completions.create(
                model=self.default_model,
                messages=[
                    {"role": "system", "content": "You are a mycology research assistant with expertise in scientific literature analysis and research trends."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1500
            )
            
            analysis_text = response.choices[0].message.content
            
            return {
                "raw_response": analysis_text
            }
            
        except Exception as e:
            logger.error(f"Error analyzing research literature: {str(e)}")
            return {
                "error": str(e),
                "raw_response": None
            }
    
    def _construct_sample_analysis_prompt(self, sample_data: Dict[str, Any]) -> str:
        """
        Construct a prompt for sample analysis.
        
        Args:
            sample_data: Dictionary containing sample information
            
        Returns:
            str: Formatted prompt
        """
        prompt = "Analyze the following mycological sample data and provide insights:\n\n"
        
        # Add sample details
        prompt += f"Sample ID: {sample_data.get('id', 'Unknown')}\n"
        prompt += f"Name: {sample_data.get('name', 'Unknown')}\n"
        prompt += f"Species: {sample_data.get('species', 'Unknown')}\n"
        prompt += f"Collection Date: {sample_data.get('collection_date', 'Unknown')}\n"
        prompt += f"Location: {sample_data.get('location', 'Unknown')}\n"
        
        # Add sample metadata if available
        if sample_data.get('sample_metadata'):
            prompt += "\nMetadata:\n"
            for key, value in sample_data['sample_metadata'].items():
                prompt += f"- {key}: {value}\n"
        
        # Add compounds if available
        if sample_data.get('compounds'):
            prompt += "\nCompounds:\n"
            for compound in sample_data['compounds']:
                prompt += f"- {compound.get('name', 'Unknown Compound')}"
                if compound.get('formula'):
                    prompt += f" (Formula: {compound['formula']})"
                if compound.get('bioactivity_index'):
                    prompt += f" (Bioactivity Index: {compound['bioactivity_index']})"
                prompt += "\n"
        
        # Analysis requests
        prompt += "\nPlease provide the following analysis:\n"
        prompt += "1. Overall assessment of the sample\n"
        prompt += "2. Notable characteristics and potential research value\n"
        prompt += "3. Recommendations for further analysis\n"
        prompt += "4. Potential applications or significance\n"
        
        # Request structured format
        prompt += "\nPlease include a structured JSON summary with the following schema:\n"
        prompt += "{\n"
        prompt += '  "overall_assessment": "text",\n'
        prompt += '  "notable_characteristics": ["item1", "item2", ...],\n'
        prompt += '  "recommended_analyses": ["analysis1", "analysis2", ...],\n'
        prompt += '  "potential_applications": ["application1", "application2", ...],\n'
        prompt += '  "research_value_score": 1-10 (integer),\n'
        prompt += '  "confidence_level": "high/medium/low"\n'
        prompt += "}\n"
        
        return prompt
    
    def _construct_compound_analysis_prompt(self, compound_data: List[Dict[str, Any]]) -> str:
        """
        Construct a prompt for compound analysis.
        
        Args:
            compound_data: List of dictionaries containing compound information
            
        Returns:
            str: Formatted prompt
        """
        prompt = "Analyze the following compounds from mycological samples and provide insights on bioactivity and potential applications:\n\n"
        
        for i, compound in enumerate(compound_data, 1):
            prompt += f"Compound {i}:\n"
            prompt += f"- Name: {compound.get('name', 'Unknown')}\n"
            prompt += f"- Formula: {compound.get('formula', 'Unknown')}\n"
            prompt += f"- Molecular Weight: {compound.get('molecular_weight', 'Unknown')}\n"
            prompt += f"- Bioactivity Index: {compound.get('bioactivity_index', 'Unknown')}\n"
            prompt += f"- Concentration: {compound.get('concentration', 'Unknown')}\n"
            
            # Add compound metadata if available
            if compound.get('compound_metadata'):
                prompt += "- Additional Properties:\n"
                for key, value in compound['compound_metadata'].items():
                    prompt += f"  * {key}: {value}\n"
            prompt += "\n"
        
        # Analysis requests
        prompt += "\nPlease provide the following analysis:\n"
        prompt += "1. Bioactivity assessment for each compound\n"
        prompt += "2. Potential medicinal applications\n"
        prompt += "3. Structural similarities and differences between compounds\n"
        prompt += "4. Recommendations for further characterization\n"
        prompt += "5. Comparison with known medicinal compounds if applicable\n"
        
        return prompt
    
    def _construct_hypothesis_prompt(self, context_data: Dict[str, Any]) -> str:
        """
        Construct a prompt for generating research hypotheses.
        
        Args:
            context_data: Dictionary containing research context
            
        Returns:
            str: Formatted prompt
        """
        prompt = "Generate novel research hypotheses based on the following mycological research context:\n\n"
        
        # Add research area
        prompt += f"Research Area: {context_data.get('research_area', 'General Mycology')}\n"
        
        # Add existing findings
        if context_data.get('existing_findings'):
            prompt += "\nExisting Findings:\n"
            for finding in context_data['existing_findings']:
                prompt += f"- {finding}\n"
        
        # Add target objectives
        if context_data.get('objectives'):
            prompt += "\nResearch Objectives:\n"
            for objective in context_data['objectives']:
                prompt += f"- {objective}\n"
        
        # Add available resources/methods
        if context_data.get('available_methods'):
            prompt += "\nAvailable Methods and Resources:\n"
            for method in context_data['available_methods']:
                prompt += f"- {method}\n"
        
        # Analysis requests
        prompt += "\nPlease generate 3-5 novel research hypotheses that are:\n"
        prompt += "1. Scientifically plausible based on existing data\n"
        prompt += "2. Testable with available methods\n"
        prompt += "3. Aligned with the research objectives\n"
        prompt += "4. Novel and contribute to advancing the field\n"
        
        prompt += "\nFor each hypothesis, please provide:\n"
        prompt += "- The hypothesis statement\n"
        prompt += "- Rationale based on existing data\n"
        prompt += "- Suggested experimental approach\n"
        prompt += "- Potential significance if confirmed\n"
        
        return prompt
    
    def _construct_species_comparison_prompt(self, species_data: List[Dict[str, Any]]) -> str:
        """
        Construct a prompt for species comparison.
        
        Args:
            species_data: List of dictionaries containing species information
            
        Returns:
            str: Formatted prompt
        """
        prompt = "Analyze and compare the following fungal species:\n\n"
        
        for i, species in enumerate(species_data, 1):
            prompt += f"Species {i}:\n"
            prompt += f"- Scientific Name: {species.get('scientific_name', 'Unknown')}\n"
            prompt += f"- Common Name: {species.get('common_name', 'Unknown')}\n"
            
            # Add habitat if available
            if species.get('habitat'):
                prompt += f"- Habitat: {species['habitat']}\n"
            
            # Add properties if available
            if species.get('properties'):
                prompt += "- Properties:\n"
                for key, value in species['properties'].items():
                    prompt += f"  * {key}: {value}\n"
            
            # Add compounds if available
            if species.get('compounds'):
                prompt += "- Known Compounds:\n"
                for compound in species['compounds']:
                    prompt += f"  * {compound}\n"
            
            prompt += "\n"
        
        # Analysis requests
        prompt += "\nPlease provide the following comparative analysis:\n"
        prompt += "1. Taxonomic relationships between the species\n"
        prompt += "2. Ecological and habitat similarities/differences\n"
        prompt += "3. Common compounds or biochemical pathways\n"
        prompt += "4. Evolutionary insights\n"
        prompt += "5. Research or commercial significance of each species\n"
        prompt += "6. Recommendations for comparative studies\n"
        
        return prompt
    
    def _construct_literature_analysis_prompt(self, literature_data: List[Dict[str, Any]]) -> str:
        """
        Construct a prompt for literature analysis.
        
        Args:
            literature_data: List of dictionaries containing literature information
            
        Returns:
            str: Formatted prompt
        """
        prompt = "Analyze the following mycological research literature and provide a summary of findings and trends:\n\n"
        
        for i, publication in enumerate(literature_data, 1):
            prompt += f"Publication {i}:\n"
            prompt += f"- Title: {publication.get('title', 'Unknown')}\n"
            prompt += f"- Authors: {publication.get('authors', 'Unknown')}\n"
            prompt += f"- Journal: {publication.get('journal', 'Unknown')}\n"
            prompt += f"- Year: {publication.get('year', 'Unknown')}\n"
            
            # Add abstract if available
            if publication.get('abstract'):
                abstract = publication['abstract']
                # Truncate very long abstracts
                if len(abstract) > 500:
                    abstract = abstract[:497] + "..."
                prompt += f"- Abstract: {abstract}\n"
            
            # Add keywords if available
            if publication.get('keywords'):
                keywords = ", ".join(publication['keywords'])
                prompt += f"- Keywords: {keywords}\n"
            
            prompt += "\n"
        
        # Analysis requests
        prompt += "\nPlease provide the following analysis of this literature:\n"
        prompt += "1. Summary of key findings across publications\n"
        prompt += "2. Identification of research trends and emerging areas\n"
        prompt += "3. Notable methodologies used\n"
        prompt += "4. Gaps or contradictions in the research\n"
        prompt += "5. Suggestions for future research directions\n"
        
        return prompt


# Utility functions for direct use
def analyze_sample(sample_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Utility function to analyze a sample.
    
    Args:
        sample_data: Dictionary containing sample information
        
    Returns:
        Dict: Analysis results
    """
    assistant = AIAssistant()
    return assistant.analyze_sample_data(sample_data)


def generate_hypothesis(context_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Utility function to generate research hypotheses.
    
    Args:
        context_data: Dictionary containing research context
        
    Returns:
        Dict: Generated hypotheses
    """
    assistant = AIAssistant()
    return assistant.generate_research_hypothesis(context_data)


def generate_code(task_description: str, language: str = "python") -> Dict[str, Any]:
    """
    Utility function to generate code.
    
    Args:
        task_description: Description of the coding task
        language: Programming language (default: python)
        
    Returns:
        Dict: Generated code
    """
    assistant = AIAssistant()
    return assistant.generate_code_sample(task_description, language)