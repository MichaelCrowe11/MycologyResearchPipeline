"""
Scientific Database Integration for Mycology Research Pipeline.

This module provides integration with major scientific databases for enhanced
mushroom identification and validation:
- iNaturalist API for crowd-sourced observations
- GBIF for research-grade specimen data
- MycoBank and Index Fungorum for taxonomic validation
"""

import os
import json
import requests
import time
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class iNaturalistAPI:
    """Integration with iNaturalist for crowd-sourced observation data."""
    
    def __init__(self):
        self.base_url = "https://api.inaturalist.org/v1"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'MycologyResearchPipeline/1.0',
            'Accept': 'application/json'
        })
    
    def search_species(self, species_name: str, limit: int = 20) -> Dict[str, Any]:
        """
        Search for species observations on iNaturalist.
        
        Args:
            species_name: Scientific or common name
            limit: Maximum number of results
            
        Returns:
            Dict containing observation data
        """
        try:
            endpoint = f"{self.base_url}/observations"
            params = {
                'q': species_name,
                'iconic_taxa': 'Fungi',
                'quality_grade': 'research',
                'photos': 'true',
                'per_page': limit,
                'order': 'desc',
                'order_by': 'votes'
            }
            
            response = self.session.get(endpoint, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            return self._process_inaturalist_data(data)
            
        except requests.exceptions.RequestException as e:
            logger.error(f"iNaturalist API error: {e}")
            return {'error': str(e), 'results': []}
    
    def get_taxon_info(self, taxon_id: int) -> Dict[str, Any]:
        """
        Get detailed taxonomic information for a specific taxon.
        
        Args:
            taxon_id: iNaturalist taxon ID
            
        Returns:
            Dict containing taxon details
        """
        try:
            endpoint = f"{self.base_url}/taxa/{taxon_id}"
            response = self.session.get(endpoint, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            return self._process_taxon_data(data)
            
        except requests.exceptions.RequestException as e:
            logger.error(f"iNaturalist taxon API error: {e}")
            return {'error': str(e)}
    
    def _process_inaturalist_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process raw iNaturalist observation data."""
        results = []
        
        for observation in data.get('results', []):
            processed_obs = {
                'id': observation.get('id'),
                'species_guess': observation.get('species_guess'),
                'taxon': observation.get('taxon', {}),
                'location': observation.get('place_guess'),
                'observed_on': observation.get('observed_on'),
                'quality_grade': observation.get('quality_grade'),
                'photos': [photo.get('url') for photo in observation.get('photos', [])],
                'identifications_count': observation.get('identifications_count', 0),
                'community_taxon': observation.get('community_taxon', {}),
                'geojson': observation.get('geojson')
            }
            results.append(processed_obs)
        
        return {
            'total_results': data.get('total_results', 0),
            'results': results,
            'source': 'iNaturalist'
        }
    
    def _process_taxon_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process raw iNaturalist taxon data."""
        taxon = data.get('results', [{}])[0] if data.get('results') else {}
        
        return {
            'id': taxon.get('id'),
            'name': taxon.get('name'),
            'rank': taxon.get('rank'),
            'common_name': taxon.get('preferred_common_name'),
            'wikipedia_url': taxon.get('wikipedia_url'),
            'taxonomy': {
                'kingdom': taxon.get('kingdom'),
                'phylum': taxon.get('phylum'),
                'class': taxon.get('taxon_class'),
                'order': taxon.get('order'),
                'family': taxon.get('family'),
                'genus': taxon.get('genus')
            },
            'conservation_status': taxon.get('conservation_status'),
            'source': 'iNaturalist'
        }


class GBIFAPI:
    """Integration with GBIF (Global Biodiversity Information Facility)."""
    
    def __init__(self):
        self.base_url = "https://api.gbif.org/v1"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'MycologyResearchPipeline/1.0',
            'Accept': 'application/json'
        })
    
    def search_species(self, species_name: str, limit: int = 20) -> Dict[str, Any]:
        """
        Search for species in GBIF database.
        
        Args:
            species_name: Scientific name
            limit: Maximum number of results
            
        Returns:
            Dict containing occurrence data
        """
        try:
            # First get species key
            species_key = self._get_species_key(species_name)
            if not species_key:
                return {'error': 'Species not found', 'results': []}
            
            # Then get occurrences
            endpoint = f"{self.base_url}/occurrence/search"
            params = {
                'taxonKey': species_key,
                'hasCoordinate': 'true',
                'hasGeospatialIssue': 'false',
                'limit': limit,
                'kingdom': 'Fungi'
            }
            
            response = self.session.get(endpoint, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            return self._process_gbif_data(data, species_name)
            
        except requests.exceptions.RequestException as e:
            logger.error(f"GBIF API error: {e}")
            return {'error': str(e), 'results': []}
    
    def _get_species_key(self, species_name: str) -> Optional[int]:
        """Get GBIF species key for a given species name."""
        try:
            endpoint = f"{self.base_url}/species/match"
            params = {'name': species_name, 'kingdom': 'Fungi'}
            
            response = self.session.get(endpoint, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            return data.get('usageKey')
            
        except requests.exceptions.RequestException:
            return None
    
    def _process_gbif_data(self, data: Dict[str, Any], species_name: str) -> Dict[str, Any]:
        """Process raw GBIF occurrence data."""
        results = []
        
        for occurrence in data.get('results', []):
            processed_occ = {
                'key': occurrence.get('key'),
                'species': occurrence.get('species'),
                'scientific_name': occurrence.get('scientificName'),
                'country': occurrence.get('country'),
                'locality': occurrence.get('locality'),
                'latitude': occurrence.get('decimalLatitude'),
                'longitude': occurrence.get('decimalLongitude'),
                'event_date': occurrence.get('eventDate'),
                'basis_of_record': occurrence.get('basisOfRecord'),
                'institution_code': occurrence.get('institutionCode'),
                'collection_code': occurrence.get('collectionCode'),
                'catalog_number': occurrence.get('catalogNumber'),
                'recorded_by': occurrence.get('recordedBy'),
                'identified_by': occurrence.get('identifiedBy')
            }
            results.append(processed_occ)
        
        return {
            'count': data.get('count', 0),
            'results': results,
            'source': 'GBIF'
        }


class MycologyDatabases:
    """Integration with specialized mycology databases."""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'MycologyResearchPipeline/1.0',
            'Accept': 'application/json'
        })
    
    def search_mycobank(self, species_name: str) -> Dict[str, Any]:
        """
        Search MycoBank for taxonomic information.
        
        Args:
            species_name: Scientific name
            
        Returns:
            Dict containing MycoBank data
        """
        try:
            # MycoBank uses a different API structure
            # This is a simplified implementation for demonstration
            base_url = "https://www.mycobank.org/page/Name%20details%20page"
            
            # Note: MycoBank requires specific authentication for API access
            # This would need proper API credentials in production
            return {
                'species_name': species_name,
                'source': 'MycoBank',
                'note': 'Requires API credentials for full access',
                'web_url': f"https://www.mycobank.org/quicksearch.aspx?qs={species_name.replace(' ', '%20')}"
            }
            
        except Exception as e:
            logger.error(f"MycoBank search error: {e}")
            return {'error': str(e), 'source': 'MycoBank'}
    
    def search_index_fungorum(self, species_name: str) -> Dict[str, Any]:
        """
        Search Index Fungorum for nomenclatural information.
        
        Args:
            species_name: Scientific name
            
        Returns:
            Dict containing Index Fungorum data
        """
        try:
            # Index Fungorum API endpoint
            base_url = "http://www.indexfungorum.org/ixfwebservice/fungus.asmx"
            
            # Note: Index Fungorum has specific API requirements
            # This is a simplified implementation
            return {
                'species_name': species_name,
                'source': 'Index Fungorum',
                'note': 'Requires specific API setup for full access',
                'web_url': f"http://www.indexfungorum.org/names/names.asp?strgenus={species_name.split()[0] if ' ' in species_name else species_name}"
            }
            
        except Exception as e:
            logger.error(f"Index Fungorum search error: {e}")
            return {'error': str(e), 'source': 'Index Fungorum'}


class ScientificDataIntegrator:
    """Main class for integrating all scientific databases."""
    
    def __init__(self):
        self.inaturalist = iNaturalistAPI()
        self.gbif = GBIFAPI()
        self.mycology_dbs = MycologyDatabases()
    
    def comprehensive_species_search(self, species_name: str) -> Dict[str, Any]:
        """
        Search all databases for comprehensive species information.
        
        Args:
            species_name: Scientific or common name
            
        Returns:
            Dict containing aggregated results from all sources
        """
        logger.info(f"Starting comprehensive search for: {species_name}")
        
        results = {
            'species_name': species_name,
            'search_timestamp': datetime.now().isoformat(),
            'sources': {}
        }
        
        # Search iNaturalist
        logger.info("Searching iNaturalist...")
        inaturalist_data = self.inaturalist.search_species(species_name)
        results['sources']['inaturalist'] = inaturalist_data
        
        # Search GBIF
        logger.info("Searching GBIF...")
        gbif_data = self.gbif.search_species(species_name)
        results['sources']['gbif'] = gbif_data
        
        # Search MycoBank
        logger.info("Searching MycoBank...")
        mycobank_data = self.mycology_dbs.search_mycobank(species_name)
        results['sources']['mycobank'] = mycobank_data
        
        # Search Index Fungorum
        logger.info("Searching Index Fungorum...")
        index_fungorum_data = self.mycology_dbs.search_index_fungorum(species_name)
        results['sources']['index_fungorum'] = index_fungorum_data
        
        # Generate confidence score based on data quality
        results['confidence_analysis'] = self._analyze_confidence(results['sources'])
        
        return results
    
    def _analyze_confidence(self, sources: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze confidence based on cross-database validation."""
        confidence_factors = {
            'inaturalist_observations': len(sources.get('inaturalist', {}).get('results', [])),
            'gbif_occurrences': len(sources.get('gbif', {}).get('results', [])),
            'research_grade_observations': sum(1 for obs in sources.get('inaturalist', {}).get('results', []) 
                                             if obs.get('quality_grade') == 'research'),
            'museum_specimens': len([occ for occ in sources.get('gbif', {}).get('results', []) 
                                   if occ.get('basis_of_record') == 'PRESERVED_SPECIMEN'])
        }
        
        # Calculate overall confidence score (0-100)
        base_score = min(100, (
            confidence_factors['inaturalist_observations'] * 2 +
            confidence_factors['gbif_occurrences'] * 3 +
            confidence_factors['research_grade_observations'] * 5 +
            confidence_factors['museum_specimens'] * 10
        ))
        
        return {
            'overall_confidence': base_score,
            'factors': confidence_factors,
            'recommendation': self._get_confidence_recommendation(base_score)
        }
    
    def _get_confidence_recommendation(self, score: int) -> str:
        """Get recommendation based on confidence score."""
        if score >= 80:
            return "High confidence - Multiple verified sources confirm identification"
        elif score >= 50:
            return "Moderate confidence - Some verification available, recommend expert review"
        elif score >= 20:
            return "Low confidence - Limited data available, requires expert identification"
        else:
            return "Very low confidence - Insufficient data, manual verification required"


# Utility functions for easy integration
def search_all_databases(species_name: str) -> Dict[str, Any]:
    """
    Convenient function to search all scientific databases.
    
    Args:
        species_name: Scientific or common name
        
    Returns:
        Comprehensive search results
    """
    integrator = ScientificDataIntegrator()
    return integrator.comprehensive_species_search(species_name)


def get_species_confidence(species_name: str) -> int:
    """
    Get confidence score for species identification.
    
    Args:
        species_name: Scientific name
        
    Returns:
        Confidence score (0-100)
    """
    results = search_all_databases(species_name)
    return results.get('confidence_analysis', {}).get('overall_confidence', 0)