"""
PubMed integration for the Mycology Research Pipeline.

This module handles communication with NCBI's PubMed API to retrieve
scientific literature relevant to mycological research.
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional

import pandas as pd
from Bio import Entrez

from models import LiteratureReference, Sample

logger = logging.getLogger(__name__)

# Default email for Entrez API
DEFAULT_EMAIL = "admin@mycology-research-pipeline.org"

# Default search queries for common species
SPECIES_KEYWORDS = {
    "Hericium erinaceus": "Hericium erinaceus AND (neuroprotective OR cognition)",
    "Ganoderma lucidum": "Ganoderma lucidum AND (immune OR cancer)",
    "Lentinula edodes": "Lentinula edodes AND (immunity OR cholesterol)",
    "Cordyceps militaris": "Cordyceps militaris AND (energy OR mitochondria)",
    "Trametes versicolor": "Trametes versicolor AND (cancer OR PSK)",
    "Psilocybe cubensis": "Psilocybin AND (depression OR therapy)",
    "Agaricus bisporus": "Agaricus bisporus AND (nutrition OR immune)",
    "Pleurotus ostreatus": "Pleurotus ostreatus AND (cholesterol OR antioxidant)",
    "Inonotus obliquus": "Inonotus obliquus AND (chaga OR antioxidant)",
    "Auricularia auricula": "Auricularia auricula AND (blood OR cardiovascular)"
}


def initialize_entrez(email: Optional[str] = None) -> None:
    """
    Initialize the Entrez API with user email.
    
    Args:
        email: Email address to use for NCBI API (required by their terms of service)
    """
    # Use provided email, environment variable, or default
    email = email or os.environ.get("ENTREZ_EMAIL", DEFAULT_EMAIL)
    Entrez.email = email
    logger.info(f"Initialized Entrez API with email: {email}")


def fetch_pubmed_articles(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """
    Fetch articles from PubMed based on a search query.
    
    Args:
        query: PubMed search query string
        max_results: Maximum number of results to return
        
    Returns:
        List of article summaries as dictionaries
    """
    logger.info(f"Fetching PubMed articles with query: {query}")
    
    try:
        # Search PubMed for article IDs
        handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results)
        record = Entrez.read(handle)
        handle.close()
        
        ids = record["IdList"]
        logger.debug(f"Found {len(ids)} PubMed IDs: {ids}")
        
        if not ids:
            logger.warning(f"No results found for query: {query}")
            return []
        
        # Fetch article summaries
        summaries = []
        for pmid in ids:
            summary = Entrez.esummary(db="pubmed", id=pmid)
            doc = Entrez.read(summary)[0]
            summary.close()
            
            # Extract article information
            authors = doc.get("AuthorList", [])
            author_str = ", ".join(authors[:3])
            if len(authors) > 3:
                author_str += f" et al. ({len(authors)} authors)"
                
            pub_date = doc.get("PubDate", "").split(" ")[0]
            try:
                year = int(pub_date)
            except (ValueError, TypeError):
                year = None
                
            summaries.append({
                "pmid": pmid,
                "title": doc.get("Title", "Unknown Title"),
                "authors": author_str,
                "journal": doc.get("FullJournalName", "Unknown Journal"),
                "year": year,
                "pub_date": pub_date,
                "doi": doc.get("DOI", None),
                "abstract": "", # Abstract requires a separate fetch
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
            })
        
        logger.info(f"Successfully fetched {len(summaries)} article summaries")
        return summaries
        
    except Exception as e:
        logger.error(f"Error fetching PubMed articles: {str(e)}")
        return []


def fetch_species_literature(species: str, custom_query: Optional[str] = None, 
                            max_results: int = 5) -> List[Dict[str, Any]]:
    """
    Fetch literature for a specific fungal species.
    
    Args:
        species: Scientific name of the fungal species
        custom_query: Optional custom query to use instead of default
        max_results: Maximum number of results to return
        
    Returns:
        List of article summaries
    """
    # Use custom query if provided, otherwise look up in defaults or build simple query
    if custom_query:
        query = custom_query
    elif species in SPECIES_KEYWORDS:
        query = SPECIES_KEYWORDS[species]
    else:
        # Create a basic query for any species not in our predefined list
        query = f"{species} AND (mycology OR medicinal OR mushroom)"
    
    return fetch_pubmed_articles(query, max_results)


def fetch_and_save_literature(species_list: Optional[List[str]] = None, 
                             output_file: str = "mycology_literature.csv") -> pd.DataFrame:
    """
    Fetch literature for multiple species and save to CSV.
    
    Args:
        species_list: List of species to fetch literature for
        output_file: Path to save the output CSV
        
    Returns:
        DataFrame with all fetched literature
    """
    # If no species list is provided, use all predefined species
    if species_list is None:
        species_list = list(SPECIES_KEYWORDS.keys())
    
    # Initialize Entrez
    initialize_entrez()
    
    all_entries = []
    for species in species_list:
        results = fetch_species_literature(species)
        for r in results:
            r["species"] = species
            all_entries.append(r)
    
    # Create and save DataFrame
    df = pd.DataFrame(all_entries)
    df.to_csv(output_file, index=False)
    
    logger.info(f"Saved {len(all_entries)} literature references to {output_file}")
    return df


def add_reference_to_database(db, species_id: int, pmid: str, title: str, 
                             authors: str, journal: str, year: Optional[int], 
                             url: str, abstract: Optional[str] = None) -> LiteratureReference:
    """
    Add a literature reference to the database.
    
    Args:
        db: SQLAlchemy database instance
        species_id: ID of the associated sample
        pmid: PubMed ID
        title: Article title
        authors: Article authors
        journal: Journal name
        year: Publication year
        url: URL to the article
        abstract: Article abstract (optional)
        
    Returns:
        Created LiteratureReference object
    """
    reference = LiteratureReference(
        sample_id=species_id,
        reference_id=pmid,
        title=title,
        authors=authors,
        journal=journal,
        year=year,
        url=url,
        abstract=abstract,
        reference_type="pubmed",
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow()
    )
    
    db.session.add(reference)
    db.session.commit()
    
    logger.info(f"Added literature reference to database: {pmid} - {title}")
    return reference


def update_sample_literature(db, sample_id: int, max_results: int = 5) -> List[LiteratureReference]:
    """
    Update literature references for a specific sample.
    
    Args:
        db: SQLAlchemy database instance
        sample_id: ID of the sample to update
        max_results: Maximum number of references to add
        
    Returns:
        List of created or updated LiteratureReference objects
    """
    # Get the sample
    sample = db.session.query(Sample).get(sample_id)
    if not sample:
        logger.error(f"Sample with ID {sample_id} not found")
        return []
    
    # Fetch literature for the sample's species
    if not sample.species:
        logger.warning(f"Sample {sample_id} has no species information")
        return []
    
    # Initialize Entrez
    initialize_entrez()
    
    # Fetch articles
    articles = fetch_species_literature(sample.species, max_results=max_results)
    
    # Add to database
    references = []
    for article in articles:
        # Check if reference already exists
        existing = db.session.query(LiteratureReference).filter_by(
            sample_id=sample_id, 
            reference_id=article["pmid"]
        ).first()
        
        if existing:
            logger.debug(f"Reference {article['pmid']} already exists for sample {sample_id}")
            references.append(existing)
            continue
        
        # Add new reference
        reference = add_reference_to_database(
            db,
            sample_id,
            article["pmid"],
            article["title"],
            article["authors"],
            article["journal"],
            article["year"],
            article["url"],
            article.get("abstract")
        )
        references.append(reference)
    
    return references


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    initialize_entrez()
    
    # Fetch literature for Reishi and Lion's Mane
    df = fetch_and_save_literature(
        species_list=["Ganoderma lucidum", "Hericium erinaceus"],
        output_file="mycology_literature_example.csv"
    )
    
    print(f"Fetched {len(df)} articles")
    print(df.head())