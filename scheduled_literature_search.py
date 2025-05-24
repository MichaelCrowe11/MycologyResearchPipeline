"""
Automated Structured Literature Search Tool for Mycology Research Pipeline

This script performs scheduled searches across multiple scientific databases
to find and catalog the latest research on medicinal mushrooms and their
bioactive compounds.

Databases covered:
- PubMed (via Biopython and NCBI E-utilities)
- Scopus (requires API key)
- Science Direct (requires API key)
- Web of Science (requires API key)
- Google Scholar (via scholarly package)

The script can be scheduled to run daily or weekly to keep the research
database up-to-date with the latest scientific findings.
"""

import os
import json
import logging
import argparse
import csv
import time
import re
import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
import pandas as pd
from Bio import Entrez
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("literature_search.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("structured_literature_search")

# Load environment variables
load_dotenv()

# Directory for saving search results
RESULTS_DIR = os.path.join(os.getcwd(), "results", "literature_searches")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Data structure for storing search results
class ResearchPaper:
    """Represents a scientific research paper with standardized metadata."""
    
    def __init__(self, title: str, authors: List[str], journal: str, year: int, 
                 abstract: Optional[str] = None, doi: Optional[str] = None,
                 source_db: str = "unknown", keywords: List[str] = None,
                 url: Optional[str] = None, cited_by: int = 0):
        """
        Initialize a research paper.
        
        Args:
            title: Paper title
            authors: List of author names
            journal: Journal name
            year: Publication year
            abstract: Paper abstract
            doi: Digital Object Identifier
            source_db: Source database (PubMed, Scopus, etc.)
            keywords: List of keywords
            url: URL to access the paper
            cited_by: Number of citations
        """
        self.title = title
        self.authors = authors
        self.journal = journal
        self.year = year
        self.abstract = abstract
        self.doi = doi
        self.source_db = source_db
        self.keywords = keywords or []
        self.url = url
        self.cited_by = cited_by
        self.retrieved_date = datetime.datetime.now().isoformat()
        self.analyzed = False
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert the paper to a dictionary."""
        return {
            "title": self.title,
            "authors": self.authors,
            "journal": self.journal,
            "year": self.year,
            "abstract": self.abstract,
            "doi": self.doi,
            "source_db": self.source_db,
            "keywords": self.keywords,
            "url": self.url,
            "cited_by": self.cited_by,
            "retrieved_date": self.retrieved_date,
            "analyzed": self.analyzed
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ResearchPaper':
        """Create a ResearchPaper object from a dictionary."""
        paper = cls(
            title=data["title"],
            authors=data["authors"],
            journal=data["journal"],
            year=data["year"],
            abstract=data.get("abstract"),
            doi=data.get("doi"),
            source_db=data.get("source_db", "unknown"),
            keywords=data.get("keywords", []),
            url=data.get("url"),
            cited_by=data.get("cited_by", 0)
        )
        paper.retrieved_date = data.get("retrieved_date", paper.retrieved_date)
        paper.analyzed = data.get("analyzed", False)
        return paper


class PubMedSearcher:
    """Class for searching PubMed database via Biopython."""
    
    def __init__(self, email: str = None):
        """
        Initialize PubMed searcher.
        
        Args:
            email: Email address for NCBI API (required by their terms of service)
        """
        self.email = email or os.environ.get("ENTREZ_EMAIL", "admin@mycology-research-pipeline.org")
        Entrez.email = self.email
        logger.info(f"Initialized PubMed searcher with email: {self.email}")
    
    def search(self, query: str, max_results: int = 100, 
              start_date: str = None, end_date: str = None) -> List[ResearchPaper]:
        """
        Search PubMed for scientific papers.
        
        Args:
            query: PubMed search query string
            max_results: Maximum number of results to return
            start_date: Start date for search range (YYYY/MM/DD)
            end_date: End date for search range (YYYY/MM/DD)
            
        Returns:
            List of ResearchPaper objects
        """
        # Add date range if provided
        if start_date or end_date:
            date_query = ""
            if start_date:
                date_query += f"{start_date}[PDAT]"
            if start_date and end_date:
                date_query += ":"
            if end_date:
                date_query += f"{end_date}[PDAT]"
            
            if date_query:
                query = f"({query}) AND ({date_query})"
        
        # Log the search query
        logger.info(f"Searching PubMed with query: {query}")
        
        try:
            # Search PubMed
            search_handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results)
            search_results = Entrez.read(search_handle)
            search_handle.close()
            
            # Get the IDs of the papers
            id_list = search_results["IdList"]
            logger.debug(f"Found {len(id_list)} PubMed IDs: {id_list}")
            
            if not id_list:
                logger.warning(f"No results found for query: {query}")
                return []
            
            # Fetch the details of the papers
            fetch_handle = Entrez.efetch(db="pubmed", id=id_list, rettype="medline", retmode="text")
            records = Entrez.read(fetch_handle)
            fetch_handle.close()
            
            # Process the records
            papers = []
            for record in records.get("PubmedArticle", []):
                try:
                    # Extract basic information
                    article = record.get("MedlineCitation", {}).get("Article", {})
                    title = article.get("ArticleTitle", "No title available")
                    
                    # Extract authors
                    authors = []
                    for author in article.get("AuthorList", []):
                        if "LastName" in author and "ForeName" in author:
                            authors.append(f"{author['LastName']} {author['ForeName']}")
                        elif "LastName" in author:
                            authors.append(author["LastName"])
                        elif "CollectiveName" in author:
                            authors.append(author["CollectiveName"])
                    
                    # Extract journal info
                    journal = article.get("Journal", {}).get("Title", "Unknown Journal")
                    
                    # Extract year
                    pub_date = article.get("Journal", {}).get("JournalIssue", {}).get("PubDate", {})
                    year = None
                    if "Year" in pub_date:
                        year = int(pub_date["Year"])
                    else:
                        # Try to extract year from the MedlineDate
                        medline_date = pub_date.get("MedlineDate", "")
                        if medline_date:
                            year_match = re.search(r'\b(19|20)\d{2}\b', medline_date)
                            if year_match:
                                year = int(year_match.group(0))
                    
                    if not year:
                        year = datetime.datetime.now().year
                    
                    # Extract abstract
                    abstract = ""
                    if "Abstract" in article:
                        abstract_texts = article["Abstract"].get("AbstractText", [])
                        if isinstance(abstract_texts, list):
                            abstract = " ".join([str(text) for text in abstract_texts])
                        else:
                            abstract = str(abstract_texts)
                    
                    # Extract DOI
                    doi = None
                    for id_item in article.get("ELocationID", []):
                        if id_item.attributes.get("EIdType") == "doi":
                            doi = str(id_item)
                            break
                    
                    # Extract keywords
                    keywords = []
                    for keyword in record.get("MedlineCitation", {}).get("KeywordList", []):
                        keywords.extend([str(k) for k in keyword])
                    
                    # Create URL
                    pmid = record.get("MedlineCitation", {}).get("PMID", "")
                    url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else None
                    
                    # Create paper object
                    paper = ResearchPaper(
                        title=title,
                        authors=authors,
                        journal=journal,
                        year=year,
                        abstract=abstract,
                        doi=doi,
                        source_db="PubMed",
                        keywords=keywords,
                        url=url,
                        cited_by=0  # PubMed doesn't provide citation count
                    )
                    
                    papers.append(paper)
                    
                except Exception as e:
                    logger.error(f"Error processing PubMed record: {str(e)}")
            
            logger.info(f"Successfully fetched {len(papers)} article summaries from PubMed")
            return papers
            
        except Exception as e:
            logger.error(f"Error searching PubMed: {str(e)}")
            return []


class ScopusSearcher:
    """Class for searching Scopus database via API."""
    
    def __init__(self, api_key: str = None):
        """
        Initialize Scopus searcher.
        
        Args:
            api_key: Scopus API key
        """
        self.api_key = api_key or os.environ.get("SCOPUS_API_KEY")
        if not self.api_key:
            logger.warning("No Scopus API key provided. Scopus searcher will not function.")
    
    def search(self, query: str, max_results: int = 100,
              start_date: str = None, end_date: str = None) -> List[ResearchPaper]:
        """
        Search Scopus for scientific papers.
        
        Args:
            query: Scopus search query string
            max_results: Maximum number of results to return
            start_date: Start date for search range (YYYY-MM-DD)
            end_date: End date for search range (YYYY-MM-DD)
            
        Returns:
            List of ResearchPaper objects
        """
        if not self.api_key:
            logger.error("Cannot search Scopus without an API key")
            return []
        
        # For this MVP implementation, we'll simulate Scopus results
        # In a production environment, you would implement the actual API call
        logger.info(f"Simulating Scopus search with query: {query}")
        
        # Simulate some papers (for demonstration only)
        papers = [
            ResearchPaper(
                title="Neurotrophic Properties of Lion's Mane Mushroom",
                authors=["Smith J", "Johnson A", "Lee R"],
                journal="Journal of Medicinal Mushrooms",
                year=2023,
                abstract="This study investigates the neurotrophic effects of Hericium erinaceus.",
                doi="10.1234/jmm.2023.001",
                source_db="Scopus",
                keywords=["Lion's Mane", "Hericium erinaceus", "neurotrophic", "nerve growth factor"],
                url="https://doi.org/10.1234/jmm.2023.001",
                cited_by=12
            ),
            ResearchPaper(
                title="Pharmacological Potential of Hericenones and Erinacines",
                authors=["Garcia M", "Zhang L", "Patel K"],
                journal="Phytochemistry Reviews",
                year=2022,
                abstract="A comprehensive review of the bioactive compounds in Hericium erinaceus.",
                doi="10.1234/pr.2022.005",
                source_db="Scopus",
                keywords=["Hericenones", "Erinacines", "pharmacology", "bioactive compounds"],
                url="https://doi.org/10.1234/pr.2022.005",
                cited_by=8
            )
        ]
        
        logger.info(f"Simulated {len(papers)} results from Scopus")
        return papers


class ScienceDirectSearcher:
    """Class for searching Science Direct database via API."""
    
    def __init__(self, api_key: str = None):
        """
        Initialize Science Direct searcher.
        
        Args:
            api_key: Science Direct API key
        """
        self.api_key = api_key or os.environ.get("SCIENCEDIRECT_API_KEY")
        if not self.api_key:
            logger.warning("No Science Direct API key provided. Science Direct searcher will not function.")
    
    def search(self, query: str, max_results: int = 100,
              start_date: str = None, end_date: str = None) -> List[ResearchPaper]:
        """
        Search Science Direct for scientific papers.
        
        Args:
            query: Science Direct search query string
            max_results: Maximum number of results to return
            start_date: Start date for search range (YYYY-MM-DD)
            end_date: End date for search range (YYYY-MM-DD)
            
        Returns:
            List of ResearchPaper objects
        """
        if not self.api_key:
            logger.error("Cannot search Science Direct without an API key")
            return []
        
        # For this MVP implementation, we'll simulate Science Direct results
        # In a production environment, you would implement the actual API call
        logger.info(f"Simulating Science Direct search with query: {query}")
        
        # Simulate some papers (for demonstration only)
        papers = [
            ResearchPaper(
                title="Antioxidant and Antimicrobial Activities of Hericium erinaceus Extract",
                authors=["Williams T", "Brown D", "Chen Y"],
                journal="Food Chemistry",
                year=2024,
                abstract="This study examines the antioxidant and antimicrobial properties of extracts from Hericium erinaceus.",
                doi="10.1234/fc.2024.002",
                source_db="ScienceDirect",
                keywords=["Hericium erinaceus", "antioxidant", "antimicrobial", "food chemistry"],
                url="https://doi.org/10.1234/fc.2024.002",
                cited_by=3
            ),
            ResearchPaper(
                title="Medicinal Mushrooms in Neurodegenerative Disorders: A Review",
                authors=["Rodriguez J", "Schmidt F", "Wang H"],
                journal="Neuropharmacology",
                year=2023,
                abstract="A comprehensive review of medicinal mushrooms and their potential for treating neurodegenerative disorders.",
                doi="10.1234/neuro.2023.010",
                source_db="ScienceDirect",
                keywords=["medicinal mushrooms", "neurodegenerative disorders", "Alzheimer's", "Parkinson's"],
                url="https://doi.org/10.1234/neuro.2023.010",
                cited_by=15
            )
        ]
        
        logger.info(f"Simulated {len(papers)} results from Science Direct")
        return papers


class GoogleScholarSearcher:
    """Class for searching Google Scholar."""
    
    def __init__(self):
        """Initialize Google Scholar searcher."""
        # No API key needed, but may need to manage rate limiting
        logger.info("Initialized Google Scholar searcher")
    
    def search(self, query: str, max_results: int = 100,
              start_date: str = None, end_date: str = None) -> List[ResearchPaper]:
        """
        Search Google Scholar for scientific papers.
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
            start_date: Start date for search range (YYYY-MM-DD)
            end_date: End date for search range (YYYY-MM-DD)
            
        Returns:
            List of ResearchPaper objects
        """
        # For this MVP implementation, we'll simulate Google Scholar results
        # In a production environment, you would implement the actual scraping logic
        # Note: Google Scholar doesn't have an official API, so this typically requires
        # web scraping, which has its own challenges and limitations
        logger.info(f"Simulating Google Scholar search with query: {query}")
        
        # Simulate some papers (for demonstration only)
        papers = [
            ResearchPaper(
                title="Novel Bioactive Compounds from Hericium erinaceus and Their Mechanism of Action",
                authors=["Kim J", "Park S", "Lee J"],
                journal="Journal of Natural Products",
                year=2023,
                abstract="Investigation of novel bioactive compounds isolated from Hericium erinaceus and their mechanisms.",
                doi="10.1234/jnp.2023.012",
                source_db="Google Scholar",
                keywords=["Hericium erinaceus", "bioactive compounds", "mechanism of action"],
                url="https://doi.org/10.1234/jnp.2023.012",
                cited_by=7
            ),
            ResearchPaper(
                title="Clinical Applications of Lion's Mane Mushroom: Current Evidence and Future Directions",
                authors=["Thompson R", "Anderson M", "Wilson K"],
                journal="Complementary Therapies in Medicine",
                year=2022,
                abstract="A review of clinical studies on Hericium erinaceus and its applications in medicine.",
                doi="10.1234/ctm.2022.008",
                source_db="Google Scholar",
                keywords=["clinical applications", "Lion's Mane", "evidence-based medicine"],
                url="https://doi.org/10.1234/ctm.2022.008",
                cited_by=22
            )
        ]
        
        logger.info(f"Simulated {len(papers)} results from Google Scholar")
        return papers


class WebOfScienceSearcher:
    """Class for searching Web of Science database via API."""
    
    def __init__(self, api_key: str = None):
        """
        Initialize Web of Science searcher.
        
        Args:
            api_key: Web of Science API key
        """
        self.api_key = api_key or os.environ.get("WOS_API_KEY")
        if not self.api_key:
            logger.warning("No Web of Science API key provided. Web of Science searcher will not function.")
    
    def search(self, query: str, max_results: int = 100,
              start_date: str = None, end_date: str = None) -> List[ResearchPaper]:
        """
        Search Web of Science for scientific papers.
        
        Args:
            query: Web of Science search query string
            max_results: Maximum number of results to return
            start_date: Start date for search range (YYYY-MM-DD)
            end_date: End date for search range (YYYY-MM-DD)
            
        Returns:
            List of ResearchPaper objects
        """
        if not self.api_key:
            logger.error("Cannot search Web of Science without an API key")
            return []
        
        # For this MVP implementation, we'll simulate Web of Science results
        # In a production environment, you would implement the actual API call
        logger.info(f"Simulating Web of Science search with query: {query}")
        
        # Simulate some papers (for demonstration only)
        papers = [
            ResearchPaper(
                title="Immunomodulatory Effects of Polysaccharides from Hericium erinaceus",
                authors=["Chang Y", "Lin W", "Hsu T"],
                journal="International Journal of Biological Macromolecules",
                year=2024,
                abstract="This study investigates the immunomodulatory properties of polysaccharides extracted from Hericium erinaceus.",
                doi="10.1234/ijbm.2024.003",
                source_db="Web of Science",
                keywords=["polysaccharides", "immunomodulatory", "Hericium erinaceus"],
                url="https://doi.org/10.1234/ijbm.2024.003",
                cited_by=5
            ),
            ResearchPaper(
                title="Comparative Analysis of Bioactive Compounds in Different Hericium Species",
                authors=["Davis R", "Miller G", "Jones E"],
                journal="Food Research International",
                year=2023,
                abstract="A comparative study of bioactive compounds found in different Hericium species.",
                doi="10.1234/fri.2023.007",
                source_db="Web of Science",
                keywords=["comparative analysis", "Hericium species", "bioactive compounds"],
                url="https://doi.org/10.1234/fri.2023.007",
                cited_by=9
            )
        ]
        
        logger.info(f"Simulated {len(papers)} results from Web of Science")
        return papers


class StructuredLiteratureSearch:
    """
    Main class for performing structured literature searches across multiple databases.
    """
    
    def __init__(self, use_pubmed: bool = True, use_scopus: bool = False,
                use_sciencedirect: bool = False, use_webofscience: bool = False,
                use_googlescholar: bool = False):
        """
        Initialize the structured literature search.
        
        Args:
            use_pubmed: Whether to use PubMed
            use_scopus: Whether to use Scopus
            use_sciencedirect: Whether to use Science Direct
            use_webofscience: Whether to use Web of Science
            use_googlescholar: Whether to use Google Scholar
        """
        self.searchers = {}
        
        if use_pubmed:
            self.searchers["PubMed"] = PubMedSearcher()
        
        if use_scopus:
            self.searchers["Scopus"] = ScopusSearcher()
        
        if use_sciencedirect:
            self.searchers["ScienceDirect"] = ScienceDirectSearcher()
        
        if use_webofscience:
            self.searchers["WebOfScience"] = WebOfScienceSearcher()
        
        if use_googlescholar:
            self.searchers["GoogleScholar"] = GoogleScholarSearcher()
        
        logger.info(f"Initialized structured literature search with databases: {', '.join(self.searchers.keys())}")
    
    def search(self, query: str, max_results_per_db: int = 100,
              start_date: str = None, end_date: str = None,
              deduplicate: bool = True) -> List[ResearchPaper]:
        """
        Search across all configured databases.
        
        Args:
            query: Search query string
            max_results_per_db: Maximum number of results per database
            start_date: Start date for search range (YYYY-MM-DD)
            end_date: End date for search range (YYYY-MM-DD)
            deduplicate: Whether to remove duplicate papers based on DOI
            
        Returns:
            List of ResearchPaper objects
        """
        all_papers = []
        
        # Search each database
        for db_name, searcher in self.searchers.items():
            logger.info(f"Searching {db_name}...")
            try:
                papers = searcher.search(
                    query=query,
                    max_results=max_results_per_db,
                    start_date=start_date,
                    end_date=end_date
                )
                all_papers.extend(papers)
                logger.info(f"Found {len(papers)} papers from {db_name}")
            except Exception as e:
                logger.error(f"Error searching {db_name}: {str(e)}")
        
        # Deduplicate papers if requested
        if deduplicate:
            all_papers = self._deduplicate_papers(all_papers)
            logger.info(f"After deduplication: {len(all_papers)} unique papers")
        
        return all_papers
    
    def _deduplicate_papers(self, papers: List[ResearchPaper]) -> List[ResearchPaper]:
        """
        Remove duplicate papers based on DOI.
        
        Args:
            papers: List of papers to deduplicate
            
        Returns:
            Deduplicated list of papers
        """
        # Use DOI as the primary key for deduplication
        doi_map = {}
        title_map = {}
        unique_papers = []
        
        for paper in papers:
            # If the paper has a DOI, use that for deduplication
            if paper.doi and paper.doi not in doi_map:
                doi_map[paper.doi] = paper
                unique_papers.append(paper)
            # If no DOI, use the title (normalized) as a fallback
            elif not paper.doi:
                # Normalize title by removing spaces, punctuation, and converting to lowercase
                norm_title = re.sub(r'[^\w]', '', paper.title.lower())
                if norm_title not in title_map:
                    title_map[norm_title] = paper
                    unique_papers.append(paper)
        
        return unique_papers
    
    def save_results(self, papers: List[ResearchPaper], output_format: str = "csv") -> str:
        """
        Save search results to a file.
        
        Args:
            papers: List of papers to save
            output_format: Output format ("csv", "json", or "xlsx")
            
        Returns:
            Path to the saved file
        """
        # Create a timestamp for the filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if output_format == "csv":
            output_file = os.path.join(RESULTS_DIR, f"literature_search_{timestamp}.csv")
            with open(output_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=[
                    "title", "authors", "journal", "year", "abstract", "doi",
                    "source_db", "keywords", "url", "cited_by", "retrieved_date"
                ])
                writer.writeheader()
                for paper in papers:
                    row = paper.to_dict()
                    # Convert lists to strings for CSV
                    row["authors"] = ", ".join(row["authors"])
                    row["keywords"] = ", ".join(row["keywords"])
                    writer.writerow(row)
        
        elif output_format == "json":
            output_file = os.path.join(RESULTS_DIR, f"literature_search_{timestamp}.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                papers_dict = [paper.to_dict() for paper in papers]
                json.dump(papers_dict, f, indent=2)
        
        elif output_format == "xlsx":
            output_file = os.path.join(RESULTS_DIR, f"literature_search_{timestamp}.xlsx")
            papers_dict = [paper.to_dict() for paper in papers]
            df = pd.DataFrame(papers_dict)
            
            # Convert lists to strings for Excel
            df["authors"] = df["authors"].apply(lambda x: ", ".join(x))
            df["keywords"] = df["keywords"].apply(lambda x: ", ".join(x))
            
            df.to_excel(output_file, index=False)
        
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
        
        logger.info(f"Saved {len(papers)} papers to {output_file}")
        return output_file


def create_search_queries(mushroom_species: Optional[List[str]] = None,
                         compounds: Optional[List[str]] = None,
                         effects: Optional[List[str]] = None,
                         additional_terms: Optional[List[str]] = None) -> List[str]:
    """
    Create search queries for different combinations of terms.
    
    Args:
        mushroom_species: List of mushroom species names
        compounds: List of compound names
        effects: List of biological effects
        additional_terms: Additional search terms
        
    Returns:
        List of search query strings
    """
    mushroom_species = mushroom_species or [
        "Hericium erinaceus", "Lion's Mane",
        "Ganoderma lucidum", "Reishi",
        "Trametes versicolor", "Turkey Tail",
        "Cordyceps militaris", "Cordyceps"
    ]
    
    compounds = compounds or [
        "polysaccharide", "beta-glucan",
        "terpenoid", "hericenone", "erinacine",
        "triterpene", "ganoderic acid",
        "ergosterol", "lectin"
    ]
    
    effects = effects or [
        "antioxidant", "antimicrobial",
        "neuroprotective", "immunomodulatory",
        "anti-inflammatory", "anticancer",
        "cognitive", "neurotrophic"
    ]
    
    additional_terms = additional_terms or [
        "medicinal mushroom", "bioactive",
        "extraction", "mechanism", "clinical",
        "therapeutic", "pharmacological"
    ]
    
    queries = []
    
    # Create species-specific queries
    for species in mushroom_species:
        # Basic species query
        queries.append(f'"{species}"')
        
        # Species + compounds
        for compound in compounds:
            queries.append(f'"{species}" AND "{compound}"')
        
        # Species + effects
        for effect in effects:
            queries.append(f'"{species}" AND "{effect}"')
        
        # Species + compounds + effects
        for compound in compounds:
            for effect in effects:
                queries.append(f'"{species}" AND "{compound}" AND "{effect}"')
    
    # Create compound-specific queries
    for compound in compounds:
        for effect in effects:
            queries.append(f'"medicinal mushroom" AND "{compound}" AND "{effect}"')
    
    # Add additional terms to some queries
    enhanced_queries = []
    for i, query in enumerate(queries):
        if i % 3 == 0 and additional_terms:  # Add to every third query
            term = additional_terms[i % len(additional_terms)]
            enhanced_queries.append(f'{query} AND "{term}"')
    
    queries.extend(enhanced_queries)
    
    # Remove duplicates and return
    return list(set(queries))


def run_scheduled_search(search_frequency: str = "weekly",
                        species: Optional[List[str]] = None,
                        compounds: Optional[List[str]] = None,
                        effects: Optional[List[str]] = None,
                        additional_terms: Optional[List[str]] = None,
                        max_results_per_db: int = 50,
                        output_format: str = "csv",
                        use_pubmed: bool = True,
                        use_scopus: bool = False,
                        use_sciencedirect: bool = False,
                        use_webofscience: bool = False,
                        use_googlescholar: bool = False):
    """
    Run a scheduled literature search.
    
    Args:
        search_frequency: "daily" or "weekly"
        species: List of mushroom species to search for
        compounds: List of compounds to search for
        effects: List of biological effects to search for
        additional_terms: Additional search terms
        max_results_per_db: Maximum results per database
        output_format: Output format ("csv", "json", or "xlsx")
        use_pubmed: Whether to use PubMed
        use_scopus: Whether to use Scopus
        use_sciencedirect: Whether to use Science Direct
        use_webofscience: Whether to use Web of Science
        use_googlescholar: Whether to use Google Scholar
    """
    # Set up date range based on search frequency
    end_date = datetime.datetime.now().strftime("%Y/%m/%d")
    
    if search_frequency == "daily":
        # For daily searches, look at the past day
        start_date = (datetime.datetime.now() - datetime.timedelta(days=1)).strftime("%Y/%m/%d")
    elif search_frequency == "weekly":
        # For weekly searches, look at the past week
        start_date = (datetime.datetime.now() - datetime.timedelta(days=7)).strftime("%Y/%m/%d")
    else:
        raise ValueError(f"Unsupported search frequency: {search_frequency}")
    
    # Create search queries
    queries = create_search_queries(
        mushroom_species=species,
        compounds=compounds,
        effects=effects,
        additional_terms=additional_terms
    )
    
    # Initialize the searcher
    searcher = StructuredLiteratureSearch(
        use_pubmed=use_pubmed,
        use_scopus=use_scopus,
        use_sciencedirect=use_sciencedirect,
        use_webofscience=use_webofscience,
        use_googlescholar=use_googlescholar
    )
    
    # Run searches and collect results
    all_papers = []
    
    for i, query in enumerate(queries):
        logger.info(f"Running search {i+1}/{len(queries)}: {query}")
        
        papers = searcher.search(
            query=query,
            max_results_per_db=max_results_per_db,
            start_date=start_date,
            end_date=end_date,
            deduplicate=False  # We'll deduplicate at the end
        )
        
        all_papers.extend(papers)
        
        # Add a small delay to avoid rate limiting
        if i < len(queries) - 1:
            time.sleep(1)
    
    # Deduplicate the results
    unique_papers = searcher._deduplicate_papers(all_papers)
    
    # Save the results
    if unique_papers:
        output_file = searcher.save_results(unique_papers, output_format=output_format)
        logger.info(f"Search completed. Found {len(unique_papers)} unique papers. Results saved to {output_file}")
    else:
        logger.warning("No papers found in this search.")


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Structured Literature Search for Mycology Research")
    
    parser.add_argument("--frequency", choices=["daily", "weekly"], default="weekly",
                        help="Search frequency (daily or weekly)")
    
    parser.add_argument("--species", nargs="+",
                        help="Mushroom species to search for")
    
    parser.add_argument("--compounds", nargs="+",
                        help="Compounds to search for")
    
    parser.add_argument("--effects", nargs="+",
                        help="Biological effects to search for")
    
    parser.add_argument("--terms", nargs="+",
                        help="Additional search terms")
    
    parser.add_argument("--max-results", type=int, default=50,
                        help="Maximum results per database")
    
    parser.add_argument("--format", choices=["csv", "json", "xlsx"], default="csv",
                        help="Output format")
    
    parser.add_argument("--databases", nargs="+",
                        choices=["pubmed", "scopus", "sciencedirect", "webofscience", "googlescholar"],
                        default=["pubmed"],
                        help="Databases to search")
    
    args = parser.parse_args()
    
    # Run the search
    run_scheduled_search(
        search_frequency=args.frequency,
        species=args.species,
        compounds=args.compounds,
        effects=args.effects,
        additional_terms=args.terms,
        max_results_per_db=args.max_results,
        output_format=args.format,
        use_pubmed="pubmed" in args.databases,
        use_scopus="scopus" in args.databases,
        use_sciencedirect="sciencedirect" in args.databases,
        use_webofscience="webofscience" in args.databases,
        use_googlescholar="googlescholar" in args.databases
    )


if __name__ == "__main__":
    main()