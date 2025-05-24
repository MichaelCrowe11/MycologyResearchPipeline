#!/usr/bin/env python3
"""
Mycology Research Literature Fetching Tool

This standalone script demonstrates how to fetch scientific literature
related to medicinal mushrooms using the NCBI PubMed API via Biopython.
"""

import argparse
import csv
import json
import os
import sys
from datetime import datetime

from Bio import Entrez

# Default species and their research areas
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


def setup_entrez(email):
    """Set up Entrez email for NCBI API access."""
    Entrez.email = email
    print(f"PubMed access initialized with email: {email}")


def fetch_pubmed_articles(query, max_results=5):
    """
    Fetch articles from PubMed based on a search query.
    
    Args:
        query: PubMed search query string
        max_results: Maximum number of results to return
        
    Returns:
        List of article summaries as dictionaries
    """
    print(f"Searching PubMed for: {query}")
    print(f"Fetching up to {max_results} results...")
    
    try:
        # Search PubMed for article IDs
        handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results)
        record = Entrez.read(handle)
        handle.close()
        
        ids = record["IdList"]
        print(f"Found {len(ids)} PubMed IDs")
        
        if not ids:
            print("No results found")
            return []
        
        # Fetch article summaries
        summaries = []
        for pmid in ids:
            print(f"Fetching details for PMID: {pmid}...")
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
                
            article = {
                "pmid": pmid,
                "title": doc.get("Title", "Unknown Title"),
                "authors": author_str,
                "journal": doc.get("FullJournalName", "Unknown Journal"),
                "year": year,
                "pub_date": pub_date,
                "doi": doc.get("DOI", None),
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
            }
            summaries.append(article)
            print(f"Added: {article['title'][:60]}...")
        
        print(f"Successfully fetched {len(summaries)} article summaries")
        return summaries
        
    except Exception as e:
        print(f"Error fetching PubMed articles: {str(e)}")
        return []


def save_to_csv(articles, filename):
    """
    Save fetched articles to a CSV file.
    
    Args:
        articles: List of article dictionaries
        filename: Path to save the CSV file
    """
    if not articles:
        print("No articles to save")
        return
    
    print(f"Saving {len(articles)} articles to {filename}...")
    
    fieldnames = ["species", "pmid", "title", "authors", "journal", "year", "url"]
    
    try:
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for article in articles:
                writer.writerow({
                    field: article.get(field, "") for field in fieldnames
                })
        print(f"Successfully saved to {filename}")
    except Exception as e:
        print(f"Error saving to CSV: {str(e)}")


def save_to_json(articles, filename):
    """
    Save fetched articles to a JSON file.
    
    Args:
        articles: List of article dictionaries
        filename: Path to save the JSON file
    """
    if not articles:
        print("No articles to save")
        return
    
    print(f"Saving {len(articles)} articles to {filename}...")
    
    try:
        with open(filename, 'w', encoding='utf-8') as jsonfile:
            json.dump(articles, jsonfile, indent=2, ensure_ascii=False)
        print(f"Successfully saved to {filename}")
    except Exception as e:
        print(f"Error saving to JSON: {str(e)}")


def fetch_all_species(species_list=None, max_results=5):
    """
    Fetch literature for multiple species.
    
    Args:
        species_list: List of species to fetch literature for
        max_results: Maximum number of results per species
        
    Returns:
        List of article dictionaries
    """
    if species_list is None:
        species_list = list(SPECIES_KEYWORDS.keys())
    
    all_articles = []
    for species in species_list:
        print(f"\n{'='*60}")
        print(f"Processing species: {species}")
        print(f"{'='*60}")
        
        # Use predefined query or create a basic one
        if species in SPECIES_KEYWORDS:
            query = SPECIES_KEYWORDS[species]
            print(f"Using predefined query: {query}")
        else:
            query = f"{species} AND (mycology OR medicinal OR mushroom)"
            print(f"Using basic query: {query}")
        
        # Fetch articles
        articles = fetch_pubmed_articles(query, max_results)
        
        # Add species to each article
        for article in articles:
            article["species"] = species
        
        all_articles.extend(articles)
    
    return all_articles


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Fetch scientific literature for medicinal mushrooms")
    parser.add_argument("-e", "--email", required=True, 
                        help="Email address for NCBI API access (required)")
    parser.add_argument("-s", "--species", nargs="+", 
                        help="Species to search for (default: all predefined species)")
    parser.add_argument("-q", "--query", 
                        help="Custom PubMed query (overrides species search)")
    parser.add_argument("-m", "--max-results", type=int, default=5, 
                        help="Maximum number of results per species (default: 5)")
    parser.add_argument("-o", "--output", default="mycology_literature.csv", 
                        help="Output filename (default: mycology_literature.csv)")
    parser.add_argument("-j", "--json", action="store_true", 
                        help="Save as JSON instead of CSV")
    parser.add_argument("--show-species", action="store_true", 
                        help="Show list of predefined species and their search queries")
    
    args = parser.parse_args()
    
    # Show predefined species and exit if requested
    if args.show_species:
        print("Predefined species and their search queries:")
        for species, query in SPECIES_KEYWORDS.items():
            print(f"  - {species}: {query}")
        return
    
    # Initialize Entrez with email
    setup_entrez(args.email)
    
    # Fetch articles
    if args.query:
        print(f"Using custom query: {args.query}")
        articles = fetch_pubmed_articles(args.query, args.max_results)
        # Add a placeholder species for custom queries
        for article in articles:
            article["species"] = "Custom Query"
    else:
        articles = fetch_all_species(args.species, args.max_results)
    
    # Save results
    if args.json:
        output_file = args.output.replace('.csv', '.json') if args.output.endswith('.csv') else args.output
        save_to_json(articles, output_file)
    else:
        save_to_csv(articles, args.output)
    
    print(f"\nFetched a total of {len(articles)} articles for {len(set(a['species'] for a in articles))} species")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)