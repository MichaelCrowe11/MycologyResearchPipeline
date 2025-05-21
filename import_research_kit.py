#!/usr/bin/env python3
"""
CMID Research Intelligence Kit Importer

This script imports data from the CMID Research Intelligence Kit into the
Mycology Research Pipeline database.
"""

import argparse
import csv
import json
import os
import sys
from datetime import datetime

from app import create_app, db
from models import Sample, Compound, LiteratureReference
from literature import initialize_entrez

def import_species(filepath, verbose=False):
    """Import species data from CSV file."""
    if verbose:
        print(f"Importing species data from: {filepath}")
    
    imported = 0
    updated = 0
    
    with open(filepath, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            common_name = row.get('Common Name')
            scientific_name = row.get('Scientific Name')
            category = row.get('Category')
            
            if not scientific_name:
                continue
            
            # Check if sample with this species already exists
            existing = Sample.query.filter_by(species=scientific_name).first()
            
            if existing:
                # Update existing sample
                if common_name and not existing.name:
                    existing.name = common_name
                
                if category:
                    # Store category in metadata
                    metadata = existing.sample_metadata or {}
                    metadata['category'] = category
                    existing.sample_metadata = metadata
                
                db.session.add(existing)
                updated += 1
                if verbose:
                    print(f"Updated: {scientific_name} ({common_name})")
            else:
                # Create new sample
                new_sample = Sample(
                    name=common_name or scientific_name,
                    species=scientific_name,
                    description=f"Imported from CMID Research Kit: {common_name or scientific_name}",
                    sample_metadata={"category": category, "source": "CMID Research Kit"},
                    created_at=datetime.utcnow(),
                    updated_at=datetime.utcnow()
                )
                db.session.add(new_sample)
                imported += 1
                if verbose:
                    print(f"Imported: {scientific_name} ({common_name})")
    
    db.session.commit()
    
    return {
        'imported': imported,
        'updated': updated,
        'total': imported + updated
    }


def import_compounds(filepath, verbose=False):
    """Import compound data from CSV file."""
    if verbose:
        print(f"Importing compound data from: {filepath}")
    
    imported = 0
    skipped = 0
    
    with open(filepath, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            compound_name = row.get('Compound Name')
            species_name = row.get('Associated Species')
            iupac_name = row.get('IUPAC Name')
            smiles = row.get('SMILES')
            target = row.get('Pharmacological Target')
            
            if not compound_name or not species_name:
                skipped += 1
                continue
            
            # Find sample with this species
            sample = Sample.query.filter_by(species=species_name).first()
            
            if not sample:
                if verbose:
                    print(f"Sample not found for species: {species_name}, skipping compound: {compound_name}")
                skipped += 1
                continue
            
            # Check if compound already exists
            existing = Compound.query.filter_by(sample_id=sample.id, name=compound_name).first()
            
            if existing:
                if verbose:
                    print(f"Compound already exists: {compound_name} for {species_name}, skipping")
                skipped += 1
                continue
            
            # Create new compound
            metadata = {
                "iupac_name": iupac_name,
                "smiles": smiles,
                "pharmacological_target": target,
                "source": "CMID Research Kit"
            }
            
            new_compound = Compound(
                sample_id=sample.id,
                name=compound_name,
                formula=None,  # Not available in the data
                compound_metadata=metadata,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            db.session.add(new_compound)
            imported += 1
            
            if verbose:
                print(f"Imported compound: {compound_name} for {species_name}")
    
    db.session.commit()
    
    return {
        'imported': imported,
        'skipped': skipped,
        'total': imported + skipped
    }


def import_citations(filepath, verbose=False):
    """Import citation data from CSV file."""
    if verbose:
        print(f"Importing citation data from: {filepath}")
    
    imported = 0
    skipped = 0
    
    with open(filepath, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            species_name = row.get('Species')
            pmid = row.get('PMID')
            title = row.get('Title')
            authors = row.get('Authors')
            year = row.get('Year')
            journal = row.get('Journal')
            doi = row.get('DOI')
            
            if not species_name or not pmid or not title:
                skipped += 1
                continue
            
            # Find sample with this species
            sample = Sample.query.filter_by(species=species_name).first()
            
            if not sample:
                if verbose:
                    print(f"Sample not found for species: {species_name}, skipping citation: {pmid}")
                skipped += 1
                continue
            
            # Check if reference already exists
            existing = LiteratureReference.query.filter_by(reference_id=pmid, reference_type='pubmed').first()
            
            if existing:
                if verbose:
                    print(f"Citation already exists: {pmid} - {title}, skipping")
                skipped += 1
                continue
            
            # Convert year to integer if possible
            try:
                year_int = int(year) if year else None
            except (ValueError, TypeError):
                year_int = None
            
            # Create citation URL
            url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else None
            
            # Create new reference
            new_reference = LiteratureReference(
                sample_id=sample.id,
                reference_id=pmid,
                title=title,
                authors=authors,
                journal=journal,
                year=year_int,
                url=url,
                reference_type='pubmed',
                reference_metadata={"doi": doi, "source": "CMID Research Kit"},
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            db.session.add(new_reference)
            imported += 1
            
            if verbose:
                print(f"Imported citation: {pmid} - {title}")
    
    db.session.commit()
    
    return {
        'imported': imported,
        'skipped': skipped,
        'total': imported + skipped
    }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Import CMID Research Intelligence Kit data")
    parser.add_argument("-d", "--data-dir", default="research_kit",
                        help="Directory containing the research kit files")
    parser.add_argument("-s", "--species-file", default="CMID_v2_Expanded_Species.csv",
                        help="Species data CSV file name")
    parser.add_argument("-c", "--citations-file", default="CMID_v2_PubMed_Citations.csv",
                        help="Citations data CSV file name")
    parser.add_argument("-m", "--compounds-file", default="CMID_v2_Compound_Structure_Matrix.csv",
                        help="Compounds data CSV file name")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Enable verbose output")
    parser.add_argument("--skip-species", action="store_true",
                        help="Skip species import")
    parser.add_argument("--skip-compounds", action="store_true",
                        help="Skip compounds import")
    parser.add_argument("--skip-citations", action="store_true",
                        help="Skip citations import")
    
    args = parser.parse_args()
    
    # Make paths absolute
    data_dir = os.path.abspath(args.data_dir)
    species_file = os.path.join(data_dir, args.species_file)
    citations_file = os.path.join(data_dir, args.citations_file)
    compounds_file = os.path.join(data_dir, args.compounds_file)
    
    # Check if files exist
    if not os.path.exists(species_file) and not args.skip_species:
        print(f"Error: Species file not found: {species_file}")
        return 1
    
    if not os.path.exists(citations_file) and not args.skip_citations:
        print(f"Error: Citations file not found: {citations_file}")
        return 1
    
    if not os.path.exists(compounds_file) and not args.skip_compounds:
        print(f"Error: Compounds file not found: {compounds_file}")
        return 1
    
    # Create Flask app context
    app = create_app()
    with app.app_context():
        # Initialize Entrez for PubMed API access (may be needed for related operations)
        initialize_entrez()
        
        results = {}
        
        # Import species
        if not args.skip_species:
            print("\n=== Importing Species Data ===")
            species_results = import_species(species_file, args.verbose)
            results['species'] = species_results
            print(f"Species Import: {species_results['imported']} imported, {species_results['updated']} updated")
        
        # Import compounds
        if not args.skip_compounds:
            print("\n=== Importing Compound Data ===")
            compound_results = import_compounds(compounds_file, args.verbose)
            results['compounds'] = compound_results
            print(f"Compound Import: {compound_results['imported']} imported, {compound_results['skipped']} skipped")
        
        # Import citations
        if not args.skip_citations:
            print("\n=== Importing Citation Data ===")
            citation_results = import_citations(citations_file, args.verbose)
            results['citations'] = citation_results
            print(f"Citation Import: {citation_results['imported']} imported, {citation_results['skipped']} skipped")
        
        print("\n=== Import Complete ===")
        print(json.dumps(results, indent=2))
    
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)