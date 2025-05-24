"""
Data Validation Module for the Mycology Research Pipeline.

This module provides functions for validating data integrity, ensuring
proper relationships between entities, and generating validation reports.
"""

import logging
import re
import os
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
from sqlalchemy import inspect, func
from datetime import datetime

from app import db
from models import Sample, Compound, Analysis, BatchJob, LiteratureReference

logger = logging.getLogger(__name__)

class ValidationResult:
    """Container for validation results."""
    
    def __init__(self):
        self.errors = []
        self.warnings = []
        self.suggestions = []
    
    @property
    def has_errors(self) -> bool:
        """Check if any errors were found."""
        return len(self.errors) > 0
    
    @property
    def has_warnings(self) -> bool:
        """Check if any warnings were found."""
        return len(self.warnings) > 0
    
    @property
    def has_suggestions(self) -> bool:
        """Check if any suggestions were found."""
        return len(self.suggestions) > 0
    
    @property
    def is_valid(self) -> bool:
        """Check if the validation passed with no errors."""
        return not self.has_errors
    
    def add_error(self, entity_type: str, entity_id: int, field: str, message: str):
        """Add an error to the validation results."""
        self.errors.append({
            'entity_type': entity_type,
            'entity_id': entity_id,
            'field': field,
            'message': message,
            'level': 'error'
        })
    
    def add_warning(self, entity_type: str, entity_id: int, field: str, message: str):
        """Add a warning to the validation results."""
        self.warnings.append({
            'entity_type': entity_type,
            'entity_id': entity_id,
            'field': field,
            'message': message,
            'level': 'warning'
        })
    
    def add_suggestion(self, entity_type: str, entity_id: int, field: str, message: str):
        """Add a suggestion to the validation results."""
        self.suggestions.append({
            'entity_type': entity_type,
            'entity_id': entity_id,
            'field': field,
            'message': message,
            'level': 'suggestion'
        })
    
    def as_dict(self) -> Dict[str, List[Dict[str, Any]]]:
        """Convert validation results to a dictionary."""
        return {
            'errors': self.errors,
            'warnings': self.warnings,
            'suggestions': self.suggestions,
            'summary': {
                'total_errors': len(self.errors),
                'total_warnings': len(self.warnings),
                'total_suggestions': len(self.suggestions),
                'is_valid': self.is_valid
            }
        }
    
    def as_dataframe(self) -> pd.DataFrame:
        """Convert validation results to a pandas DataFrame."""
        all_issues = self.errors + self.warnings + self.suggestions
        if not all_issues:
            return pd.DataFrame(columns=['entity_type', 'entity_id', 'field', 'message', 'level'])
        return pd.DataFrame(all_issues)
    
    def print_summary(self):
        """Print a summary of the validation results."""
        print(f"Validation Summary:")
        print(f"- Errors: {len(self.errors)}")
        print(f"- Warnings: {len(self.warnings)}")
        print(f"- Suggestions: {len(self.suggestions)}")
        print(f"- Valid: {self.is_valid}")
        
        if self.has_errors:
            print("\nErrors:")
            for e in self.errors:
                print(f"- [{e['entity_type']} {e['entity_id']}] {e['field']}: {e['message']}")
        
        if self.has_warnings:
            print("\nWarnings:")
            for w in self.warnings:
                print(f"- [{w['entity_type']} {w['entity_id']}] {w['field']}: {w['message']}")


def validate_sample(sample: Sample) -> ValidationResult:
    """
    Validate a single sample.
    
    Args:
        sample: The sample to validate
        
    Returns:
        ValidationResult object with validation results
    """
    result = ValidationResult()
    
    # Check required fields
    if not sample.name:
        result.add_error('Sample', sample.id, 'name', 'Sample name is required')
    
    # Check name length
    if sample.name and len(sample.name) < 3:
        result.add_warning('Sample', sample.id, 'name', 'Sample name is too short')
    
    # Validate species format if present
    if sample.species:
        # Scientific names should be in the format "Genus species"
        pattern = r'^[A-Z][a-z]+ [a-z]+$'
        if not re.match(pattern, sample.species):
            result.add_warning('Sample', sample.id, 'species', 
                              'Species name should be in the format "Genus species"')
    else:
        result.add_suggestion('Sample', sample.id, 'species', 
                             'Consider adding the species information')
    
    # Check if sample has associated compounds
    if not sample.compounds or len(sample.compounds) == 0:
        result.add_suggestion('Sample', sample.id, 'compounds', 
                             'Sample has no associated compounds')
    
    # Check if sample has been analyzed
    if not sample.analyses or len(sample.analyses) == 0:
        result.add_suggestion('Sample', sample.id, 'analyses', 
                             'Sample has not been analyzed')
    
    # Check for literature references
    if not sample.literature_references or len(sample.literature_references) == 0:
        result.add_suggestion('Sample', sample.id, 'literature', 
                             'Sample has no associated literature references')
    
    return result


def validate_compound(compound: Compound) -> ValidationResult:
    """
    Validate a single compound.
    
    Args:
        compound: The compound to validate
        
    Returns:
        ValidationResult object with validation results
    """
    result = ValidationResult()
    
    # Check required fields
    if not compound.name:
        result.add_error('Compound', compound.id, 'name', 'Compound name is required')
    
    # Validate that compound is associated with a sample
    if not compound.sample_id:
        result.add_error('Compound', compound.id, 'sample_id', 
                        'Compound must be associated with a sample')
    
    # Check if the associated sample exists
    if compound.sample_id:
        sample = Sample.query.get(compound.sample_id)
        if not sample:
            result.add_error('Compound', compound.id, 'sample_id', 
                            f'Associated sample (ID: {compound.sample_id}) does not exist')
    
    # Validate molecular structure if present
    if compound.molecular_structure:
        # Check if it's a valid SMILES or InChI string
        # This is a simple check, in a real application you'd use a chemistry library
        if not (compound.molecular_structure.startswith('C') or 
                compound.molecular_structure.startswith('O') or
                compound.molecular_structure.startswith('N') or
                compound.molecular_structure.startswith('InChI=')):
            result.add_warning('Compound', compound.id, 'molecular_structure', 
                              'Molecular structure format may not be valid')
    else:
        result.add_suggestion('Compound', compound.id, 'molecular_structure', 
                             'Consider adding molecular structure information')
    
    return result


def validate_literature_reference(reference: LiteratureReference) -> ValidationResult:
    """
    Validate a single literature reference.
    
    Args:
        reference: The literature reference to validate
        
    Returns:
        ValidationResult object with validation results
    """
    result = ValidationResult()
    
    # Check required fields
    if not reference.title:
        result.add_error('LiteratureReference', reference.id, 'title', 'Title is required')
    
    if not reference.reference_id:
        result.add_error('LiteratureReference', reference.id, 'reference_id', 'Reference ID is required')
    
    if not reference.reference_type:
        result.add_error('LiteratureReference', reference.id, 'reference_type', 'Reference type is required')
    
    # Check PubMed links
    if reference.reference_type == 'pubmed':
        # Verify the URL format
        if reference.url:
            if not reference.url.startswith('https://pubmed.ncbi.nlm.nih.gov/'):
                result.add_warning('LiteratureReference', reference.id, 'url', 
                                'PubMed URL should start with "https://pubmed.ncbi.nlm.nih.gov/"')
        
        # Ensure we have the PMID
        if not reference.reference_id.isdigit():
            result.add_warning('LiteratureReference', reference.id, 'reference_id', 
                            'PubMed ID should be a numeric value')
    
    # Check for sample association
    if not reference.sample_id:
        result.add_suggestion('LiteratureReference', reference.id, 'sample_id', 
                             'Literature reference is not associated with any sample')
    
    return result


def validate_database_integrity() -> ValidationResult:
    """
    Validate overall database integrity.
    
    Returns:
        ValidationResult object with validation results
    """
    result = ValidationResult()
    
    # Check for orphaned compounds (compounds with non-existent samples)
    orphaned_compounds = db.session.query(Compound).outerjoin(
        Sample, Compound.sample_id == Sample.id
    ).filter(Sample.id.is_(None)).all()
    
    for compound in orphaned_compounds:
        result.add_error('Compound', compound.id, 'sample_id', 
                        f'Compound refers to non-existent sample (ID: {compound.sample_id})')
    
    # Check for orphaned analyses
    orphaned_analyses = db.session.query(Analysis).outerjoin(
        Sample, Analysis.sample_id == Sample.id
    ).filter(Sample.id.is_(None)).all()
    
    for analysis in orphaned_analyses:
        result.add_error('Analysis', analysis.id, 'sample_id', 
                        f'Analysis refers to non-existent sample (ID: {analysis.sample_id})')
    
    # Check for orphaned literature references
    orphaned_refs = db.session.query(LiteratureReference).outerjoin(
        Sample, LiteratureReference.sample_id == Sample.id
    ).filter(
        LiteratureReference.sample_id.isnot(None),
        Sample.id.is_(None)
    ).all()
    
    for ref in orphaned_refs:
        result.add_error('LiteratureReference', ref.id, 'sample_id', 
                        f'Literature reference refers to non-existent sample (ID: {ref.sample_id})')
    
    # Check for duplicate compounds within the same sample
    duplicates = db.session.query(
        Compound.sample_id, Compound.name, func.count(Compound.id)
    ).group_by(Compound.sample_id, Compound.name).having(
        func.count(Compound.id) > 1
    ).all()
    
    for sample_id, name, count in duplicates:
        sample = Sample.query.get(sample_id)
        if sample:
            result.add_warning('Sample', sample_id, 'compounds', 
                              f'Sample has {count} duplicate compounds named "{name}"')
    
    return result


def validate_all_samples() -> ValidationResult:
    """
    Validate all samples in the database.
    
    Returns:
        ValidationResult object with validation results
    """
    samples = Sample.query.all()
    result = ValidationResult()
    
    for sample in samples:
        sample_result = validate_sample(sample)
        
        # Add sample validation results to the overall result
        for error in sample_result.errors:
            result.add_error(error['entity_type'], error['entity_id'], error['field'], error['message'])
        
        for warning in sample_result.warnings:
            result.add_warning(warning['entity_type'], warning['entity_id'], warning['field'], warning['message'])
        
        for suggestion in sample_result.suggestions:
            result.add_suggestion(suggestion['entity_type'], suggestion['entity_id'], suggestion['field'], suggestion['message'])
    
    return result


def validate_all_compounds() -> ValidationResult:
    """
    Validate all compounds in the database.
    
    Returns:
        ValidationResult object with validation results
    """
    compounds = Compound.query.all()
    result = ValidationResult()
    
    for compound in compounds:
        compound_result = validate_compound(compound)
        
        # Add compound validation results to the overall result
        for error in compound_result.errors:
            result.add_error(error['entity_type'], error['entity_id'], error['field'], error['message'])
        
        for warning in compound_result.warnings:
            result.add_warning(warning['entity_type'], warning['entity_id'], warning['field'], warning['message'])
        
        for suggestion in compound_result.suggestions:
            result.add_suggestion(suggestion['entity_type'], suggestion['entity_id'], suggestion['field'], suggestion['message'])
    
    return result


def validate_all_literature_references() -> ValidationResult:
    """
    Validate all literature references in the database.
    
    Returns:
        ValidationResult object with validation results
    """
    references = LiteratureReference.query.all()
    result = ValidationResult()
    
    for reference in references:
        ref_result = validate_literature_reference(reference)
        
        # Add reference validation results to the overall result
        for error in ref_result.errors:
            result.add_error(error['entity_type'], error['entity_id'], error['field'], error['message'])
        
        for warning in ref_result.warnings:
            result.add_warning(warning['entity_type'], warning['entity_id'], warning['field'], warning['message'])
        
        for suggestion in ref_result.suggestions:
            result.add_suggestion(suggestion['entity_type'], suggestion['entity_id'], suggestion['field'], suggestion['message'])
    
    return result


def validate_all() -> ValidationResult:
    """
    Validate all data in the database.
    
    Returns:
        ValidationResult object with validation results
    """
    # Start by checking database integrity
    result = validate_database_integrity()
    
    # Add sample validations
    sample_result = validate_all_samples()
    for error in sample_result.errors:
        result.add_error(error['entity_type'], error['entity_id'], error['field'], error['message'])
    for warning in sample_result.warnings:
        result.add_warning(warning['entity_type'], warning['entity_id'], warning['field'], warning['message'])
    for suggestion in sample_result.suggestions:
        result.add_suggestion(suggestion['entity_type'], suggestion['entity_id'], suggestion['field'], suggestion['message'])
    
    # Add compound validations
    compound_result = validate_all_compounds()
    for error in compound_result.errors:
        result.add_error(error['entity_type'], error['entity_id'], error['field'], error['message'])
    for warning in compound_result.warnings:
        result.add_warning(warning['entity_type'], warning['entity_id'], warning['field'], warning['message'])
    for suggestion in compound_result.suggestions:
        result.add_suggestion(suggestion['entity_type'], suggestion['entity_id'], suggestion['field'], suggestion['message'])
    
    # Add literature reference validations
    ref_result = validate_all_literature_references()
    for error in ref_result.errors:
        result.add_error(error['entity_type'], error['entity_id'], error['field'], error['message'])
    for warning in ref_result.warnings:
        result.add_warning(warning['entity_type'], warning['entity_id'], warning['field'], warning['message'])
    for suggestion in ref_result.suggestions:
        result.add_suggestion(suggestion['entity_type'], suggestion['entity_id'], suggestion['field'], suggestion['message'])
    
    return result


def export_validation_report(result: ValidationResult, format: str = 'csv', output_file: Optional[str] = None) -> str:
    """
    Export validation results to a file.
    
    Args:
        result: ValidationResult object
        format: Output format ('csv', 'json', or 'html')
        output_file: Path to save the output file (optional)
        
    Returns:
        Path to the saved file
    """
    # Generate default output filename if not provided
    if not output_file:
        timestamp = datetime.utcnow().strftime('%Y%m%d%H%M%S')
        output_file = f"validation_report_{timestamp}.{format}"
    
    # Convert results to DataFrame
    df = result.as_dataframe()
    
    # Export based on format
    if format == 'csv':
        df.to_csv(output_file, index=False)
    elif format == 'json':
        with open(output_file, 'w') as f:
            f.write(df.to_json(orient='records'))
    elif format == 'html':
        # Create a more detailed HTML report with styling
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Validation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #2c3e50; }}
                .summary {{ margin: 20px 0; padding: 15px; background-color: #f8f9fa; border-radius: 5px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:hover {{ background-color: #f5f5f5; }}
                .error {{ color: #721c24; background-color: #f8d7da; }}
                .warning {{ color: #856404; background-color: #fff3cd; }}
                .suggestion {{ color: #0c5460; background-color: #d1ecf1; }}
            </style>
        </head>
        <body>
            <h1>Validation Report</h1>
            <div class="summary">
                <h2>Summary</h2>
                <p>Total Errors: {len(result.errors)}</p>
                <p>Total Warnings: {len(result.warnings)}</p>
                <p>Total Suggestions: {len(result.suggestions)}</p>
                <p>Valid: {result.is_valid}</p>
                <p>Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
            </div>
        """
        
        if len(result.errors) > 0:
            html_content += """
            <h2>Errors</h2>
            <table>
                <tr>
                    <th>Entity Type</th>
                    <th>Entity ID</th>
                    <th>Field</th>
                    <th>Message</th>
                </tr>
            """
            for error in result.errors:
                html_content += f"""
                <tr class="error">
                    <td>{error['entity_type']}</td>
                    <td>{error['entity_id']}</td>
                    <td>{error['field']}</td>
                    <td>{error['message']}</td>
                </tr>
                """
            html_content += "</table>"
        
        if len(result.warnings) > 0:
            html_content += """
            <h2>Warnings</h2>
            <table>
                <tr>
                    <th>Entity Type</th>
                    <th>Entity ID</th>
                    <th>Field</th>
                    <th>Message</th>
                </tr>
            """
            for warning in result.warnings:
                html_content += f"""
                <tr class="warning">
                    <td>{warning['entity_type']}</td>
                    <td>{warning['entity_id']}</td>
                    <td>{warning['field']}</td>
                    <td>{warning['message']}</td>
                </tr>
                """
            html_content += "</table>"
        
        if len(result.suggestions) > 0:
            html_content += """
            <h2>Suggestions</h2>
            <table>
                <tr>
                    <th>Entity Type</th>
                    <th>Entity ID</th>
                    <th>Field</th>
                    <th>Message</th>
                </tr>
            """
            for suggestion in result.suggestions:
                html_content += f"""
                <tr class="suggestion">
                    <td>{suggestion['entity_type']}</td>
                    <td>{suggestion['entity_id']}</td>
                    <td>{suggestion['field']}</td>
                    <td>{suggestion['message']}</td>
                </tr>
                """
            html_content += "</table>"
        
        html_content += """
        </body>
        </html>
        """
        
        with open(output_file, 'w') as f:
            f.write(html_content)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    return output_file


def main():
    """Run validation on the database and print results."""
    print("Running data validation...")
    result = validate_all()
    result.print_summary()
    
    # Export report
    report_path = export_validation_report(result, format='html')
    print(f"Validation report saved to: {report_path}")


if __name__ == "__main__":
    main()