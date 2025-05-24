#!/usr/bin/env python3
"""
API client for the mycology research pipeline.
"""

import requests
import json
from typing import Dict, Any, Optional, List, Union


class ApiClient:
    """Client for interacting with the pipeline API"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize the API client
        
        Args:
            base_url: Base URL of the API
        """
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
    
    def _request(self, method: str, endpoint: str, 
                 params: Optional[Dict[str, Any]] = None,
                 data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Make a request to the API
        
        Args:
            method: HTTP method (GET, POST, PUT, DELETE)
            endpoint: API endpoint
            params: Query parameters
            data: Request body data
            
        Returns:
            Response data as a dictionary
        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        response = self.session.request(
            method=method,
            url=url,
            params=params,
            json=data,
            headers={"Content-Type": "application/json"}
        )
        
        response.raise_for_status()
        
        return response.json()
    
    def get_data(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Make a GET request to the API
        
        Args:
            endpoint: API endpoint
            params: Query parameters
            
        Returns:
            Response data
        """
        return self._request("GET", endpoint, params=params)
    
    def post_data(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a POST request to the API
        
        Args:
            endpoint: API endpoint
            data: Request body data
            
        Returns:
            Response data
        """
        return self._request("POST", endpoint, data=data)
    
    def update_data(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a PUT request to the API
        
        Args:
            endpoint: API endpoint
            data: Request body data
            
        Returns:
            Response data
        """
        return self._request("PUT", endpoint, data=data)
    
    def delete_data(self, endpoint: str) -> Dict[str, Any]:
        """
        Make a DELETE request to the API
        
        Args:
            endpoint: API endpoint
            
        Returns:
            Response data
        """
        return self._request("DELETE", endpoint)
    
    # Convenience methods for common endpoints
    
    def health_check(self) -> Dict[str, Any]:
        """
        Check the health of the service
        
        Returns:
            Health check status
        """
        return self.get_data("api/health")
    
    def process_sample(self, input_data: str, parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process a sample for analysis
        
        Args:
            input_data: Input data or file reference
            parameters: Processing parameters
            
        Returns:
            Processing results
        """
        data = {
            "input_data": input_data
        }
        
        if parameters:
            data["parameters"] = parameters
        
        return self.post_data("api/process", data)
    
    def get_samples(self, limit: int = 50, offset: int = 0) -> Dict[str, Any]:
        """
        Get a list of samples
        
        Args:
            limit: Maximum number of samples to return
            offset: Offset for pagination
            
        Returns:
            List of samples
        """
        return self.get_data("api/samples", params={"limit": limit, "offset": offset})
    
    def get_sample(self, sample_id: int) -> Dict[str, Any]:
        """
        Get a specific sample by ID
        
        Args:
            sample_id: ID of the sample
            
        Returns:
            Sample details
        """
        return self.get_data(f"api/samples/{sample_id}")
    
    def create_sample(self, name: str, **kwargs) -> Dict[str, Any]:
        """
        Create a new sample
        
        Args:
            name: Name of the sample
            **kwargs: Additional sample attributes
            
        Returns:
            Created sample details
        """
        data = {"name": name, **kwargs}
        return self.post_data("api/samples", data)
    
    def get_analysis(self, analysis_id: int) -> Dict[str, Any]:
        """
        Get a specific analysis by ID
        
        Args:
            analysis_id: ID of the analysis
            
        Returns:
            Analysis details
        """
        return self.get_data(f"api/analyses/{analysis_id}")
    
    def create_batch_job(self, input_file: str, name: str = None, 
                        description: str = None, parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Create a batch processing job
        
        Args:
            input_file: Path to input file
            name: Name of the batch job
            description: Description of the batch job
            parameters: Processing parameters
            
        Returns:
            Created batch job details
        """
        data = {"input_file": input_file}
        
        if name:
            data["name"] = name
        
        if description:
            data["description"] = description
        
        if parameters:
            data["parameters"] = parameters
        
        return self.post_data("api/batch", data)
    
    def get_batch_job(self, job_id: int) -> Dict[str, Any]:
        """
        Get a specific batch job by ID
        
        Args:
            job_id: ID of the batch job
            
        Returns:
            Batch job details
        """
        return self.get_data(f"api/batch/{job_id}")


if __name__ == "__main__":
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description="Mycology Research Pipeline API Client")
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Health check command
    health_parser = subparsers.add_parser("health", help="Check service health")
    
    # Process sample command
    process_parser = subparsers.add_parser("process", help="Process a sample")
    process_parser.add_argument("input", help="Input data or file path")
    process_parser.add_argument("--parameters", type=json.loads, help="JSON parameters")
    
    # Get samples command
    samples_parser = subparsers.add_parser("samples", help="List samples")
    samples_parser.add_argument("--limit", type=int, default=50, help="Maximum samples to return")
    samples_parser.add_argument("--offset", type=int, default=0, help="Offset for pagination")
    
    # Get sample command
    sample_parser = subparsers.add_parser("sample", help="Get a specific sample")
    sample_parser.add_argument("id", type=int, help="Sample ID")
    
    # Create sample command
    create_sample_parser = subparsers.add_parser("create-sample", help="Create a new sample")
    create_sample_parser.add_argument("name", help="Sample name")
    create_sample_parser.add_argument("--description", help="Sample description")
    create_sample_parser.add_argument("--species", help="Sample species")
    create_sample_parser.add_argument("--location", help="Sample location")
    
    # Get analysis command
    analysis_parser = subparsers.add_parser("analysis", help="Get a specific analysis")
    analysis_parser.add_argument("id", type=int, help="Analysis ID")
    
    # Create batch job command
    batch_parser = subparsers.add_parser("batch", help="Create a batch job")
    batch_parser.add_argument("input_file", help="Input file path")
    batch_parser.add_argument("--name", help="Batch job name")
    batch_parser.add_argument("--description", help="Batch job description")
    batch_parser.add_argument("--parameters", type=json.loads, help="JSON parameters")
    
    # Get batch job command
    get_batch_parser = subparsers.add_parser("get-batch", help="Get a batch job")
    get_batch_parser.add_argument("id", type=int, help="Batch job ID")
    
    args = parser.parse_args()
    
    # Create API client
    client = ApiClient(args.url)
    
    try:
        # Execute command
        if args.command == "health":
            result = client.health_check()
        elif args.command == "process":
            result = client.process_sample(args.input, args.parameters)
        elif args.command == "samples":
            result = client.get_samples(args.limit, args.offset)
        elif args.command == "sample":
            result = client.get_sample(args.id)
        elif args.command == "create-sample":
            kwargs = {k: v for k, v in vars(args).items() if k not in ["command", "url", "name"] and v is not None}
            result = client.create_sample(args.name, **kwargs)
        elif args.command == "analysis":
            result = client.get_analysis(args.id)
        elif args.command == "batch":
            result = client.create_batch_job(args.input_file, args.name, args.description, args.parameters)
        elif args.command == "get-batch":
            result = client.get_batch_job(args.id)
        else:
            print("Please specify a valid command.")
            parser.print_help()
            sys.exit(1)
            
        # Print result as JSON
        print(json.dumps(result, indent=2))
        
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError:
        print("Error: Invalid JSON response", file=sys.stderr)
        sys.exit(1)
