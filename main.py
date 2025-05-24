#!/usr/bin/env python3
"""
Main entry point for the Mycology Research Pipeline application.
"""

import os
import sys
import argparse
import logging
import json
from datetime import datetime

from app import create_app
from model import load_model
from batch_processor import process_batch, save_batch_results
from monitoring import start_metrics_collection_thread

# Create a Flask application instance for Gunicorn to use
app = create_app()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Mycology Research Pipeline for analyzing bioactivity patterns'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Server command
    server_parser = subparsers.add_parser('server', help='Run the web server')
    server_parser.add_argument('--port', type=int, default=5000, 
                              help='Port to run the web server on (default: 5000)')
    server_parser.add_argument('--host', default='0.0.0.0',
                              help='Host to run the web server on (default: 0.0.0.0)')
    server_parser.add_argument('--debug', action='store_true',
                              help='Run in debug mode')
    
    # API server command
    api_parser = subparsers.add_parser('api', help='Run the API server')
    api_parser.add_argument('--port', type=int, default=8000, 
                           help='Port to run the API server on (default: 8000)')
    api_parser.add_argument('--host', default='0.0.0.0',
                           help='Host to run the API server on (default: 0.0.0.0)')
    api_parser.add_argument('--debug', action='store_true',
                           help='Run in debug mode')
    
    # Batch processing command
    batch_parser = subparsers.add_parser('batch', help='Run batch processing')
    batch_parser.add_argument('input_file', help='Path to input CSV file')
    batch_parser.add_argument('--output-file', help='Path to output file')
    batch_parser.add_argument('--config', help='Path to configuration file')
    batch_parser.add_argument('--format', choices=['csv', 'excel'], default='csv',
                             help='Output file format (default: csv)')
    
    # Model command
    model_parser = subparsers.add_parser('model', help='Model operations')
    model_subparsers = model_parser.add_subparsers(dest='model_command', help='Model command')
    
    # Model train command
    train_parser = model_subparsers.add_parser('train', help='Train a new model')
    train_parser.add_argument('input_file', help='Path to training data CSV')
    train_parser.add_argument('--output-file', help='Path to save the model')
    train_parser.add_argument('--target-column', required=True, 
                             help='Name of the target column')
    train_parser.add_argument('--feature-columns', nargs='+',
                             help='List of feature columns (default: all numeric columns)')
    train_parser.add_argument('--model-type', choices=['regressor', 'classifier'], 
                             default='regressor', help='Type of model to train')
    
    # Model predict command
    predict_parser = model_subparsers.add_parser('predict', help='Make predictions with a model')
    predict_parser.add_argument('model_file', help='Path to the model file')
    predict_parser.add_argument('input_file', help='Path to input data CSV')
    predict_parser.add_argument('--output-file', help='Path to save predictions')
    predict_parser.add_argument('--feature-columns', nargs='+',
                               help='List of feature columns (default: all numeric columns)')
    
    # Version command
    version_parser = subparsers.add_parser('version', help='Show version information')
    
    return parser.parse_args()


def run_server(host, port, debug, api_only=False):
    """Run the web server."""
    app = create_app()
    
    if api_only:
        logger.info(f"Starting API server on {host}:{port}")
    else:
        logger.info(f"Starting web server on {host}:{port}")
    
    # Start metrics collection thread
    if app.config.get('ENABLE_METRICS', True):
        start_metrics_collection_thread()
    
    app.run(host=host, port=port, debug=debug)


def run_batch_processing(input_file, output_file=None, config=None, format='csv'):
    """Run batch processing on input file."""
    logger.info(f"Starting batch processing for file: {input_file}")
    
    # Load configuration if provided
    parameters = {}
    if config:
        try:
            with open(config, 'r') as f:
                parameters = json.load(f)
            logger.info(f"Loaded configuration from {config}")
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            sys.exit(1)
    
    # Process the batch
    try:
        results = process_batch(input_file, parameters=parameters)
        
        # Determine output file path if not provided
        if not output_file:
            input_name = os.path.splitext(os.path.basename(input_file))[0]
            timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
            output_file = f"{input_name}_results_{timestamp}.{format}"
        
        # Save results
        save_batch_results(results, output_file, format)
        logger.info(f"Batch processing completed. Results saved to {output_file}")
        
    except Exception as e:
        logger.error(f"Error processing batch: {str(e)}")
        sys.exit(1)


def train_model(input_file, target_column, output_file=None, feature_columns=None, model_type='regressor'):
    """Train a new model."""
    logger.info(f"Training {model_type} model using {input_file}")
    
    try:
        # Load data
        import pandas as pd
        df = pd.read_csv(input_file)
        
        # Validate target column
        if target_column not in df.columns:
            logger.error(f"Target column '{target_column}' not found in input file")
            sys.exit(1)
        
        # Determine feature columns
        if feature_columns:
            # Validate feature columns
            missing_cols = [col for col in feature_columns if col not in df.columns]
            if missing_cols:
                logger.warning(f"Missing feature columns: {missing_cols}")
                feature_columns = [col for col in feature_columns if col in df.columns]
        else:
            # Use all numeric columns except target as features
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            if target_column in numeric_cols:
                numeric_cols.remove(target_column)
            feature_columns = numeric_cols
        
        logger.info(f"Using features: {feature_columns}")
        
        # Extract features and target
        X = df[feature_columns]
        y = df[target_column]
        
        # Create and train model
        model = load_model(model_type=model_type)
        model.fit(X, y)
        
        # Determine output file path if not provided
        if not output_file:
            input_name = os.path.splitext(os.path.basename(input_file))[0]
            timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
            output_file = f"model_{model_type}_{input_name}_{timestamp}.pkl"
        
        # Save model
        model.save(output_file)
        logger.info(f"Model trained and saved to {output_file}")
        
    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        sys.exit(1)


def predict_with_model(model_file, input_file, output_file=None, feature_columns=None):
    """Make predictions with a model."""
    logger.info(f"Making predictions using model {model_file} on {input_file}")
    
    try:
        # Load data
        import pandas as pd
        df = pd.read_csv(input_file)
        
        # Load model
        from model import MycolModel
        model = MycolModel.load(model_file)
        
        # Determine feature columns
        if feature_columns:
            # Validate feature columns
            missing_cols = [col for col in feature_columns if col not in df.columns]
            if missing_cols:
                logger.warning(f"Missing feature columns: {missing_cols}")
                feature_columns = [col for col in feature_columns if col in df.columns]
        else:
            # Use all numeric columns as features
            feature_columns = df.select_dtypes(include=['number']).columns.tolist()
        
        logger.info(f"Using features: {feature_columns}")
        
        # Extract features
        X = df[feature_columns]
        
        # Make predictions
        predictions = model.predict(X)
        
        # Add predictions to dataframe
        if model.model_type == 'regressor':
            df['predicted_bioactivity'] = predictions['bioactivity_scores']
            # Add confidence intervals
            ci_low = [ci[0] for ci in predictions['confidence_intervals']]
            ci_high = [ci[1] for ci in predictions['confidence_intervals']]
            df['predicted_ci_low'] = ci_low
            df['predicted_ci_high'] = ci_high
        else:
            df['predicted_category'] = predictions['categories']
            df['predicted_probability'] = predictions['probabilities']
        
        # Determine output file path if not provided
        if not output_file:
            input_name = os.path.splitext(os.path.basename(input_file))[0]
            timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
            output_file = f"{input_name}_predictions_{timestamp}.csv"
        
        # Save predictions
        df.to_csv(output_file, index=False)
        logger.info(f"Predictions saved to {output_file}")
        
    except Exception as e:
        logger.error(f"Error making predictions: {str(e)}")
        sys.exit(1)


def show_version():
    """Show version information."""
    print("Mycology Research Pipeline v0.1.0")
    print("A tool for analyzing bioactivity patterns in medicinal compounds")
    print("\nDeveloped for mycology research and analysis")


def main():
    """Main entry point."""
    args = parse_arguments()
    
    if args.command == 'server':
        run_server(args.host, args.port, args.debug)
    elif args.command == 'api':
        run_server(args.host, args.port, args.debug, api_only=True)
    elif args.command == 'batch':
        run_batch_processing(args.input_file, args.output_file, args.config, args.format)
    elif args.command == 'model':
        if args.model_command == 'train':
            train_model(
                args.input_file, 
                args.target_column, 
                args.output_file, 
                args.feature_columns, 
                args.model_type
            )
        elif args.model_command == 'predict':
            predict_with_model(
                args.model_file,
                args.input_file,
                args.output_file,
                args.feature_columns
            )
        else:
            logger.error("Please specify a valid model command")
            sys.exit(1)
    elif args.command == 'version':
        show_version()
    else:
        # Default to running the web server if no command specified
        app = create_app()
        app.run(host='0.0.0.0', port=5000, debug=True)


if __name__ == '__main__':
    main()
