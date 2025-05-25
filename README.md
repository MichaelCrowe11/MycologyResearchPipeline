# Mycology Research Pipeline

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/downloads/)
[![Flask](https://img.shields.io/badge/flask-3.0.0-green)](https://flask.palletsprojects.com/)

A comprehensive platform for mycological research and analysis, providing advanced tools for mushroom identification, bioactivity prediction, literature analysis, and data visualization.

## 🍄 Features

### Core Functionality
- **🔬 Advanced Image Analysis**: Computer vision-powered mushroom identification using state-of-the-art ML models
- **📊 Bioactivity Prediction**: Machine learning predictions based on 30,000+ authentic bioactivity records
- **📚 Literature Search**: Automated literature review with integration to scientific databases
- **🔄 Batch Processing**: Process multiple samples efficiently with priority queue support
- **🌐 Multi-Database Integration**: Cross-validation with iNaturalist, GBIF, and MycoBank
- **🤖 AI Research Assistant**: AI-powered research assistance for mycological queries

### Premium Features
- **💎 Professional Analysis**: Comprehensive dried specimen analysis with expert validation
- **📈 Advanced Reporting**: Detailed PDF reports with scientific citations
- **🔗 API Access**: RESTful API for programmatic access
- **👥 Multi-user Support**: Enterprise-grade user management and permissions

## 🚀 Quick Start

### Prerequisites

- Python 3.12 or higher
- pip (Python package manager)
- Git
- Redis (optional, for caching and rate limiting)
- PostgreSQL or MySQL (optional, SQLite used by default)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/MycologyResearchPipeline.git
   cd MycologyResearchPipeline
   ```

2. **Create and activate a virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

5. **Initialize the database**
   ```bash
   flask db init
   flask db migrate -m "Initial migration"
   flask db upgrade
   ```

6. **Run the application**
   ```bash
   flask run
   ```

The application will be available at `http://localhost:5000`

## 📋 Configuration

### Environment Variables

Create a `.env` file based on `.env.example`. Key configurations include:

```env
# Essential Configuration
SECRET_KEY=your-secret-key-here
DATABASE_URL=sqlite:///mycology.db

# Stripe Payment (for premium features)
STRIPE_PUBLIC_KEY=pk_test_your_key
STRIPE_SECRET_KEY=sk_test_your_key

# External APIs
OPENAI_API_KEY=your-openai-key
INATURALIST_API_KEY=your-inaturalist-key
GBIF_API_KEY=your-gbif-key
```

See `.env.example` for complete configuration options.

## 🏗️ Project Structure

```
MycologyResearchPipeline/
├── app.py                    # Application entry point
├── config.py                 # Configuration settings
├── models.py                 # Database models
├── requirements.txt          # Python dependencies
├── .env.example             # Environment variables template
│
├── routes/                   # Route handlers
│   ├── auth_routes.py       # Authentication endpoints
│   ├── payment_routes.py    # Payment processing
│   ├── api_routes.py        # REST API endpoints
│   └── web_routes.py        # Web interface routes
│
├── services/                 # Business logic
│   ├── computer_vision.py   # Image analysis
│   ├── ml_bioactivity.py    # Bioactivity predictions
│   ├── literature.py        # Literature search
│   └── ai_assistant.py      # AI integration
│
├── templates/               # HTML templates
│   ├── base.html           # Base template
│   ├── index.html          # Landing page
│   └── payment/            # Payment pages
│
├── static/                  # Static assets
│   ├── css/                # Stylesheets
│   ├── js/                 # JavaScript
│   └── images/             # Images
│
└── tests/                   # Test suite
    ├── test_api.py         # API tests
    ├── test_models.py      # Model tests
    └── test_services.py    # Service tests
```

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=.

# Run specific test file
pytest tests/test_api.py

# Run with verbose output
pytest -v
```

## 📚 API Documentation

### Authentication

All API endpoints require authentication via JWT tokens:

```bash
# Get access token
curl -X POST http://localhost:5000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email": "user@example.com", "password": "password"}'

# Use token in requests
curl -X GET http://localhost:5000/api/samples \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN"
```

### Key Endpoints

- `POST /api/analyze` - Analyze mushroom image
- `GET /api/samples` - List user samples
- `POST /api/batch` - Submit batch processing job
- `GET /api/literature/search` - Search scientific literature

See full API documentation at `/api/docs` when running the application.

## 🚢 Deployment

### Docker Deployment

```bash
# Build Docker image
docker build -t mycology-pipeline .

# Run container
docker run -p 5000:5000 --env-file .env mycology-pipeline
```

### Production Deployment

1. **Set environment to production**
   ```bash
   export FLASK_ENV=production
   export DEBUG=False
   ```

2. **Use a production WSGI server**
   ```bash
   gunicorn -w 4 -b 0.0.0.0:5000 app:app
   ```

3. **Set up reverse proxy** (Nginx example)
   ```nginx
   server {
       listen 80;
       server_name yourdomain.com;
       
       location / {
           proxy_pass http://localhost:5000;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
       }
   }
   ```

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guide
- Add tests for new features
- Update documentation as needed
- Use meaningful commit messages

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Scientific data provided by iNaturalist, GBIF, and MycoBank
- ML models trained on authentic mycological datasets
- Community contributors and beta testers

## 📞 Support

- **Documentation**: [docs.mycologyresearch.com](https://docs.mycologyresearch.com)
- **Issues**: [GitHub Issues](https://github.com/yourusername/MycologyResearchPipeline/issues)
- **Email**: support@mycologyresearch.com
- **Discord**: [Join our community](https://discord.gg/mycology)

## 🔒 Security

For security concerns, please email security@mycologyresearch.com instead of using public issue trackers.

---

Made with 🍄 by the Mycology Research Team 