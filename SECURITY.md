# Security Policy

## Supported Versions

We release patches for security vulnerabilities. Which versions are eligible for receiving such patches depends on the CVSS v3.0 Rating:

| Version | Supported          |
| ------- | ------------------ |
| 1.x.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

We take the security of Mycology Research Pipeline seriously. If you have discovered a security vulnerability in our project, please follow these steps:

### 1. Do NOT disclose publicly

Please do **NOT** create a public GitHub issue for security vulnerabilities. This helps protect our users while we work on a fix.

### 2. Email us directly

Send details to: **security@mycologyresearch.com**

Please include:
- Type of vulnerability
- Full paths of source file(s) related to the vulnerability
- Location of the affected source code (tag/branch/commit or direct URL)
- Step-by-step instructions to reproduce the issue
- Proof-of-concept or exploit code (if possible)
- Impact of the issue, including how an attacker might exploit it

### 3. Response timeline

- **Initial response**: Within 48 hours
- **Assessment**: Within 5 business days
- **Resolution timeline**: Depends on severity
  - Critical: 7-14 days
  - High: 14-30 days
  - Medium: 30-60 days
  - Low: 60-90 days

### 4. Disclosure process

1. Security report received and acknowledged
2. We investigate and validate the report
3. We develop and test fixes
4. We prepare security advisory
5. We release the fix and publish advisory
6. Credit given to reporter (unless anonymity requested)

## Security Best Practices for Users

### Environment Variables
- Never commit `.env` files to version control
- Use strong, unique values for all secret keys
- Rotate keys regularly
- Use different keys for development and production

### Database Security
- Use strong passwords for database access
- Enable SSL/TLS for database connections in production
- Regular backups with encryption
- Limit database user permissions

### API Keys
- Keep all API keys secret
- Use environment variables for API keys
- Implement rate limiting
- Monitor API usage for anomalies

### User Data
- All passwords are hashed using bcrypt
- Enable 2FA when available
- Regular security audits
- GDPR compliance for EU users

## Security Features

### Built-in Security
- CSRF protection on all forms
- SQL injection prevention via SQLAlchemy ORM
- XSS protection through template escaping
- Secure session management
- Rate limiting on API endpoints

### Recommended Additional Security
- Use HTTPS in production
- Set up Web Application Firewall (WAF)
- Regular dependency updates
- Security headers (HSTS, CSP, etc.)
- Regular penetration testing

## Vulnerability Disclosure Hall of Fame

We thank the following security researchers for responsibly disclosing vulnerabilities:

- *Your name could be here!*

## Contact

- Security issues: security@mycologyresearch.com
- General support: support@mycologyresearch.com
- PGP key: [Download our PGP key](https://mycologyresearch.com/pgp-key.asc) 