from app import app
import web_routes
import auth_routes

# Register blueprints
app.register_blueprint(web_routes.web_bp)
app.register_blueprint(auth_routes.auth_bp, url_prefix='/auth')

# Register Replit auth blueprint
app.register_blueprint(auth_routes.make_replit_blueprint(), url_prefix='/auth')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)