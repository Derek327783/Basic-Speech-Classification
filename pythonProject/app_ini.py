from waitress import serve
from server import app  # Replace with your actual Flask app import

serve(app, host='127.0.0.1', port=5050)