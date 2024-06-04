from flask import Flask

app = Flask(__name__)

@app.route("/events")
def fetch_events():
    return "<p>Hello, World!</p>"
