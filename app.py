from flask import Flask

app = Flask(__name__)

@app.route("/")
def home():
    return "Hola mundo, mi primera app web!"

if __name__ == "__main__":
    app.run()
