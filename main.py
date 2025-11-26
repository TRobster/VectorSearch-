from api import app

@app.get("/")
def root():
    return {"message": "Local Vector Search Engine Running"}