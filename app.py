from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from IPadvisor import IPAdvisor
import json

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize IPAdvisor
ip_advisor = IPAdvisor()

@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/search_prior_art")
async def search_prior_art(request: Request):
    data = await request.json()
    query = data.get("query", "")
    results = ip_advisor.search_prior_art(query)
    return results

@app.post("/analyze_patent")
async def analyze_patent(request: Request):
    data = await request.json()
    patent_text = data.get("patent_text", "")
    analysis = ip_advisor.extract_patent_sections(patent_text)
    return {"analysis": analysis}

@app.post("/check_similarity")
async def check_similarity(request: Request):
    data = await request.json()
    patent_text = data.get("patent_text", "")
    threshold = data.get("threshold", 0.7)
    similar_patents = ip_advisor.analyze_patent_similarity(patent_text, threshold)
    return similar_patents

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8006) 