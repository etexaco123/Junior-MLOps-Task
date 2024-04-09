from fastapi import FastAPI
from pydantic import BaseModel
from model import Handler
from typing import List

# See https://fastapi.tiangolo.com/tutorial/first-steps/ for more general information on FastAPI
app: FastAPI = FastAPI()

# Create schemas for the input and output for the POST requests
# See: https://fastapi.tiangolo.com/tutorial/body/
# Create instance of Handler
handler = Handler()

# Define request and response models
class TextRequest(BaseModel):
    text: str


class SimilarityRequest(BaseModel):
    text1: str
    text2: str


class EmbeddingResponse(BaseModel):
    embedding: list[float]


class SimilarityResponse(BaseModel):
    similarity_score: float



# Create FastAPI endpoints here
# See an example of a POST request here: https://fastapi.tiangolo.com/tutorial/body/#declare-it-as-a-parameter

# Define endpoints
@app.post("/embed", response_model=EmbeddingResponse)
async def get_embedding(text_request: TextRequest):
    text = text_request.text
    embedding = handler.embed(text)
    return {"embedding": embedding}


@app.post("/similarity", response_model=SimilarityResponse)
async def get_similarity(similarity_request: SimilarityRequest):
    text1 = similarity_request.text1
    text2 = similarity_request.text2
    similarity_score = handler.similarity(text1, text2)
    return {"similarity_score": similarity_score}

# This part runs the FastAPI app if you execute this Python file using `python app.py`.
# You do not have to change this part.
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8080, reload=True)
