from fastapi import FastAPI
from fastapi.responses import FileResponse
from .api import endpoints
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Voice-Based Image Description API", docs_url=None, redoc_url=None)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

app.include_router(endpoints.router, prefix="/api")

@app.get("/", response_class=FileResponse)
async def read_index():
    return FileResponse('template/index.html') 