import json
from fastapi import FastAPI, UploadFile, File, HTTPException, status, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi import FastAPI
from contextlib import asynccontextmanager

from starlette.staticfiles import StaticFiles

from detection_post import router, img_object_detection_to_img

app = FastAPI(
    title="Object Detection using YOLOv8 and FastAPI Template",
    description="""Obtain object value out of image and return image and json result""",
)

# Setup templates and static files
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.include_router(router)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

async def save_openapi_json(app):
    openapi_data = app.openapi()
    with open("openapi.json", "w") as file:
        json.dump(openapi_data, file)

@asynccontextmanager
async def lifespan(app: FastAPI):
    await save_openapi_json(app)
    yield
    # Tambahkan kode yang perlu dijalankan saat aplikasi berhenti
    pass

app = FastAPI(lifespan=lifespan)

@app.get("/", response_class=HTMLResponse)
def read_item(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/description", response_class=HTMLResponse)
def read_description(request: Request):
    return templates.TemplateResponse("description.html", {"request": request})

# Create a new route for processing the image and rendering the result page
@app.post("/detection/img_object_detection_to_img", response_class=HTMLResponse)
async def process_image_and_display(request: Request, file: UploadFile = File(...)):
    # Process the image and get the result
    result_image_bytes = img_object_detection_to_img(file.file.read())

    # Save the result image to a file (you may want to save it temporarily or handle it differently)
    result_image_path = "static/result_image.jpg"
    with open(result_image_path, "wb") as result_file:
        result_file.write(result_image_bytes)

    # Render the result.html page with the image path
    return templates.TemplateResponse("result.html", {"request": request, "image_path": result_image_path})