import os
import uuid
import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from fastapi.concurrency import run_in_threadpool

# Import the refactored processing function from run_dpsk_ocr_pdf
from run_dpsk_ocr_pdf import run_ocr_on_pdf, llm, sampling_params

# Create a data directory for output files
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

app = FastAPI(
    title="DeepSeek OCR API",
    description="An API to process PDF files using DeepSeek-OCR.",
)

class PDFProcessRequest(BaseModel):
    file_path: str

class PDFProcessResponse(BaseModel):
    markdown_content: str
    image_paths: list[str]
    output_dir: str

@app.on_event("startup")
async def startup_event():
    """
    This function is called when the FastAPI application starts.
    We can use it to warm up the model if needed, but VLLM handles initialization.
    The `llm` object is already initialized when imported.
    """
    print("DeepSeek OCR API started. Model is ready.")

@app.post("/process-pdf", response_model=PDFProcessResponse)
async def process_pdf_endpoint(request: PDFProcessRequest):
    """
    Process a PDF file to extract markdown content and images.
    """
    input_pdf_path = request.file_path
    if not os.path.exists(input_pdf_path):
        raise HTTPException(status_code=404, detail="Input PDF file not found.")

    run_id = str(uuid.uuid4())
    output_dir = os.path.abspath(os.path.join(DATA_DIR, run_id))
    
    try:
        # Run the synchronous OCR function in a thread pool with a timeout
        markdown_content, image_paths = await asyncio.wait_for(
            run_in_threadpool(
                run_ocr_on_pdf,
                llm=llm,
                sampling_params=sampling_params,
                input_pdf_path=input_pdf_path,
                output_dir=output_dir
            ),
            timeout=600  # 10 minutes
        )
        
        # Make image paths absolute for the response
        absolute_image_paths = [os.path.abspath(p) for p in image_paths]

        return PDFProcessResponse(
            markdown_content=markdown_content,
            image_paths=absolute_image_paths,
            output_dir=output_dir
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=408, detail="Request timed out after 10 minutes.")
    except Exception as e:
        # Log the exception for debugging
        print(f"An error occurred during PDF processing: {e}")
        # Optionally, clean up the created directory on failure
        # import shutil
        # if os.path.exists(output_dir):
        #     shutil.rmtree(output_dir)
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
