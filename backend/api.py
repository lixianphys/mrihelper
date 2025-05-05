from fastapi import FastAPI, UploadFile, HTTPException, Request, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import StreamingResponse
import aiofiles
import uuid
import os

import json
import logging
import sys
import numpy as np
from vtkmodules.util.numpy_support import vtk_to_numpy
import vtk
import nibabel as nib
import io
import gzip
import asyncio # Added for SSE
from sse_starlette.sse import EventSourceResponse # Added for SSE
import redis.asyncio as aioredis # Use asyncio version for FastAPI
from enum import Enum
from celery import chain # Import chain for task linking
import uvicorn
from s3storage import s3_storage, USE_S3_STORAGE, S3_OUTPUT_PREFIX, S3_UPLOAD_PREFIX
from tasks import process_volume, segment_volume, prepare_3d_data 
from task_store import TaskHandler
from nibabel.fileholders import FileHolder # Make sure FileHolder is imported
## Comment this line to avoid loading .env file in production environment like container/pod, which should not contain any secrets.
# from dotenv import load_dotenv
# load_dotenv(dotenv_path=os.getenv("ENV_FILE"))

# --- Environment Variables ---
REDIS_HOST = os.getenv('REDIS_HOST')
REDIS_PORT = os.getenv('REDIS_PORT')
REDIS_DB = os.getenv('REDIS_DB')
REDIS_DB_PUBSUB = os.getenv('REDIS_DB_PUBSUB')
LARGE_FILE_SIZE = os.getenv('LARGE_FILE_SIZE', 20000000) # 20MB
# --- Configure logging to output to stdout ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("mrimaster-backend")


# --- Directories ---
OUTPUT_DIR = 'output/'
TEMP_DATA_DIR = "temp_data/"
UPLOAD_DIR = 'uploads/'

logger.info(f"REDIS_HOST: {REDIS_HOST}, REDIS_PORT: {REDIS_PORT}, REDIS_DB_PUBSUB: {REDIS_DB_PUBSUB}")
logger.debug(f"USE_S3_STORAGE: {USE_S3_STORAGE}")

# --- FastAPI App ---
app = FastAPI(
    title="MRI Master API",
    description="API for processing and visualizing MRI data."
)

origins = ["http://frontend:8000", "http://localhost:8000"]
if os.getenv("ALLOWED_ORIGINS"):
    origins.extend(os.getenv("ALLOWED_ORIGINS").split(","))

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins = origins,
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Add GZipMiddleware to compress responses
app.add_middleware(GZipMiddleware, minimum_size=1000)


# Ensure directories exist
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(TEMP_DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

task_handler = TaskHandler(redis_host=REDIS_HOST, redis_port=REDIS_PORT, redis_db=REDIS_DB)

# --- Redis Connection for SSE ---
# Use connection details consistent with tasks.py publisher
redis_listen_client = None

@app.on_event("startup")
async def startup_event():
    global redis_listen_client
    try:
        redis_listen_client = await aioredis.from_url(
            f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB_PUBSUB}",
            encoding="utf-8",
            decode_responses=True
        )
        await redis_listen_client.ping() # Verify connection
        logger.info(f"Connected to Redis for SSE listening on {REDIS_HOST}:{REDIS_PORT} DB {REDIS_DB_PUBSUB}")
    except Exception as e:
        logger.error(f"Failed to connect to Redis for SSE on startup: {e}", exc_info=True)
        redis_listen_client = None # Ensure it's None if connection failed

@app.on_event("shutdown")
async def shutdown_event():
    if redis_listen_client:
        await redis_listen_client.close()
        logger.info("Closed Redis SSE connection.")

# --- Workflow Definition ---
class WorkflowType(str, Enum):
    VISUALIZE_ONLY = "visualize_only"
    SEGMENT_AND_VISUALIZE = "segment_and_visualize"

# --- API Endpoints ---

@app.post("/workflow/start", status_code=202) # 202 Accepted indicates processing started
async def start_workflow(
    file: UploadFile,
    workflow_type: WorkflowType = Form(...)
) -> dict:
    """
    Starts a processing workflow for an uploaded NIfTI file.
    Accepts the file and workflow type, saves the file, triggers the
    appropriate Celery task chain, and returns a workflow ID.
    """
    # --- Check if file is a NIFTI file ---
    if not file.filename.endswith('.nii.gz'):
        logger.warning(f"Upload rejected: Invalid file type {file.filename}")
        raise HTTPException(status_code=415, detail="Only .nii.gz files are supported")

    # --- File Saving ---
    volume_id = str(uuid.uuid4())
    file_extension = "nii.gz"
    
    # --- Check if file size is too large ---
    if file.size > LARGE_FILE_SIZE: # 20MB
        logger.warning(f"Upload rejected: File size too large {file.size} bytes")
        raise HTTPException(status_code=413, detail=f"File size exceeds {LARGE_FILE_SIZE/1024/1024} MB")

    file_path = ""
    
    # -- Save nifti file to S3 or local --
    try:
        if USE_S3_STORAGE:
            # For S3 storage
            content = await file.read()
            s3_key = f"{S3_UPLOAD_PREFIX}{volume_id}.{file_extension}"
            s3_storage.save_bytes(content, s3_key)
            file_path = s3_key
            logger.info(f"Successfully saved {file.filename} to S3: {file_path}")
        else:
            # For local storage
            file_path = os.path.join(UPLOAD_DIR, f"{volume_id}.{file_extension}")
            async with aiofiles.open(file_path, "wb") as f:
                content = await file.read()
                await f.write(content)
            logger.info(f"Successfully saved {file.filename} to local path: {file_path}")
    except Exception as e:
        logger.error(f"Failed to save uploaded file {file.filename}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {str(e)}")

    # --- Workflow Triggering ---
    workflow_id = str(uuid.uuid4())
    logger.info(f"Starting workflow: {workflow_type.value} with ID: {workflow_id} for file: {file_path}")

    # Define the final preparation task signature (with correct path for S3 or local)
    output_path = S3_OUTPUT_PREFIX if USE_S3_STORAGE else OUTPUT_DIR
    prepare_sig = prepare_3d_data.s(workflow_id=workflow_id, output_path=output_path)

    try:
        if workflow_type == WorkflowType.VISUALIZE_ONLY:
            # Chain: process_volume -> prepare_3d_data
            task_chain = chain(process_volume.s(nii_path=file_path, workflow_id=workflow_id), prepare_sig)
            task_chain.apply_async()
        elif workflow_type == WorkflowType.SEGMENT_AND_VISUALIZE:
            # Chain: segment_volume -> prepare_3d_data
            task_chain = chain(segment_volume.s(image_path=file_path, workflow_id=workflow_id), prepare_sig)
            task_chain.apply_async()
        else:
            logger.error(f"Invalid workflow type received: {workflow_type}")
            raise HTTPException(status_code=400, detail="Invalid workflow type specified")

        logger.info(f"Workflow {workflow_id} ({workflow_type.value}) task chain initiated.")
        return {"workflow_id": workflow_id}

    except Exception as e:
        logger.error(f"Failed to trigger Celery task chain for workflow {workflow_id}: {e}", exc_info=True)
        # Attempt cleanup of saved file if task trigger fails
        if USE_S3_STORAGE:
            if file_path:
                try:
                    s3_storage.delete_file(file_path)
                    logger.info(f"Cleaned up S3 file {file_path} after task trigger failure.")
                except Exception as remove_err:
                    logger.error(f"Failed to cleanup S3 file {file_path}: {remove_err}")
        else:
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    logger.info(f"Cleaned up local file {file_path} after task trigger failure.")
                except OSError as remove_err:
                    logger.error(f"Failed to cleanup local file {file_path}: {remove_err}")
        raise HTTPException(status_code=500, detail=f"Failed to start workflow processing: {str(e)}")


@app.get("/workflow/events/{workflow_id}")
async def workflow_events(request: Request, workflow_id: str):
    """
    Server-Sent Events endpoint to stream status updates for a given workflow_id.
    """
    if not redis_listen_client:
        logger.error("SSE endpoint called but Redis listener is not available.")
        raise HTTPException(status_code=503, detail="Server cannot process events at this time.")

    logger.info(f"SSE connection requested for workflow: {workflow_id}")

    async def event_generator():
        pubsub = None
        try:
            pubsub = redis_listen_client.pubsub()
            channel = f"workflow:{workflow_id}:status"
            await pubsub.subscribe(channel)
            logger.info(f"SSE subscribed to Redis channel: {channel}")

            # Send an initial connected message
            yield json.dumps({"status": "connected", "workflow_id": workflow_id})

            while True:
                # Check if client disconnected
                if await request.is_disconnected():
                    logger.info(f"SSE client disconnected for workflow: {workflow_id}")
                    break

                # Wait for message with timeout
                try:
                    message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=10.0) # Timeout allows checking disconnect
                    if message is not None and message.get("type") == "message":
                        data = message["data"] # Data is already decoded by aioredis client
                        logger.debug(f"SSE received from Redis ({channel}): {data}")
                        yield data # Send raw JSON string received from Redis
                    else:
                        # Send a keep-alive message (or just continue loop after timeout)
                        # yield json.dumps({"status": "ping"})
                        pass
                except asyncio.TimeoutError:
                     # No message received, loop again to check disconnect and wait for next message
                     pass
                except Exception as e:
                    logger.error(f"Error reading from Redis pubsub channel {channel}: {e}", exc_info=True)
                    # Send an error message to the client before breaking
                    try:
                        yield json.dumps({"status": "error", "message": "Error reading backend events"})
                    except: pass # Ignore if yield fails because client disconnected
                    break

        except Exception as e:
            logger.error(f"Error in SSE event generator for workflow {workflow_id}: {e}", exc_info=True)
            # Attempt to send a final error message
            try:
                yield json.dumps({"status": "error", "message": "Failed to subscribe to backend events"})
            except: pass

        finally:
            if pubsub:
                try:
                    await pubsub.unsubscribe(channel)
                    await pubsub.close()
                    logger.info(f"SSE unsubscribed and closed pubsub for channel: {channel}")
                except Exception as e:
                    logger.error(f"Error closing pubsub for {channel}: {e}")
            logger.info(f"SSE event stream closed for workflow: {workflow_id}")

    return EventSourceResponse(event_generator(), media_type="text/event-stream")


# Endpoint to serve generated files (VTI, NPY, etc.)
@app.get("/datafile")
async def serve_datafile(path: str):
    if USE_S3_STORAGE:
        try:
            # Instead of redirecting, proxy the file through our API
            file_data = s3_storage.load_file(path)
            from fastapi.responses import Response
            filename = path.split("/")[-1]
            return Response(
                content=file_data,
                media_type="application/octet-stream",
                headers={"Content-Disposition": f"attachment; filename={filename}"}
            )
        except Exception as e:
            logger.error(f"Failed to retrieve S3 file {path}: {e}", exc_info=True)
            raise HTTPException(status_code=404, detail=f"File not found or access denied: {path}")
    else:
        # For local filesystem: Basic security check to prevent accessing files outside allowed directories
        normalized_path = os.path.normpath(path)
        # Ensure base_dirs are absolute paths for reliable comparison
        base_dirs = [os.path.abspath(d) for d in [UPLOAD_DIR, TEMP_DATA_DIR, OUTPUT_DIR]]

        is_safe = False
        try:
            abs_req_path = os.path.abspath(normalized_path)
            is_safe = any(abs_req_path.startswith(base_dir) for base_dir in base_dirs)
        except Exception as e:
            logger.error(f"Security check failed for path: {path}, error: {e}")
            raise HTTPException(status_code=403, detail="Invalid path specified")

        if not is_safe:
            logger.warning(f"Attempted access to disallowed path: {path}")
            raise HTTPException(status_code=403, detail="Access denied to requested path")

        if not os.path.exists(normalized_path):
            logger.warning(f"Requested file not found: {normalized_path}")
            raise HTTPException(status_code=404, detail=f"File not found: {path}")

        # Serve the file
        from fastapi.responses import FileResponse
        return FileResponse(
            path=normalized_path,
            filename=os.path.basename(normalized_path),
            media_type="application/octet-stream"
        )



@app.get("/download/segmentation/nifti")
async def download_segmentation_nifti(prediction_vti_path: str):
    logger.info(f"Request to download segmentation as NIfTI. VTI: {prediction_vti_path}")
    def load_vti_data(path:str)->np.ndarray | None:
        try:
            reader = vtk.vtkXMLImageDataReader()
            if USE_S3_STORAGE:
                vti_bytes = s3_storage.load_file(path)
                if not vti_bytes: return None
                reader.SetReadFromInputString(True)
                reader.SetInputString(vti_bytes)
            else:
                if not os.path.exists(path): 
                    logger.error(f"Requested file not found: {path}")
                    return None
                reader.SetFileName(path)
            reader.Update()
            image_data = reader.GetOutput()

            dims = image_data.GetDimensions()

            if not image_data: 
                logger.error("Failed to read get image data from VTI reader for path: {path}")
                return None
            
            vtk_array = image_data.GetPointData().GetScalars()
            numpy_array = vtk_to_numpy(vtk_array)
            try:
                numpy_array_3d = numpy_array.reshape(dims[0], dims[1], dims[2])
            except Exception as e:
                logger.error(f"Error reshaping numpy array: {e}")
                return None

            logger.info(f"Numpy array shape: {numpy_array_3d.shape}")
            return numpy_array_3d
        except Exception as e:
            logger.error(f"Error loading VTI data from {path}: {e}")
            raise HTTPException(status_code=500, detail=f"Error loading VTI data: {str(e)}")        
        
    # Load the segmentation data from the VTI file
    segmentation_data = load_vti_data(prediction_vti_path)
    if segmentation_data is None:
        raise HTTPException(status_code=404, detail="Segmentation data not found or failed to load.")
  
    try:
        # Ensure data type is appropriate for segmentation (e.g., integers)
        # Adjust dtype based on your label range (0, 1, 2, 3?)
        segmentation_data_int = segmentation_data.astype(np.uint8)
        logger.info(f"Loaded segmentation data shape for NIfTI conversion: {segmentation_data_int.shape}")

        # Create a new NIfTI image using the segmentation data array
        # Use default identity affine matrix and let nibabel create a basic header.
        affine = np.eye(4)
        new_nii = nib.Nifti1Image(segmentation_data_int, affine=affine, header=None)
        # Set the data type in the header explicitly
        new_nii.header.set_data_dtype(np.uint8) # Match the array dtype

        # --- Save to an in-memory gzipped buffer using to_file_map and FileHolder ---
        buffer = io.BytesIO()
        # Use a with statement to ensure the GzipFile is closed/flushed correctly
        with gzip.GzipFile(fileobj=buffer, mode='wb') as gz_file:
            # Wrap the GzipFile with a FileHolder
            fileholder = FileHolder(filename='', fileobj=gz_file)
            # Call to_file_map, passing the FileHolder for both header and image
            new_nii.to_file_map({'header': fileholder, 'image': fileholder})
        # The 'with' block ensures gz_file is closed here.

        buffer.seek(0) # Rewind buffer

        # Determine download filename
        download_filename = "segmentation.nii.gz"
        try:
            base_filename = prediction_vti_path.split('/')[-1]
            base_filename = base_filename.replace(".vti", "")
            parts = base_filename.split('_')
            if len(parts) > 1 and len(parts[-1]) == 8: # Simple check for _uuidhex
                 base_filename = '_'.join(parts[:-1])
            if base_filename:
                 download_filename = f"{base_filename}.nii.gz"
        except Exception as fname_e:
            logger.warning(f"Could not parse base filename from {prediction_vti_path}: {fname_e}")
            download_filename = "segmentation.nii.gz"

        logger.info(f"Sending segmentation as {download_filename}")
        return StreamingResponse(
            buffer,
            media_type="application/gzip",
            headers={"Content-Disposition": f"attachment; filename={download_filename}"}
        )
    except Exception as e:
        logger.error(f"Error creating or streaming NIfTI file from VTI {prediction_vti_path}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to create NIfTI file for download.")
    
@app.get("/health", status_code=200)
async def health_check():
    return {"status": "ok"}
        

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)