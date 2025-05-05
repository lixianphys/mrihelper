from onnxruntime import InferenceSession
import numpy as np
import os
from typing import Optional
from celery import Celery, states
from celery.exceptions import Ignore
import vtk
from vtkmodules.util.numpy_support import numpy_to_vtk
import nibabel as nib
import uuid
import json
import redis
import logging
import tempfile
from s3storage import s3_storage, USE_S3_STORAGE

## Comment this line to avoid loading .env file in production environment, which should not contain any secrets.
# from dotenv import load_dotenv
# load_dotenv(dotenv_path=os.getenv("ENV_FILE"))

TEMP_DATA_DIR = "temp_data/"
OUTPUT_DIR = "output/"

# --- Environment Variables ---

REDIS_HOST = os.getenv("REDIS_HOST")
REDIS_PORT = os.getenv("REDIS_PORT")
REDIS_DB = os.getenv("REDIS_DB")
REDIS_DB_PUBSUB = os.getenv("REDIS_DB_PUBSUB")
BROKER_URL = os.getenv("BROKER_URL")
BACKEND_URL = os.getenv("BACKEND_URL")


# Define a directory for temporary data files

os.makedirs(TEMP_DATA_DIR, exist_ok=True)
logger = logging.getLogger("mrimaster-celery")

# --- Redis Client for Pub/Sub ---
redis_publisher = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB_PUBSUB, decode_responses=True)

def publish_status(workflow_id: str, status: str, data: Optional[dict] = None):
    """Publishes a status update to the Redis channel for a specific workflow."""
    if not workflow_id:
        logger.warning("Warning: workflow_id is missing, cannot publish status.")
        return
    channel = f"workflow:{workflow_id}:status"
    message = json.dumps({"status": status, "data": data or {}})
    try:
        redis_publisher.publish(channel, message)
        logger.info(f"Published to {channel}: {message}")
    except redis.exceptions.ConnectionError as e:
        logger.error(f"Error publishing to Redis channel {channel}: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during Redis publish: {e}")


# Custom Celery Task class to automatically publish status updates
class WorkflowTask(Celery.Task):
    def on_failure(self, exc, task_id, args, kwargs, einfo):
        workflow_id = kwargs.get('workflow_id')
        task_name = self.name.split('.')[-1] # Get simple task name
        publish_status(workflow_id, "error", {"task": task_name, "message": str(exc), "details": str(einfo)})
        logger.error(f"Task {task_id} ({task_name}) failed: {exc}")

    def on_success(self, retval, task_id, args, kwargs):
        # Success status is typically published within the task logic
        # Or at the end of a chain
        workflow_id = kwargs.get('workflow_id')
        task_name = self.name.split('.')[-1]
        logger.info(f"Task {task_id} ({task_name}) succeeded.")
        # If this task is the *end* of a chain, we might publish final success here.
        # For intermediate tasks, success is implied by the next task starting.

    def __call__(self, *args, **kwargs):
        workflow_id = kwargs.get('workflow_id')
        task_name = self.name.split('.')[-1]
        publish_status(workflow_id, "task_started", {"task": task_name})
        logger.info(f"Starting task {self.request.id} ({task_name}) for workflow {workflow_id}")
        return super().__call__(*args, **kwargs)


def preprocess_volume(workflow_id, volume_path, use_cropping=False, cropping_shape=None):
    """
    Preprocesses a volume by cropping to desired shape for the neural network model input.
    """    # --- Helper Functions ---
    def center_crop(img, target_shape=(1, 4, 208, 208, 144)):
        """Crop the center of the image to the target shape."""
        crop_slices = tuple(
            slice((dim - target) // 2, (dim - target) // 2 + target)
            for dim, target in zip(img.shape, target_shape)
        )
        return img[crop_slices]

    # Load volume data from S3 or local file
    if USE_S3_STORAGE:
        # Load from S3
        byte_data = s3_storage.load_file(volume_path)
        
        # Save to a temporary file first
        with tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False) as temp_file:
            temp_path = temp_file.name
            temp_file.write(byte_data)
        
        try:
            # Load it the same way as a local file
            volume = nib.load(temp_path).get_fdata()
        except Exception as e:
            logger.error(f"Error loading volume from temp file: {e}")
            raise e
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)
    else:
        # Load from local file
        volume = nib.load(volume_path).get_fdata()

    # --- Check if the volume has the correct dimension ---
    logger.info(f"Volume dimension: {volume.ndim}, shape: {volume.shape}")

    if volume.ndim == 4:
        # add batch dimension [X,Y,Z,C] -> [B,X,Y,Z,C]
        volume = np.expand_dims(volume, axis=0)
        # [B, X, Y, Z, C] -> [B, C, X, Y, Z]
        volume = np.transpose(volume, (0, 4, 1, 2, 3))
        dim_B, dim_C = volume.shape[0], volume.shape[1]
        # crop to the largest possible cube
        if use_cropping and cropping_shape:
            volume = center_crop(volume, cropping_shape)
        volume = volume.astype(np.float32)
        # turn [FLAIR, T1, T1c, T2] -> [T1c, T1, T2, FLAIR]
        channel_order = [2, 1, 3, 0]
        volume[:] = volume[:, channel_order, ...]

        return volume
    else:
        try: 
            publish_status(workflow_id, "error", {"message": f"Your nii.gz file has the wrong dimension. It should be [x, y, z, channel] but it is {volume.shape}"})
            logger.info(f"Published status: Error: Volume with shape {volume.shape} is not [x, y, z, channel]")
        except Exception as e:
            logger.error(f"Error publishing status: {e}")
        raise ValueError(f"Volume with shape {volume.shape} is not [x, y, z, channel]")

    

# Create Celery app
celery_app = Celery(
    'inference',
    broker= BROKER_URL,
    backend= BACKEND_URL
)

# Configure Celery
celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='Europe/Berlin',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=1800,  # 30 minutes
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=50,
    task_base_class=WorkflowTask # Use our custom task class
)


@celery_app.task(bind=True, name='process_volume')
def process_volume(self: WorkflowTask, nii_path: str, workflow_id: str):
    """
    Celery task to convert a NIfTI file to a numpy array and save it as a .npy file.
    Returns the path to the .npy file. Publishes status updates.
    Links to prepare_3d_data on success.
    """
    publish_status(workflow_id, "processing_volume_started")
    original_volume_path = None # Initialize to None
    try:
        input_tensor = preprocess_volume(workflow_id=workflow_id, volume_path=nii_path)
        # --- Save to S3 or local ---
        unique_id = str(uuid.uuid4())
        
        # Save the processed volume to S3 or local file
  
        # Save to local file
        original_volume_path = os.path.join(TEMP_DATA_DIR, f"original_{unique_id}.npy")
        np.save(original_volume_path, input_tensor)
        
        # Publish status, include the path for potential debugging/logging
        publish_status(workflow_id, "processing_volume_completed", {"original_volume_path": original_volume_path})

        # Prepare result for the next task in the chain (contains file path)
        result_for_chain = {
            "original_volume_path": original_volume_path,
            "prediction_path": None # No prediction in this workflow branch
        }
        return result_for_chain # Return path for chaining via result backend

    except Exception as e:
        logger.error(f"Error in process_volume: {e}")
        # Clean up created file on error
        if original_volume_path:
            if USE_S3_STORAGE:
                if s3_storage.delete_file(original_volume_path):
                    logger.info(f"Cleaned up S3 file: {original_volume_path}")
            else:
                if os.path.exists(original_volume_path):
                    try: os.remove(original_volume_path) 
                    except OSError: pass
        self.update_state(state=states.FAILURE, meta={'exc': str(e)})
        try:
            publish_status(workflow_id, "error", {"message": str(e)})
        except Exception as e:
            logger.error(f"Error publishing status: {e}")
        raise Ignore() # Prevent Celery from retrying or marking as success


@celery_app.task(bind=True, name='segment_volume')
def segment_volume(self: WorkflowTask, image_path: str, workflow_id: str):
    """
    Celery task to segment a volume using the SegResNet model.
    Saves the original preprocessed volume and the prediction tensor to .npy files.
    Returns paths to the saved files. Publishes status updates.
    Links to prepare_3d_data on success.
    """
    publish_status(workflow_id, "segmentation_started")
    original_volume_path = None
    prediction_path = None
    try:
        publish_status(workflow_id, "segmentation_preprocessing")
        input_temp_tensor = preprocess_volume(workflow_id=workflow_id, volume_path=image_path)
        # --- Select the correct model based on the input shape ---
        if input_temp_tensor.shape[2] < 208 or input_temp_tensor.shape[3] < 208 or input_temp_tensor.shape[4] < 144:
            model_path = os.getenv("MODEL_PATH", "srn128_128_128.onnx")
            model_label = "SegResNetModel (128x128x128)"
            cropping_shape = (1, 4, 128, 128, 128)
        else:
            model_path = os.getenv("MODEL_PATH_LARGE", "srn208_208_144.onnx")
            model_label = "SegResNetModel (208x208x144)"
            cropping_shape = (1, 4, 128, 128, 128)
        # --- Crop the volume to the correct shape ---
        input_tensor = preprocess_volume(workflow_id=workflow_id, volume_path=image_path, use_cropping=True, cropping_shape=cropping_shape)
        logger.info(f"Using model: {model_path}")
        session = InferenceSession(model_path)
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name

        publish_status(workflow_id, f"segmentation_running_model_{model_label}")
        # NumPy array is already ready for ONNX
        output = session.run([output_name], {input_name: input_tensor})[0]
        logger.info(f"directly after model, output shape: {output.shape}")
        output = output[0] # Remove the batch dimension
        # Apply multi-class mapping based on threshold
        output = np.where(
            output[[2]] > 0, 3,  # ET (Enhancing Tumor)
            np.where(
                output[[0]] > 0, 1,  # TC (Tumor Core)
                np.where(
                    output[[1]] > 0, 2, 0  # WT (Whole Tumor)
                )
            )
        )

        publish_status(workflow_id, "segmentation_saving_results")
        unique_id = str(uuid.uuid4())
        

        # Save to local filesystem
        original_volume_path = os.path.join(TEMP_DATA_DIR, f"original_{unique_id}.npy")
        prediction_path = os.path.join(TEMP_DATA_DIR, f"prediction_{unique_id}.npy")
        
        np.save(original_volume_path, input_tensor)
        np.save(prediction_path, output)

        # Prepare result for the next task (contains file paths)
        result_for_chain = {
            'original_volume_path': original_volume_path,
            'prediction_path': prediction_path,
        }
        # Publish status completion message with paths
        publish_status(workflow_id, "segmentation_completed", result_for_chain)
        return result_for_chain # Return paths for chaining via result backend

    except Exception as e:
        logger.error(f"Error in segment_volume: {e}")
        # Clean up created files on error
        if USE_S3_STORAGE:
            if original_volume_path:
                s3_storage.delete_file(original_volume_path)
            if prediction_path:
                s3_storage.delete_file(prediction_path)
        else:
            if original_volume_path and os.path.exists(original_volume_path):
                try: 
                    os.remove(original_volume_path)
                    logger.debug(f"Cleaned up local file: {original_volume_path}")
                except OSError: 
                    logger.error(f"Failed to clean up local file: {original_volume_path}")
            if prediction_path and os.path.exists(prediction_path):
                try: 
                    os.remove(prediction_path)
                    logger.info(f"Cleaned up local file: {prediction_path}")
                except OSError: 
                    logger.error(f"Failed to clean up local file: {prediction_path}")
        self.update_state(state=states.FAILURE, meta={'exc': str(e)})
        try:
            publish_status(workflow_id, "error", {"message": str(e)})
        except Exception as e:
            logger.error(f"Error publishing status: {e}")
        raise Ignore()


@celery_app.task(bind=True, name='prepare_3d_data')
def prepare_3d_data(self: WorkflowTask, previous_task_result: dict, workflow_id: str, output_path: str):
    """
    Loads 3D data from .npy file paths (passed from previous task),
    preprocesses them, saves each as a VTK file, and publishes final VTI paths.

    Args:
        previous_task_result (dict): Result from 'process_volume' or 'segment_volume',containing 'original_volume_path' and optionally 'prediction_path'.
        workflow_id (str): The ID for the overall workflow.
        output_path (str): Directory to save VTK files (or S3 prefix).
    """
    def save_volume_as_vti(volume_data, index, output_dir, is_prediction=False):
        """Helper function to convert numpy array to VTK ImageData and save as .vti."""

        # ---Load the data from local *.npy
        if is_prediction:
            # For prediction data, preserve exact values - DO NOT modify at all
            # Convert to float32 without any normalization
            volume_data = np.array(volume_data, dtype=np.float32)
            vtk_data = numpy_to_vtk(volume_data.ravel(order='F'), deep=True, array_type=vtk.VTK_FLOAT)
        else:
            # For original MRI data, do the normal normalization
            volume_data = np.array(volume_data, dtype=np.float32)
            if volume_data.max() > volume_data.min(): # Avoid division by zero
                volume_data = (volume_data - volume_data.min()) / (volume_data.max() - volume_data.min())
            vtk_data = numpy_to_vtk(volume_data.ravel(order='F'), deep=True, array_type=vtk.VTK_FLOAT)
        
        image_data = vtk.vtkImageData()
        image_data.SetDimensions(volume_data.shape[0], volume_data.shape[1], volume_data.shape[2])
        image_data.SetSpacing(1.0, 1.0, 1.0)
        image_data.SetOrigin(0.0, 0.0, 0.0)
        image_data.GetPointData().SetScalars(vtk_data)

        # Determine descriptive name
        descriptive_name = f"Unknown_{index}" # Default
        if is_prediction:
            descriptive_name = "Segmentation"
        else:
            descriptive_name = original_channel_map.get(index, f"Channel_{index}")
        
        # Keep UUID for filename uniqueness on disk, but base name is descriptive
        vti_filename = f"{descriptive_name}_{uuid.uuid4().hex[:8]}.vti"
        
        # --- Save to S3 or local ---
        # Create writer
        writer = vtk.vtkXMLImageDataWriter()
        writer.SetInputData(image_data)
        
        if USE_S3_STORAGE:
            # For S3: Save to memory and upload
            writer.SetWriteToOutputString(1) # write to memory
            writer.Write()
            
            # Get the data as string and save to S3
            s3_key = os.path.join(output_dir, vti_filename).replace('\\', '/')
            s3_storage.save_bytes(writer.GetOutputString(), s3_key)
            return s3_key
        else:
            # For local: Save directly to file
            vti_path = os.path.join(output_dir, vti_filename)
            writer.SetFileName(vti_path)
            writer.Write()
            return vti_path

    # --- Publish status ---
    publish_status(workflow_id, "vtk_preparation_started")
    original_volume_path = previous_task_result.get("original_volume_path")
    prediction_path = previous_task_result.get("prediction_path")

    # Create directory for local file storage
    if not USE_S3_STORAGE:
        os.makedirs(output_path, exist_ok=True)
    else:
        pass
        
    original_paths = []
    prediction_paths = []
    vti_results = {}

    # --- Define Mappings ---
    original_channel_map = {
        0: "T1c",
        1: "T1",
        2: "T2",
        3: "FLAIR"
    }
    
    try:
        # Process Original Volume Channels (load from .npy path)
        if original_volume_path:
            publish_status(workflow_id, "preparation_processing_original")
            try:
                # Load from local file
                original_volume_np = np.load(original_volume_path)
                    
                num_channels = original_volume_np.shape[1]
                for i in range(num_channels):
                     channel_data = original_volume_np[0, i, :, :, :]
                     # Pass is_prediction=False
                     path = save_volume_as_vti(channel_data, i, output_path, is_prediction=False)
                     original_paths.append(path)
                try: os.remove(original_volume_path)
                except OSError: pass
            except Exception as e:
                logger.error(f"Error processing original volume from {original_volume_path}: {e}")
                publish_status(workflow_id, "error", {"message": f"Error processing original volume : {e}"})

        # Process Prediction Labels (load from local.npy path)
        if prediction_path:
            publish_status(workflow_id, "preparation_processing_prediction")
            try:

                    # Load from local file
                prediction_np = np.load(prediction_path)
                                    
                # Just save the prediction data directly as a single VTI file
                # Keep the class values (0,1,2,3,4) intact for frontend color mapping
                path = save_volume_as_vti(prediction_np[0], 0, output_path, is_prediction=True)
                prediction_paths.append(path)
                
                try: os.remove(prediction_path)
                except OSError: pass

            except Exception as e:
                logger.error(f"Error processing prediction volume from {prediction_path}: {e}")
                publish_status(workflow_id, "error", {"message": f"Error processing prediction NPY: {e}"})

        # Final success status update with the paths to the generated VTI files
        vti_results = {'original_paths': original_paths, 'prediction_paths': prediction_paths}
        publish_status(workflow_id, "preparation_completed", vti_results) # Publish VTI paths
        return vti_results

    except Exception as e:
        logger.error(f"Error in prepare_3d_data: {e}")
        self.update_state(state=states.FAILURE, meta={'exc': str(e)})
        try:
            publish_status(workflow_id, "error", {"message": str(e)})
        except Exception as e:
            logger.error(f"Error publishing status: {e}")
        raise Ignore()

