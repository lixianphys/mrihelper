/**
 * API response types for the MRI Master application
 */

// Workflow status types
export type WorkflowStatus = 
  | 'connected'
  | 'task_started'
  | 'processing_volume_started'
  | 'processing_volume_completed'
  | 'segmentation_started'
  | 'segmentation_preprocessing'
  | 'segmentation_running_model'
  | 'segmentation_saving_results'
  | 'segmentation_completed'
  | 'preparation_started'
  | 'preparation_processing_original'
  | 'preparation_processing_prediction'
  | 'preparation_completed'
  | 'warning'
  | 'error';

// SSE message structure
export interface WorkflowMessage {
  status: WorkflowStatus;
  data?: any;
}

// Segmentation results
export interface SegmentationResult {
  workflow_id: string;
  original_paths: string[];
  prediction_paths: string[];
}

// Available VTI paths
export interface VtiPaths {
  original_paths: string[];
  prediction_paths: string[];
} 