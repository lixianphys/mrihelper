import { SegmentationResult } from '../types/api';

// Workflow types
export type WorkflowType = 'visualize_only' | 'segment_and_visualize';

// Workflow response
export interface WorkflowResponse {
  workflow_id: string;
  status: string;
  message: string;
}

// VTI paths response
export interface VtiPathsResponse {
  original_paths: string[];
  prediction_paths: string[];
}

// Get the API base URL from environment variable or use default
const apiBaseUrl = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8001';

// Type for SSE message handlers
type MessageHandler = (message: any) => void;

// Track active SSE connections
let currentEventSource: EventSource | null = null;

// MRI service class
export class MriService {
  private apiBaseUrl: string;

  constructor() {
    this.apiBaseUrl = apiBaseUrl;
  }

  /**
   * Upload a MRI file to the server for processing
   */
  async uploadMriFile(file: File, workflowType: string): Promise<WorkflowResponse> {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('workflow_type', workflowType);

    const response = await fetch(`${this.apiBaseUrl}/workflow/start`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const error = await response.text();
      throw new Error(`Upload failed: ${error}`);
    }

    return response.json();
  }

  // Get status of a workflow
  async getWorkflowStatus(workflowId: string): Promise<WorkflowResponse> {
    const response = await fetch(`${this.apiBaseUrl}/workflow/events/${workflowId}`);

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Failed to get workflow status: ${errorText}`);
    }

    return response.json();
  }

  // Download the results of a workflow (prediction VTI file)
  async downloadPrediction(predictionPath: string): Promise<Blob> {
    const response = await fetch(`${this.apiBaseUrl}/datafile?path=${encodeURIComponent(predictionPath)}`);

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Failed to download prediction: ${errorText}`);
    }

    return response.blob();
  }

  /**
   * Get the path to a VTI file
   */
  // getVtiPath(vtiPath: string): string {
  //   return `${this.apiBaseUrl}/datafile?path=${encodeURIComponent(vtiPath)}`;
  // }

  /**
   * Connect to SSE events for a specific workflow
   */
  connectToEvents(workflowId: string, onMessage: MessageHandler, onError?: (error: Event) => void): void {
    // Close any existing connection
    if (currentEventSource) {
      currentEventSource.close();
      currentEventSource = null;
    }

    // Create a new SSE connection
    const eventUrl = `${this.apiBaseUrl}/workflow/events/${workflowId}`;
    currentEventSource = new EventSource(eventUrl);

    // Set up event handlers
    currentEventSource.onopen = () => {
      console.log("SSE connection opened");
    };

    currentEventSource.onmessage = (event) => {
      try {
        const message = JSON.parse(event.data);
        onMessage(message);
      } catch (error) {
        console.error("Error parsing SSE message:", error, "Raw data:", event.data);
      }
    };

    currentEventSource.onerror = (error) => {
      console.error("SSE connection error:", error);
      if (onError) {
        onError(error);
      }
      if (currentEventSource) {
        currentEventSource.close();
        currentEventSource = null;
      }
    };
  }

  /**
   * Disconnect from SSE events
   */
  disconnectFromEvents(): void {
    if (currentEventSource) {
      currentEventSource.close();
      currentEventSource = null;
      console.log("SSE connection closed");
    }
  }

  /**
   * Download segmentation results
   */
  async downloadSegmentation(workflowId: string): Promise<Blob> {
    const response = await fetch(`${this.apiBaseUrl}/workflow/segmentation/${workflowId}/download`);
    
    if (!response.ok) {
      const error = await response.text();
      throw new Error(`Download failed: ${error}`);
    }
    
    return response.blob();
  }

  /**
   * Get segmentation results
   */
  async getSegmentationResults(workflowId: string): Promise<SegmentationResult> {
    const response = await fetch(`${this.apiBaseUrl}/workflow/segmentation/${workflowId}/results`);
    
    if (!response.ok) {
      const error = await response.text();
      throw new Error(`Failed to get results: ${error}`);
    }
    
    return response.json();
  }
}

// Create and export a singleton instance
export const mriService = new MriService(); 