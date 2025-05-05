import './App.css'
import { useState, useRef, useCallback, useEffect } from 'react'
import { FileUpload } from './components/FileUpload'
import { MriContainer } from './components/MriContainer'
import { Button } from './components/ui/button'
import { Progress } from './components/ui/progress'
import { Alert, AlertTitle, AlertDescription } from './components/ui/alert'
import { Slider } from './components/ui/slider'
import { Switch } from "@/components/ui/switch"

import { mriService } from '@/api/mriService';
import { WorkflowMessage } from '@/types/api';

import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"

import { ScrollArea } from "@/components/ui/scroll-area"

// Read API base URL from environment variable or use default
const apiBaseUrl = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8001'; 

function App() {
  const [file, setFile] = useState<File | null>(null);
  const [_, setWorkflowId] = useState<string | null>(null);
  const [status, setStatus] = useState('Ready to start');
  const [isUploading, setIsUploading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [errorMessage, setErrorMessage] = useState('');
  
  // Logs for SSE events
  const [logs, setLogs] = useState<{time: string, message: string, className: string}[]>([]);
  const logContainerRef = useRef<HTMLDivElement>(null);
  const [showViewer, setShowViewer] = useState(true);

  // State for selected files and overlay
  const [selectedPath, setSelectedPath] = useState<string>('');
  const [predictionPath, setPredictionPath] = useState<string | null>(null);
  const [overlayEnabled, setOverlayEnabled] = useState<boolean>(true);
  const [overlayOpacity, setOverlayOpacity] = useState<number>(0.5);

  // State for available paths
  const [availablePaths, setAvailablePaths] = useState<string[]>([]);

  // State for image metadata and initial slice/window/level values
  const [imageExtent, setImageExtent] = useState<[number, number, number, number, number, number]>([0, 255, 0, 255, 0, 255]);
  const [dataRange, setDataRange] = useState<[number, number]>([0, 255]);

  // Initialize slices and window/level based on default ranges
  const [sliceI, setSliceI] = useState(Math.floor((imageExtent[0] + imageExtent[1]) / 2));
  const [sliceJ, setSliceJ] = useState(Math.floor((imageExtent[2] + imageExtent[3]) / 2));
  const [sliceK, setSliceK] = useState(Math.floor((imageExtent[4] + imageExtent[5]) / 2));
  const [colorWindow, setColorWindow] = useState(dataRange[1] - dataRange[0]);
  const [colorLevel, setColorLevel] = useState(Math.floor((dataRange[0] + dataRange[1]) / 2));
  const [zoomLevel, setZoomLevel] = useState(1);
  const [statusClass, setStatusClass] = useState<string>('text-gray-500');
  const [isDownloadingPrediction, setIsDownloadingPrediction] = useState<boolean>(false);

  // Effect to automatically scroll log container to bottom
  useEffect(() => {
    if (logContainerRef.current) {
      logContainerRef.current.scrollTop = logContainerRef.current.scrollHeight;
    }
  }, [logs]); // Run this effect whenever logs change

  // Add this function to reset slice positions when a new volume is loaded
  const resetSlicePositions = useCallback(() => {
    setSliceI(0);
    setSliceJ(0);
    setSliceK(0);
    setColorWindow(0);
    setColorLevel(0);
  }, []);

  // Handle file selection
  const handleFileSelected = (selectedFile: File) => {
    setFile(selectedFile);
    setStatus(`Selected file: ${selectedFile.name}`);
    setStatusClass('text-gray-500');
    setErrorMessage('');
  };
  
  // Start the workflow process
  const startWorkflow = async (workflowType: string) => {
    if (!file) {
      setErrorMessage('Please select a file first');
      return;
    }
    
    try {
      setIsUploading(true);
      setStatus('Uploading file and starting workflow...');
      setStatusClass('text-blue-500');
      setProgress(10);
      setShowViewer(false);
      setErrorMessage('');
            
      // Upload the file and start the workflow
      const result = await mriService.uploadMriFile(file, workflowType);
      setWorkflowId(result.workflow_id);
      
      // Update status
      setStatus(`Workflow ${result.workflow_id} started. Waiting for updates...`);
      setStatusClass('text-blue-500');
      setProgress(20);
      
      // Connect to SSE for real-time updates
      connectToEvents(result.workflow_id);
      
    } catch (error) {
      console.error('Error starting workflow:', error);
      setErrorMessage(error instanceof Error ? error.message : 'Unknown error');
      setStatusClass('text-red-500');
      setStatus('Error starting workflow');
      setIsUploading(false);
    }
  };

  // Handle download prediction file
  const handleDownloadPrediction = async () => {
    // We need both the path to the prediction VTI and the original NII
    if (!predictionPath) { // predictionPath likely holds the VTI path from prepare_3d_data
      setErrorMessage('No prediction file available to download.');
      return;
    }

    setIsDownloadingPrediction(true);
    setErrorMessage(''); // Clear previous errors
    setStatusClass('text-gray-500'); // Indicate pending download

    try {
      // Construct URL for the NEW backend endpoint
      const downloadUrl = `${apiBaseUrl}/download/segmentation/nifti?prediction_vti_path=${predictionPath}`;

      const response = await fetch(downloadUrl);
      if (!response.ok) {
        // Try to get error detail from backend response
        const errorData = await response.json().catch(() => ({ detail: 'Unknown server error' }));
        throw new Error(`Failed to download file: ${response.status} ${response.statusText} - ${errorData.detail || ''}`);
      }

      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;

      // Try to get suggested filename from Content-Disposition header
      let filename = 'segmentation.nii.gz'; // Default filename
      const disposition = response.headers.get('Content-Disposition');
      if (disposition && disposition.indexOf('attachment') !== -1) {
        const filenameRegex = /filename[^;=\n]*=((['"]).*?\2|[^;\n]*)/;
        const matches = filenameRegex.exec(disposition);
        if (matches != null && matches[1]) {
          filename = matches[1].replace(/['"]/g, '');
        }
      }

      link.setAttribute('download', filename); // Set the filename to .nii.gz

      document.body.appendChild(link);
      link.click();
      link.parentNode?.removeChild(link);
      window.URL.revokeObjectURL(url);
      setStatusClass('text-green-500'); // Indicate success

    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : 'Unknown download error');
      setStatusClass('text-red-500');
    } finally {
      setIsDownloadingPrediction(false);
    }
  };


  // Connect to SSE events
  const connectToEvents = (id: string) => {
    // Set up message handler
    const handleMessage = (message: WorkflowMessage) => {
      const status = message.status;
      const data = message.data;
      
      // Create a timestamp for the log entry
      const timestamp = new Date().toLocaleTimeString();
      
      // Map the status to a user-friendly message
      const statusMap: Record<string, string> = {
        connected: "Connected to workflow...",
        task_started: `Task started: ${data?.task || '...'}`,
        processing_volume_started: "Processing volume...",
        processing_volume_completed: "Volume processed.",
        processing_volume_failed: "Failed to process volume.",
        segmentation_started: "Starting segmentation...",
        segmentation_preprocessing: "Preprocessing for segmentation...",
        segmentation_running_model: "Running segmentation model...",
        segmentation_failed: "Segmentation failed. Retry with a different volume.",
        segmentation_saving_results: "Saving segmentation results...",
        segmentation_completed: "Segmentation complete.",
        preparation_started: "Preparing 3D visualization data...",
        preparation_processing_original: "Processing original volume for visualization...",
        preparation_processing_prediction: "Processing prediction for visualization...",
        preparation_failed: "Failed to prepare visualization data.",
        preparation_completed: "Visualization data ready!",
        warning: `Warning: ${data?.message || 'Unknown issue'}`,
        error: `Error: ${data?.message || 'Unknown error'} (Task: ${data?.task || 'N/A'})`
      };
      
      // Get the message text
      const statusText = statusMap[status] || `Status: ${status}`;
      
      // Update the UI status
      setStatus(statusText);
      
      // Set the status class based on the message type
      if (status === 'error') {
        setStatusClass('text-red-500');
        setErrorMessage(data?.message || 'Unknown error');
        setIsUploading(false);
      } else if (status === 'warning') {
        setStatusClass('text-yellow-500');
      } else if (status === 'preparation_completed') {
        setStatusClass('text-green-500');
        setIsUploading(false);
      } else {
        setStatusClass('text-blue-500');
      }
      
      // Update progress based on status
      updateProgressFromStatus(status);
      
      // Add to logs
      let logClass = 'text-gray-400';
      if (status === 'error') logClass = 'text-red-400';
      if (status === 'warning') logClass = 'text-yellow-400';
      if (status === 'preparation_completed') logClass = 'text-green-400';
      
      setLogs(prev => [...prev, {
        time: timestamp,
        message: statusText,
        className: logClass
      }]);
      
      // Handle specific statuses
      if (status === 'preparation_completed' && data) {
        // Update state with received paths
        setAvailablePaths(data.original_paths || []);
        
        // Set the initially selected path (e.g., the first one)
        if (data.original_paths && data.original_paths.length > 0) {
          setSelectedPath(data.original_paths[0]);
        } else {
          setSelectedPath(''); // Clear path if none received
        }

        // Set prediction path if available
        if (data.prediction_paths && data.prediction_paths.length > 0) {
          setPredictionPath(data.prediction_paths[0]); // Assuming one prediction for now
        } else {
          setPredictionPath(null);
        }
        
        // Reset slice positions
        resetSlicePositions();
        
        // Show the viewer
        setShowViewer(status === 'preparation_completed');
        
        // Disconnect from events
        mriService.disconnectFromEvents();
      }
      
      // Close connection on error
      if (status === 'error') {
        mriService.disconnectFromEvents();
      }
    };
    
    // Set up error handler
    const handleError = (error: Event) => {
      console.error('SSE connection error:', error);
      setErrorMessage('Connection to server lost');
      setStatusClass('text-red-500');
      setStatus('Error receiving updates');
      setIsUploading(false);
    };
    
    // Connect to events
    mriService.connectToEvents(id, handleMessage, handleError);
  };

  // Update progress based on status
  const updateProgressFromStatus = (status: string) => {
    // Map statuses to progress percentages
    const progressMap: Record<string, number> = {
      connected: 25,
      processing_volume_started: 30,
      processing_volume_completed: 40,
      segmentation_started: 45,
      segmentation_preprocessing: 50,
      segmentation_running_model: 60,
      segmentation_saving_results: 80,
      segmentation_completed: 85,
      preparation_started: 90,
      preparation_processing_original: 92,
      preparation_processing_prediction: 95,
      preparation_completed: 100
    };
    
    // Update progress if we have a mapping
    if (progressMap[status]) {
      setProgress(progressMap[status]);
    }
  };

  // Callback to handle metadata loaded from the viewer
  const handleMetadataLoaded = (metadata: { extent: [number, number, number, number, number, number], dataRange: [number, number] }) => {
    setImageExtent(metadata.extent);
    setDataRange(metadata.dataRange);

    // Reset slices to middle based on new extent
    setSliceI(Math.floor((metadata.extent[0] + metadata.extent[1]) / 2));
    setSliceJ(Math.floor((metadata.extent[2] + metadata.extent[3]) / 2));
    setSliceK(Math.floor((metadata.extent[4] + metadata.extent[5]) / 2));

    // Reset window/level based on new data range
    const rangeWidth = metadata.dataRange[1] - metadata.dataRange[0];
    setColorWindow(rangeWidth > 0 ? rangeWidth : 1); // Ensure window is at least 1
    setColorLevel(Math.floor((metadata.dataRange[0] + metadata.dataRange[1]) / 2));
  };

  return (
    <>
      <header className="bg-white shadow-sm">
        <div className="container mx-auto py-4 px-4">
          <h1 className="text-3xl font-bold text-gray-900">AI-powered MRI Helper</h1>
          <p className="text-gray-600">MRI Visualization & AI Segmentation</p>
        </div>
      </header>
      <div className="mb-4 p-2 border rounded-lg shadow-sm grid grid-cols-1 md:grid-cols-3 gap-x-6 gap-y-4">
      {/* File upload section */}
        <div className="mb-6 p-4 bg-white rounded-lg shadow">
          <h3 className="text-xl font-semibold mb-2">Select MRI File (deleted after 24h)</h3>
          <FileUpload 
            onFileSelected={handleFileSelected}
            acceptedFileTypes=".nii,.nii.gz"
            disabled={isUploading}
          />
          <p className="mt-2 text-sm text-gray-500">
            Supported formats: .nii.gz (NIFTI)
          </p>
        </div>
        {/* Logs section */}
        <div className="mb-6 p-4 bg-gray-600 rounded-lg shadow">
          <h3 className="text-xl font-semibold mb-2 text-white">Processing Logs</h3>
          <ScrollArea 
            ref={logContainerRef} 
            className="bg-gray-900 rounded p-2 h-60 text-sm"
          >
            {logs.length === 0 ? (
              <p className="text-gray-400 p-2">Logs will appear here when processing starts...</p>
            ) : (
              logs.map((log, index) => (
                <p key={index} className={`${log.className} px-2 py-1`}>
                  [{log.time}] {log.message}
                </p>
              ))
            )}
          </ScrollArea>
        </div>
        <div className="mb-4 p-2 border rounded-lg shadow-sm grid grid-cols-1 md:grid-rows-1 gap-x-6 gap-y-4">
          {/* Status display */}
          <div className="mb-6 items-center">
            <div className={`font-semibold mb-2 ${statusClass} text-center`}>
              {!errorMessage && status}
            </div>

            {isUploading && (
              <Progress value={progress} className="mb-2" />
            )}
            
            {errorMessage && (
              <Alert variant="destructive" className="mb-2">
                <AlertTitle>Error</AlertTitle>
                <AlertDescription>{errorMessage}</AlertDescription>
              </Alert>
            )}
          </div>

          {/* Workflow actions */}
          <div className="grid grid-cols-1 md:grid-rows-3 gap-2">
            <Button 
              onClick={() => startWorkflow("visualize_only")}
              disabled={!file || isUploading}
              variant="default"
            >
              Process Volume
            </Button>
            
            <Button 
              onClick={() => startWorkflow("segment_and_visualize")}
              disabled={!file || isUploading}
              variant="default"
            >
              Segment Brain Tumor
            </Button>

            {/* Download Prediction Button */}
            <Button 
              onClick={handleDownloadPrediction}
              disabled={!predictionPath || isDownloadingPrediction || isUploading}
              variant="default"
            >
              {isDownloadingPrediction ? 'Downloading...' : 'Download Segmentation (.nii.gz)'}
            </Button>
          </div>
          {/* Path selection dropdown */} 
          {availablePaths.length > 1 && (
              <div className="mt-4 mb-4">
                <label htmlFor="pathSelect" className="block text-sm font-medium text-gray-700 mb-1">
                  Select Volume:
                </label>
                <Select 
                  value={selectedPath} 
                  onValueChange={(value) => setSelectedPath(value)}
                >
                  <SelectTrigger className="w-full">
                    <SelectValue placeholder="Select a volume..." />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectGroup>
                      {availablePaths.map((path) => (
                        <SelectItem key={path} value={path}>
                          {/* Display only the filename */} 
                          {path.split('/').pop()?.split('_')[0] || path}
                        </SelectItem>
                      ))}
                    </SelectGroup>
                  </SelectContent>
                </Select>
              </div>
            )}
        </div>
      </div>
    
      {/* Add the new Combined Controls Container here */} 
      {showViewer && (
        <div className="mb-4 p-2 border rounded-lg shadow-sm grid grid-cols-1 md:grid-cols-3 gap-x-6 gap-y-4">
          
          {/* --- Slice Controls --- */} 
          <div className="space-y-1">
            <h4 className="text-sm font-semibold border-b pb-1 mb-2">Slices</h4>
            <div>
              <label htmlFor="sliceI" className="block text-xs font-medium text-gray-600 px-1">
                Sagittal (I): {sliceI}
              </label>
              <Slider
                id="sliceI"
                min={imageExtent[0]}
                max={imageExtent[1]}
                value={[sliceI]}
                step={1}
                onValueChange={(value) => setSliceI(value[0])}
                className="w-full h-2" 
              />
            </div>
            <div>
              <label htmlFor="sliceJ" className="block text-xs font-medium text-gray-600 px-1">
                Coronal (J): {sliceJ}
              </label>
              <Slider
                id="sliceJ"
                min={imageExtent[2]}
                max={imageExtent[3]}
                value={[sliceJ]}
                step={1}
                onValueChange={(value) => setSliceJ(value[0])}
                className="w-full h-2"
              />
            </div>
            <div>
              <label htmlFor="sliceK" className="block text-xs font-medium text-gray-600 px-1">
                Axial (K): {sliceK}
              </label>
              <Slider
                id="sliceK"
                min={imageExtent[4]}
                max={imageExtent[5]}
                value={[sliceK]}
                step={1}
                onValueChange={(value) => setSliceK(value[0])}
                className="w-full h-2"
              />
            </div>
          </div>

          {/* --- Window/Level Controls --- */} 
          <div className="space-y-1">
             <h4 className="text-sm font-semibold border-b pb-1 mb-2">Window/Level</h4>
            <div>
              <label htmlFor="colorWindow" className="block text-xs font-medium text-gray-600 px-1">
                Window: {colorWindow}
              </label>
              <Slider
                id="colorWindow"
                min={dataRange[1] - dataRange[0] > 0 ? (dataRange[1] - dataRange[0])/10 : 1}
                max={dataRange[1] - dataRange[0] > 0 ? dataRange[1] - dataRange[0] : 1}
                value={[colorWindow]}
                step={dataRange[1] - dataRange[0] > 0 ? (dataRange[1] - dataRange[0]) / 100 : 1}
                onValueChange={(value) => setColorWindow(value[0])}
                className="w-full h-2"
              />
            </div>
            <div>
              <label htmlFor="colorLevel" className="block text-xs font-medium text-gray-600 px-1">
                Level: {colorLevel}
              </label>
              <Slider
                id="colorLevel"
                min={dataRange[0]}
                max={dataRange[1]}
                value={[colorLevel]}
                step={dataRange[1] - dataRange[0] > 0 ? (dataRange[1] - dataRange[0]) / 100 : 1}
                onValueChange={(value) => setColorLevel(value[0])}
                className="w-full h-2"
              />
            </div>
            <div>
              <label htmlFor="zoomLevel" className="block text-xs font-medium text-gray-600 px-1">
                Zoom: {zoomLevel}
              </label>
              <Slider
                id="zoomLevel"
                min={0.5}
                max={1.5}
                value={[zoomLevel]}
                step={0.1}
                onValueChange={(value) => setZoomLevel(value[0])}
                className="w-full h-2"
              />
            </div>
          </div>

          {/* --- Overlay Controls --- */}
          {predictionPath ? (
             <div className="space-y-1">
              <h4 className="text-sm font-semibold border-b pb-1 mb-2">Overlay</h4>
              <div>
                 <label htmlFor="overlayToggle" className="block text-xs font-medium text-gray-600 px-1">
                   Segmentation
                 </label>
                <div className="flex items-center space-x-2 px-1">
                  <Switch 
                    id="overlayToggle"
                    checked={overlayEnabled}
                    onCheckedChange={setOverlayEnabled} 
                  />
                  <span className="text-xs">{overlayEnabled ? 'Enabled' : 'Disabled'}</span>
                </div>
              </div>
              {overlayEnabled && (
                <div>
                  <label htmlFor="overlayOpacity" className="block text-xs font-medium text-gray-600 px-1">
                    Opacity: {overlayOpacity.toFixed(1)}
                  </label>
                  <Slider
                    id="overlayOpacity"
                    min={0}
                    max={1}
                    step={0.1}
                    value={[overlayOpacity]}
                    onValueChange={(value) => setOverlayOpacity(value[0])}
                    className="w-full h-2 px-1"
                  />
                </div>
              )}
            </div>
          ) : ( 
             <div className="space-y-1"> {/* Placeholder if no prediction */} 
                <h4 className="text-sm font-semibold border-b pb-1 mb-2 text-gray-400">Overlay</h4>
                <p className="text-xs text-gray-400 px-1 italic">Segmentation needed for overlay controls.</p>
             </div>
           )}
        </div>
      )}

      {/* MRI Viewer */}
      <MriContainer
      originalPath={selectedPath}
      predictionPath={predictionPath}
      overlayEnabled={overlayEnabled}
      overlayOpacity={overlayOpacity}
      sliceI={sliceI}
      sliceJ={sliceJ}
      sliceK={sliceK}
      zoomLevel={zoomLevel}
      colorWindow={colorWindow}
      colorLevel={colorLevel}
      onMetadataLoaded={handleMetadataLoaded}
      />

      {/* Footer  */}
      <footer className="bg-white py-4 shadow-inner">
        <div className="container mx-auto px-4 text-center text-gray-500 text-sm">
          MRI Helper &copy; {new Date().getFullYear()} - 
          MRI Processing & Segmentation
        </div>
      </footer>
    </>
  );
}

export default App;
