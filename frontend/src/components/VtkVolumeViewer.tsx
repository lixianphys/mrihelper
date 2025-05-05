import React, { useEffect, useRef, useState } from 'react';
import vtkXMLImageDataReader from '@kitware/vtk.js/IO/XML/XMLImageDataReader';
import { Button } from '@/components/ui/button';

const apiBaseUrl = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8001';

// Define the props for our component
interface Props {
  originalPath: string | null;
  predictionPath: string | null;
  overlayEnabled: boolean;
  overlayOpacity: number;
  sliceI: number;
  sliceJ: number;
  sliceK: number;
  colorWindow: number;
  colorLevel: number;
  zoomLevel: number;
  onMetadataLoaded?: (metadata: { extent: [number, number, number, number, number, number], dataRange: [number, number] }) => void;
}

type ImageData = any;

// Following the main.js approach for rendering VTI files
const VtkVolumeViewer: React.FC<Props> = ({
  originalPath,
  predictionPath,
  overlayEnabled,
  overlayOpacity,
  sliceI,
  sliceJ,
  sliceK,
  colorWindow,
  colorLevel,
  zoomLevel,
  onMetadataLoaded
}) => {
  // References for elements
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const resizeObserverRef = useRef<ResizeObserver | null>(null);
  
  // State for image data
  const [originalImageData, setOriginalImageData] = useState<ImageData | null>(null);
  const [predictionImageData, setPredictionImageData] = useState<ImageData | null>(null);
  const [activeSliceType, setActiveSliceType] = useState<'I' | 'J' | 'K'>('K'); // Default to axial view

  
  // Effect to load original volume data
  useEffect(() => {
    let isMounted = true;
        
    if (originalPath) {
      loadVolume(originalPath)
        .then(imageData => {
          if (isMounted) {
            setOriginalImageData(imageData);
            
            // Update dimensions and extent
            const ext = imageData?.getExtent() || [0, 255, 0, 255, 0, 255];
            const range = imageData?.getPointData().getScalars().getRange() || [0, 1];
            
            // Call the callback with metadata
            if (onMetadataLoaded) {
              onMetadataLoaded({ extent: ext, dataRange: range });
            }       
          }
        })
        .catch(error => {
          console.error('Error loading original volume:', error);
        });
    } else {
    }
    
    return () => {
      isMounted = false;
    };
  }, [originalPath]);
  
  // Effect to load prediction data
  useEffect(() => {
    let isMounted = true;    
    if (predictionPath) {
      loadVolume(predictionPath)
        .then(imageData => {
          if (isMounted) {
            setPredictionImageData(imageData);
          }
        })
        .catch(error => {
          console.error('Error loading prediction volume:', error);
          // Don't set test pattern for prediction
        });
    } else {
      setPredictionImageData(null);
    }
    
    return () => {
      isMounted = false;
    };
  }, [predictionPath]);
  
  // Effect to update canvas size on window resize
  useEffect(() => {
    if (!containerRef.current || !canvasRef.current) return;
    
    const setupCanvasResizing = () => {
      const container = containerRef.current;
      const canvas = canvasRef.current;
      if (!container || !canvas) return;
      
      // Cleanup any existing observer
      if (resizeObserverRef.current) {
        resizeObserverRef.current.disconnect();
      }
            
      // Function to resize canvas
      const resizeCanvas = () => {
        if (!container || !canvas) return;
        
        const containerRect = container.getBoundingClientRect();
        canvas.width = containerRect.width;
        canvas.height = containerRect.height;
                
        // Update the rendering
        renderActiveSlice();
      };
      
      // Create a resize observer
      const observer = new ResizeObserver(() => {
        requestAnimationFrame(resizeCanvas);
      });
      
      observer.observe(container);
      resizeObserverRef.current = observer;
      
      // Initial resize
      resizeCanvas();
      
      // Handle window resize too
      const handleWindowResize = () => {
        requestAnimationFrame(resizeCanvas);
      };
      
      window.addEventListener('resize', handleWindowResize);
      
      return () => {
        window.removeEventListener('resize', handleWindowResize);
        if (resizeObserverRef.current) {
          resizeObserverRef.current.disconnect();
          resizeObserverRef.current = null;
        }
      };
    };
    
    const cleanup = setupCanvasResizing();
    
    return cleanup;
  }, [originalImageData]); // Rerun when image data changes
  
  // Effect to render active slice when data or slice indices change
  useEffect(() => {
    renderActiveSlice();
  }, [originalImageData, predictionImageData, sliceI, sliceJ, sliceK, colorWindow, colorLevel, zoomLevel, activeSliceType, overlayEnabled, overlayOpacity]);
  

  // Function to load a volume from a path
  const loadVolume = async (path: string) => {
    const dataUrl = `${apiBaseUrl}/datafile?path=${encodeURIComponent(path)}`;  
    try {
      
      // Fetch the file using fetch API
      const response = await fetch(dataUrl);
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const arrayBuffer = await response.arrayBuffer();
      
      // Parse VTI file using the reader
      const reader = vtkXMLImageDataReader.newInstance();
      reader.parseAsArrayBuffer(arrayBuffer);
      
      // Get the output data
      const imageData = reader.getOutputData();
      
      return imageData;
    } catch (error) {
      console.error(`Error loading volume from ${path}:`, error);

      return null;
    }
  };
  
  // Function to render a slice to the canvas
  const renderSlice = (sliceType: 'I' | 'J' | 'K', sliceNumber: number) => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext('2d');
    
    if (!ctx || !canvas || !originalImageData) {
      console.warn('Unable to render slice: missing canvas, context, or image data');
      return;
    }
    
    // Extract relevant info from image data
    const dims = originalImageData.getDimensions();
    const ext = originalImageData.getExtent();
    const scalars = originalImageData.getPointData().getScalars();
    
    
    // Determine width and height based on slice type
    let width, height, baseIndex;
    if (sliceType === 'I') {
      width = dims[1];
      height = dims[2];
      baseIndex = sliceNumber - ext[0];
    } else if (sliceType === 'J') {
      width = dims[0];
      height = dims[2];
      baseIndex = sliceNumber - ext[2];
    } else { // K slice (axial)
      width = dims[0];
      height = dims[1];
      baseIndex = sliceNumber - ext[4];
    }
    
    // Create an image data object to hold our slice
    const imgData = ctx.createImageData(width, height);
    const pixels = imgData.data;
    
    
    // Extract data from VTK image data
    try {
      for (let j = 0; j < height; j++) {
        for (let i = 0; i < width; i++) {
          let val;
          if (sliceType === 'I') {
            val = scalars.getTuple(baseIndex + i * dims[0] * dims[1] + j * dims[0])[0];
          } else if (sliceType === 'J') {
            val = scalars.getTuple(i + baseIndex * dims[0] + j * dims[0] * dims[1])[0];
          } else { // K slice
            val = scalars.getTuple(i + j * dims[0] + baseIndex * dims[0] * dims[1])[0];
          }
          
          // Apply window/level
          const windowVal = colorWindow;
          const levelVal = colorLevel;
          let gray = Math.floor(255 * (val - (levelVal - windowVal/2)) / windowVal);
          
          // Clamp to 0-255
          gray = Math.max(0, Math.min(255, gray));
          
          // Set pixel values (RGBA)
          const pixelIndex = (j * width + i) * 4;
          pixels[pixelIndex] = gray;     // R
          pixels[pixelIndex + 1] = gray; // G
          pixels[pixelIndex + 2] = gray; // B
          pixels[pixelIndex + 3] = 255;  // A (fully opaque)
        }
      }
    } catch (error) {
      console.error('Error extracting slice data:', error);
    }
    
    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = 'black';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    // Determine if we should render side-by-side or with overlay
    const hasPrediction = predictionImageData !== null;
    const useSideBySide = hasPrediction && !overlayEnabled;
    
    // Calculate base scale to fit the image(s) in the canvas
    let baseScale;
    if (useSideBySide) {
      // Base scale for side-by-side (half width)
      baseScale = Math.min(canvas.width / 2 / width, canvas.height / height);
    } else {
      // Base scale for single image
      baseScale = Math.min(canvas.width / width, canvas.height / height);
    }

    // Apply zoom level to the base scale
    const scale = baseScale * zoomLevel;

    // Calculate offsets to center the zoomed image
    let offsetX, offsetY;
    if (useSideBySide) {
       // Center the left image in its half
       offsetX = (canvas.width / 2 - width * scale) / 2; 
       offsetY = (canvas.height - height * scale) / 2;
    } else {
        // Center the single image
        offsetX = (canvas.width - width * scale) / 2;
        offsetY = (canvas.height - height * scale) / 2;
    }

    
    // Create a temporary canvas for proper scaling
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = width;
    tempCanvas.height = height;
    const tempCtx = tempCanvas.getContext('2d');
    
    if (!tempCtx) {
      console.error('Failed to get 2D context for temp canvas');
      return;
    }
    
    tempCtx.putImageData(imgData, 0, 0);
    
    // Draw original image
    if (useSideBySide) {
      // In side-by-side mode, draw original on the left half
      ctx.drawImage(tempCanvas, offsetX, offsetY, width * scale, height * scale);
      
      // Label for original
      ctx.fillStyle = 'rgba(255, 255, 255, 0.7)';
      // Adjust label position relative to the potentially scaled/offset image
      const labelY = offsetY + height * scale + 5; // Add some padding
      ctx.fillRect(offsetX, labelY, 150, 20); 
      ctx.fillStyle = 'black';
      ctx.font = '12px Arial';
      ctx.fillText("Original", offsetX + 5, labelY + 15);
      
      // Draw prediction on the right half
      if (hasPrediction) {
        const predImgData = getPredictionSlice(originalImageData, ctx, sliceType, sliceNumber, width, height);
        if (predImgData) {
          // Create and draw prediction
          const predTempCanvas = document.createElement('canvas');
          predTempCanvas.width = width;
          predTempCanvas.height = height;
          const predTempCtx = predTempCanvas.getContext('2d');
          
          if (predTempCtx) {
            predTempCtx.putImageData(predImgData, 0, 0);
            
            // Draw on right half of canvas, adjust offset for the right side
            // The starting X for the right image needs to account for the canvas split
            const predictionOffsetX = canvas.width / 2 + (canvas.width / 2 - width * scale) / 2; // Center in the right half
            ctx.drawImage(predTempCanvas, 
                         predictionOffsetX, 
                         offsetY, 
                         width * scale, 
                         height * scale);
            
            // Label for prediction
            ctx.fillStyle = 'rgba(255, 255, 255, 0.7)';
            // Adjust label position relative to the potentially scaled/offset image
            ctx.fillRect(predictionOffsetX, labelY, 150, 20); 
            ctx.fillStyle = 'black';
            ctx.font = '12px Arial';
            ctx.fillText(`Segmentation`, predictionOffsetX + 5, labelY + 15);
          }
        }
      }
    } else if (overlayEnabled && hasPrediction) {
      // If overlay is enabled, apply overlay
      applyOverlay(imgData, sliceType, sliceNumber, width, height);
      
      // Update tempCanvas with overlay applied
      tempCtx.putImageData(imgData, 0, 0);
      
      // Draw the overlaid image
      ctx.drawImage(tempCanvas, offsetX, offsetY, width * scale, height * scale);
    } else {
      // Just draw original in center
      ctx.drawImage(tempCanvas, offsetX, offsetY, width * scale, height * scale);
    }
    
    // Add slice info overlay
    ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
    ctx.fillRect(5, 5, 150, 80);
    ctx.fillStyle = 'white';
    ctx.font = '14px Arial';
    ctx.fillText(`Slice ${sliceType}: ${sliceNumber}`, 10, 120);
    ctx.fillText(`Window: ${colorWindow.toFixed(2)}`, 10, 140);
    ctx.fillText(`Level: ${colorLevel.toFixed(2)}`, 10, 160);
  };
  
  // Helper function to get prediction slice
  const getPredictionSlice = (
    imageData: any, 
    ctx: CanvasRenderingContext2D, 
    sliceType: 'I' | 'J' | 'K', 
    sliceNumber: number, 
    width: number, 
    height: number
  ) => {
    if (!predictionImageData || !imageData) return null;
    
    // Logic to extract a prediction slice, following main.js approach
    const extent = imageData.getExtent();
    const predictionDims = predictionImageData.getDimensions();
    const predictionExtent = predictionImageData.getExtent();
    const predictionScalars = predictionImageData.getPointData().getScalars();
    
    // Calculate correct prediction slice index
    let predictionIndex;
    if (sliceType === 'I') {
      predictionIndex = Math.max(0, Math.min(
        predictionExtent[1] - predictionExtent[0],
        sliceNumber - extent[0] + (predictionExtent[0] - extent[0])
      ));
    } else if (sliceType === 'J') {
      predictionIndex = Math.max(0, Math.min(
        predictionExtent[3] - predictionExtent[2],
        sliceNumber - extent[2] + (predictionExtent[2] - extent[2])
      ));
    } else { // K slice
      predictionIndex = Math.max(0, Math.min(
        predictionExtent[5] - predictionExtent[4],
        sliceNumber - extent[4] + (predictionExtent[4] - extent[4])
      ));
    }
    
    // Create image data for prediction
    const imgData = ctx.createImageData(width, height);
    const pixels = imgData.data;
    
    // Define discrete colors for each class
    const classColors = {
      0: [0, 0, 0],       // Background: Black
      1: [0, 255, 0],     // Class 1: Green (TC - Tumor Core)
      2: [255, 0, 0],     // Class 2: Red (WT - Whole Tumor)
      3: [255, 255, 0]    // Class 3: Yellow (ET - Enhancing Tumor)
    };
    
    // Extract prediction data
    for (let j = 0; j < height; j++) {
      for (let i = 0; i < width; i++) {
        let predVal = 0;
        // Calculate the correct index in prediction data
        let idx;
        if (sliceType === 'I') {
          idx = predictionIndex + i * predictionDims[0] * predictionDims[1] + j * predictionDims[0];
        } else if (sliceType === 'J') {
          idx = i + predictionIndex * predictionDims[0] + j * predictionDims[0] * predictionDims[1];
        } else { // K slice
          idx = i + j * predictionDims[0] + predictionIndex * predictionDims[0] * predictionDims[1];
        }
        
        try {
          if (idx >= 0 && idx < predictionScalars.getNumberOfTuples()) {
            // Use exact value - don't round or modify
            predVal = predictionScalars.getTuple(idx)[0];
          }
        } catch (error) {
          console.error('Error getting prediction value:', error);
        }
        
        // Get the color for this class
        const color = classColors[predVal as keyof typeof classColors] || [0, 0, 0];
        
        // Set pixel values with class colors
        const pixelIndex = (j * width + i) * 4;
        pixels[pixelIndex] = color[0];     // R
        pixels[pixelIndex + 1] = color[1]; // G
        pixels[pixelIndex + 2] = color[2]; // B
        pixels[pixelIndex + 3] = 255;      // A (fully opaque)
      }
    }
    
    return imgData;
  };
  
  // Helper function to apply overlay (segmentation over original image)
  const applyOverlay = (
    imgData: ImageData, 
    sliceType: 'I' | 'J' | 'K', 
    sliceNumber: number, 
    width: number, 
    height: number
  ) => {
    if (!predictionImageData || !originalImageData) return;
    
    try {
      const extent = originalImageData.getExtent();
      const predictionDims = predictionImageData.getDimensions();
      const predictionExtent = predictionImageData.getExtent();
      const predictionScalars = predictionImageData.getPointData().getScalars();
      
      // Define discrete colors for each class, same as in main.js
      const classColors = {
        0: [0, 0, 0],       // Background: Black (transparent in overlay)
        1: [0, 255, 0],     // Class 1: Green (TC - Tumor Core)
        2: [255, 0, 0],     // Class 2: Red (WT - Whole Tumor)
        3: [255, 255, 0]    // Class 3: Yellow (ET - Enhancing Tumor)
      };
      
      // Calculate correct prediction slice index
      let predictionIndex;
      if (sliceType === 'I') {
        predictionIndex = Math.max(0, Math.min(
          predictionExtent[1] - predictionExtent[0],
          sliceNumber - extent[0] + (predictionExtent[0] - extent[0])
        ));
      } else if (sliceType === 'J') {
        predictionIndex = Math.max(0, Math.min(
          predictionExtent[3] - predictionExtent[2],
          sliceNumber - extent[2] + (predictionExtent[2] - extent[2])
        ));
      } else { // K slice
        predictionIndex = Math.max(0, Math.min(
          predictionExtent[5] - predictionExtent[4],
          sliceNumber - extent[4] + (predictionExtent[4] - extent[4])
        ));
      }
      
      // Apply overlay with transparency
      for (let j = 0; j < height; j++) {
        for (let i = 0; i < width; i++) {
          let predVal = 0;
          
          // Calculate the correct index in prediction data
          let idx;
          if (sliceType === 'I') {
            idx = predictionIndex + i * predictionDims[0] * predictionDims[1] + j * predictionDims[0];
          } else if (sliceType === 'J') {
            idx = i + predictionIndex * predictionDims[0] + j * predictionDims[0] * predictionDims[1];
          } else { // K slice
            idx = i + j * predictionDims[0] + predictionIndex * predictionDims[0] * predictionDims[1];
          }
          
          try {
            if (idx >= 0 && idx < predictionScalars.getNumberOfTuples()) {
              // Use exact value - don't round or modify
              predVal = predictionScalars.getTuple(idx)[0];
            }
          } catch (error) {
            console.error('Error getting prediction value for overlay:', error);
          }
          
          // Only overlay non-zero values (assuming prediction is a mask/segmentation)
          if (predVal > 0) {
            const color = classColors[predVal as keyof typeof classColors] || [0, 0, 0];
            
            const pixelIndex = (j * width + i) * 4;
            // Blend with original using alpha compositing (same as main.js)
            imgData.data[pixelIndex] = Math.round((1 - overlayOpacity) * imgData.data[pixelIndex] + 
                                      overlayOpacity * color[0]);
            imgData.data[pixelIndex + 1] = Math.round((1 - overlayOpacity) * imgData.data[pixelIndex + 1] + 
                                          overlayOpacity * color[1]);
            imgData.data[pixelIndex + 2] = Math.round((1 - overlayOpacity) * imgData.data[pixelIndex + 2] + 
                                          overlayOpacity * color[2]);
          }
        }
      }
    } catch (error) {
      console.error('Error applying overlay:', error);
    }
  };
  
  // Function to render the active slice
  const renderActiveSlice = () => {
    if (!originalImageData) return;
    
    // Get the correct slice number based on active slice type
    let sliceNumber;
    if (activeSliceType === 'I') {
      sliceNumber = sliceI;
    } else if (activeSliceType === 'J') {
      sliceNumber = sliceJ;
    } else { // K
      sliceNumber = sliceK;
    }
      renderSlice(activeSliceType, sliceNumber);
  };
  
  // Handle view type change
  const handleViewTypeChange = (sliceType: 'I' | 'J' | 'K') => {
    setActiveSliceType(sliceType);
    
    // Update button styles immediately
    const buttons = document.querySelectorAll('.slice-view-btn');
    buttons.forEach(btn => {
      if (btn.textContent?.includes(sliceType)) {
        btn.classList.add('bg-blue-500', 'text-white');
        btn.classList.remove('bg-gray-200', 'hover:bg-gray-600');
      } else {
        btn.classList.remove('bg-blue-500', 'text-white');
        btn.classList.add('bg-gray-200', 'hover:bg-gray-600');
      }
    });
  };
  
  return (
    <div 
      ref={containerRef} 
      className={`relative`} 
      style={{ width: '100%', height: '100%', minHeight: '400px' }}
    >
      {/* View selection buttons */}
      <div className="absolute top-0 left-0 right-0 flex justify-center z-10 p-2 bg-black bg-opacity-30">
        <Button 
          className={`slice-view-btn px-3 py-1 mx-1 rounded ${activeSliceType === 'I' ? 'bg-green-500 text-white' : 'bg-gray-400 hover:bg-gray-600'}`}
          onClick={() => handleViewTypeChange('I')}
        >
          Sagittal (I)
        </Button>
        <Button 
          className={`slice-view-btn px-3 py-1 mx-1 rounded ${activeSliceType === 'J' ? 'bg-blue-500 text-white' : 'bg-gray-400 hover:bg-gray-600'}`}
          onClick={() => handleViewTypeChange('J')}
        >
          Coronal (J)
        </Button>
        <Button 
          className={`slice-view-btn px-3 py-1 mx-1 rounded ${activeSliceType === 'K' ? 'bg-red-500 text-white' : 'bg-gray-400 hover:bg-gray-600'}`}
          onClick={() => handleViewTypeChange('K')}
        >
          Axial (K)
        </Button>
      </div>
      
      {/* The actual canvas where we render */}
      <canvas 
        ref={canvasRef} 
        className="w-full h-full bg-black" 
        style={{ display: 'block' }}
      />
      
      {/* Segmentation Legend */}
      {predictionImageData && ( // Only show legend if segmentation data is loaded
        <div className="absolute bottom-2 right-2 bg-gray-800 bg-opacity-70 text-white text-xs p-2 rounded z-10">
          <ul>
            <li className="flex items-center mb-1">
              <span className="w-3 h-3 inline-block mr-2 bg-green-500"></span> TC (Tumor Core)
            </li>
            <li className="flex items-center mb-1">
              <span className="w-3 h-3 inline-block mr-2 bg-red-500"></span> WT (Whole Tumor)
            </li>
            <li className="flex items-center">
              <span className="w-3 h-3 inline-block mr-2 bg-yellow-500"></span> ET (Enhancing Tumor)
            </li>
             {/* Note: Background (0) is usually transparent or black */}
          </ul>
        </div>
      )}
    </div>
  );
};

export default VtkVolumeViewer; 