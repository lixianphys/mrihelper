import VtkVolumeViewer from './VtkVolumeViewer';

interface Props {
  originalPath: string;
  predictionPath: string | null;
  overlayEnabled: boolean;
  overlayOpacity: number;
  sliceI: number;
  sliceJ: number;
  sliceK: number;
  zoomLevel: number;
  colorWindow: number;
  colorLevel: number;
  onMetadataLoaded?: (metadata: { extent: [number, number, number, number, number, number], dataRange: [number, number] }) => void;
}

export function MriContainer({
  originalPath,
  predictionPath,
  overlayEnabled,
  overlayOpacity,
  sliceI,
  sliceJ,
  sliceK,
  zoomLevel,
  colorWindow,
  colorLevel,
  onMetadataLoaded
}: Props) {

  // Only render the VTK viewer if we have a valid path
  const shouldRenderViewer = originalPath !== '';
  
  return (
    <>
      {shouldRenderViewer && (
        <VtkVolumeViewer 
          originalPath={originalPath}
          predictionPath={predictionPath || null}
          overlayEnabled ={overlayEnabled}
          overlayOpacity={overlayOpacity}
          sliceI={sliceI}
          sliceJ={sliceJ}
          sliceK={sliceK}
          zoomLevel={zoomLevel}
          colorWindow={colorWindow}
          colorLevel={colorLevel}
          onMetadataLoaded={onMetadataLoaded}
        />
      )}
    </>
  );
} 