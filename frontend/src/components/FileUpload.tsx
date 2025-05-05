import React, { useState, useRef } from 'react';
import { Button } from '@/components/ui/button';

export interface FileUploadProps {
  acceptedFileTypes?: string;
  disabled?: boolean;
  onFileSelected: (file: File) => void;
}

export function FileUpload({ 
  acceptedFileTypes = "*", 
  disabled = false,
  onFileSelected 
}: FileUploadProps) {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [dragActive, setDragActive] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleDrag = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0] && !disabled) {
      const file = e.dataTransfer.files[0];
      setSelectedFile(file);
      onFileSelected(file);
    }
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    e.preventDefault();
    
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setSelectedFile(file);
      onFileSelected(file);
    }
  };

  const handleButtonClick = () => {
    if (inputRef.current) {
      inputRef.current.click();
    }
  };

  return (
    <div 
      className={`
        flex flex-col items-center justify-center w-full
        border-2 border-dashed rounded-lg p-6 
        transition-colors duration-300 ease-in-out
        ${dragActive ? 'border-blue-500 bg-blue-50' : 'border-gray-300'}
        ${disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}
      `}
      onDragEnter={handleDrag}
      onDragLeave={handleDrag}
      onDragOver={handleDrag}
      onDrop={handleDrop}
    >
      <input
        ref={inputRef}
        type="file"
        accept={acceptedFileTypes}
        onChange={handleChange}
        disabled={disabled}
        className="hidden"
      />
      
      <div className="flex flex-col items-center justify-center space-y-2">
        <svg 
          className="w-12 h-12 text-gray-400" 
          xmlns="http://www.w3.org/2000/svg" 
          fill="none" 
          viewBox="0 0 24 24" 
          stroke="currentColor"
        >
          <path 
            strokeLinecap="round" 
            strokeLinejoin="round" 
            strokeWidth={2} 
            d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" 
          />
        </svg>
        
        <p className="text-lg font-medium text-gray-700">
          Drag and drop your file here, or
        </p>
        
        <Button 
          type="button"
          onClick={handleButtonClick}
          disabled={disabled}
        >
          Select File
        </Button>
        
        {selectedFile && (
          <p className="mt-2 text-sm text-gray-500">
            Selected: {selectedFile.name}
          </p>
        )}
      </div>
    </div>
  );
} 