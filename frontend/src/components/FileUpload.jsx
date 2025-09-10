import { useState, useRef } from "react";

export default function FileUpload({ 
  onUpload, 
  multiple = false, 
  acceptedFileTypes = ""
}) {
  const [files, setFiles] = useState([]);
  const [dragActive, setDragActive] = useState(false);
  const inputRef = useRef(null);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (files.length > 0) onUpload(multiple ? files : files[0]);
  };

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      if (multiple) {
        setFiles(Array.from(e.dataTransfer.files));
      } else {
        setFiles([e.dataTransfer.files[0]]);
      }
    }
  };

  const handleFileChange = (e) => {
    if (e.target.files.length > 0) {
      if (multiple) {
        setFiles(Array.from(e.target.files));
      } else {
        setFiles([e.target.files[0]]);
      }
    }
  };

  const removeFile = (index) => {
    setFiles(files.filter((_, i) => i !== index));
  };

  const handleButtonClick = () => {
    inputRef.current.click();
  };

  return (
    <form 
      onSubmit={handleSubmit} 
      onDragEnter={handleDrag}
      className="w-full"
    >
      <div className={`
        flex flex-col items-center justify-center w-full h-48 
        border-2 border-dashed rounded-lg 
        transition-colors
        ${dragActive ? 'border-purple-500 bg-purple-50' : 'border-gray-300 bg-gray-50'} 
        ${files.length > 0 ? 'bg-emerald-50 border-emerald-300' : ''}
      `}>
        <input
          ref={inputRef}
          type="file"
          className="hidden"
          onChange={handleFileChange}
          accept={acceptedFileTypes || ".pdf,.doc,.docx,.txt,.py,.java,.js,.jsx,.ts,.tsx,.c,.cpp,.cs,.html,.css"}
          multiple={multiple}
        />
        
        <div 
          className="flex flex-col items-center justify-center pt-5 pb-6 px-4 text-center"
          onClick={handleButtonClick}
        >
          {files.length === 0 ? (
            <>
              <svg 
                className="w-10 h-10 mb-3 text-gray-500" 
                fill="none" 
                stroke="currentColor" 
                viewBox="0 0 24 24" 
                xmlns="http://www.w3.org/2000/svg"
              >
                <path 
                  strokeLinecap="round" 
                  strokeLinejoin="round" 
                  strokeWidth="2" 
                  d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
                ></path>
              </svg>
              <p className="mb-2 text-sm text-gray-600">
                <span className="font-semibold">Click to upload</span> or drag and drop
              </p>
              <p className="text-xs text-gray-500">
                {multiple 
                  ? "Upload multiple files - code, documents, and text files (MAX. 10MB each)" 
                  : "PDF, DOC, DOCX or TXT (MAX. 10MB)"}
              </p>
              {acceptedFileTypes && (
                <p className="text-xs text-gray-400 mt-1">
                  Accepted formats: {acceptedFileTypes.split(',').map(ext => ext.replace('.', '')).join(', ')}
                </p>
              )}
            </>
          ) : (
            <>
              <svg 
                className="w-10 h-10 mb-3 text-emerald-500" 
                fill="none" 
                stroke="currentColor" 
                viewBox="0 0 24 24" 
                xmlns="http://www.w3.org/2000/svg"
              >
                <path 
                  strokeLinecap="round" 
                  strokeLinejoin="round" 
                  strokeWidth="2" 
                  d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"
                ></path>
              </svg>
              <p className="mb-2 text-sm font-medium text-emerald-600">
                {files.length === 1 
                  ? "File ready for upload" 
                  : `${files.length} files ready for upload`}
              </p>
              {files.length === 1 && (
                <p className="text-xs text-emerald-500">{files[0].name}</p>
              )}
            </>
          )}
        </div>
        
        {dragActive && 
          <div 
            className="absolute inset-0" 
            onDragEnter={handleDrag} 
            onDragLeave={handleDrag} 
            onDragOver={handleDrag} 
            onDrop={handleDrop}
          ></div>
        }
      </div>

      {files.length > 1 && (
        <div className="mt-4 border border-gray-200 rounded-md p-2 max-h-48 overflow-y-auto">
          <p className="text-sm font-medium text-gray-700 mb-2">Selected Files:</p>
          <ul className="space-y-1">
            {files.map((file, index) => (
              <li key={index} className="flex items-center justify-between text-sm bg-gray-50 p-2 rounded">
                <span className="truncate max-w-[80%]">{file.name}</span>
                <button
                  type="button"
                  onClick={() => removeFile(index)}
                  className="text-red-500 hover:text-red-700"
                >
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              </li>
            ))}
          </ul>
        </div>
      )}

      <div className="mt-6 flex justify-center">
        <button
          type="submit"
          disabled={files.length === 0}
          className={`
            px-6 py-3 rounded-md font-medium text-white 
            transition-colors duration-200
            ${files.length > 0
              ? 'bg-purple-700 hover:bg-purple-800 shadow-md' 
              : 'bg-gray-400 cursor-not-allowed'
            }
          `}
        >
          {multiple ? "Compare Files" : "Analyze Document"}
        </button>
      </div>
    </form>
  );
}
