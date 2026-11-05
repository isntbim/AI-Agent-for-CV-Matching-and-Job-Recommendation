import React, { useCallback, useState, useEffect } from 'react';
import { useDropzone } from 'react-dropzone';
import { motion, AnimatePresence } from 'framer-motion';
import {
  CloudArrowUpIcon,
  DocumentTextIcon,
  XMarkIcon,
  CheckCircleIcon,
  ExclamationTriangleIcon,
} from '@heroicons/react/24/outline';

interface FileUploadProps {
  onFileSelect: (file: File | null) => void;
  selectedFile?: File | null;
  accept?: Record<string, string[]>;
  maxSize?: number;
  disabled?: boolean;
}

const DEFAULT_ACCEPT: Record<string, string[]> = {
  'application/pdf': ['.pdf'],
  'application/vnd.openxmlformats-officedocument.wordprocessingml.document': [
    '.docx',
  ],
  'text/plain': ['.txt'],
};

const FileUpload: React.FC<FileUploadProps> = ({
  onFileSelect,
  selectedFile: externalSelectedFile,
  accept = DEFAULT_ACCEPT,
  maxSize = 5 * 1024 * 1024, // 5 MB
  disabled = false,
}) => {
  const [uploadedFile, setUploadedFile] = useState<File | null>(
    externalSelectedFile ?? null
  );
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (externalSelectedFile !== undefined) {
      setUploadedFile(externalSelectedFile);
    }
  }, [externalSelectedFile]);

  const onDrop = useCallback(
    (acceptedFiles: File[], rejectedFiles: any[]) => {
      setError(null);

      if (rejectedFiles.length > 0) {
        const rejection = rejectedFiles[0];
        if (rejection.errors[0]?.code === 'file-too-large') {
          setError('File size exceeds the 5 MB limit.');
        } else if (rejection.errors[0]?.code === 'file-invalid-type') {
          setError('Unsupported file format. Upload a .pdf, .docx, or .txt file.');
        } else {
          setError('Could not read the selected file. Please try another file.');
        }
        return;
      }

      if (acceptedFiles.length > 0) {
        const file = acceptedFiles[0];
        setUploadedFile(file);
        onFileSelect(file);
      }
    },
    [onFileSelect]
  );

  const { getRootProps, getInputProps, isDragActive, isDragReject } =
    useDropzone({
      onDrop,
      accept,
      maxSize,
      multiple: false,
      disabled,
    });

  const removeFile = () => {
    setUploadedFile(null);
    setError(null);
    onFileSelect(null);
  };

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const getFileBadge = (fileName: string) => {
    const extension = fileName.split('.').pop()?.toLowerCase();
    switch (extension) {
      case 'pdf':
        return 'PDF';
      case 'docx':
        return 'DOCX';
      case 'txt':
        return 'TXT';
      default:
        return 'FILE';
    }
  };

  return (
    <div className="w-full">
      <AnimatePresence mode="wait">
        {!uploadedFile ? (
          <motion.div
            key="dropzone"
            initial={{ opacity: 0, scale: 0.98 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.98 }}
            transition={{ duration: 0.2 }}
          >
            <div
              {...getRootProps()}
              aria-label="Upload CV file (.pdf, .docx, or .txt up to 5 MB)"
              className={`relative border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-all duration-300 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2 ${
                isDragActive && !isDragReject
                  ? 'border-primary-500 bg-primary-50 scale-[1.01]'
                  : isDragReject || error
                    ? 'border-error-500 bg-error-50'
                    : disabled
                      ? 'border-secondary-200 bg-secondary-50 cursor-not-allowed'
                      : 'border-secondary-300 bg-white hover:border-primary-400 hover:bg-primary-25'
              }`}
            >
              <input {...getInputProps()} aria-label="CV file input" />

              <div className="flex flex-col items-center space-y-4">
                <motion.div
                  animate={{
                    y: isDragActive ? -4 : 0,
                    scale: isDragActive ? 1.08 : 1,
                  }}
                  transition={{ duration: 0.2 }}
                  className={`w-16 h-16 rounded-full flex items-center justify-center ${
                    isDragActive && !isDragReject
                      ? 'bg-primary-100 text-primary-600'
                      : isDragReject || error
                        ? 'bg-error-100 text-error-600'
                        : 'bg-secondary-100 text-secondary-600'
                  }`}
                >
                  {error ? (
                    <ExclamationTriangleIcon className="w-8 h-8" />
                  ) : (
                    <CloudArrowUpIcon className="w-8 h-8" />
                  )}
                </motion.div>

                <div className="space-y-2">
                  {error ? (
                    <div className="space-y-1">
                      <p className="text-error-600 font-medium">{error}</p>
                      <p className="text-sm text-secondary-500">
                        Click or drag another file to try again
                      </p>
                    </div>
                  ) : isDragActive ? (
                    isDragReject ? (
                      <p className="text-error-600 font-medium">
                        Unsupported file type — use PDF, DOCX, or TXT
                      </p>
                    ) : (
                      <p className="text-primary-600 font-medium">
                        Drop your CV here to select it
                      </p>
                    )
                  ) : (
                    <div className="space-y-1">
                      <p className="text-secondary-900 font-medium">
                        Drop your CV here, or{' '}
                        <span className="text-primary-600 underline">
                          browse files
                        </span>
                      </p>
                      <p className="text-sm text-secondary-500">
                        Supports PDF, DOCX, and TXT files up to 5 MB
                      </p>
                    </div>
                  )}
                </div>

                <div className="flex items-center space-x-6 text-xs text-secondary-500">
                  <div className="flex items-center space-x-1.5">
                    <span className="px-1.5 py-0.5 rounded bg-secondary-100 font-semibold text-secondary-700">
                      PDF
                    </span>
                    <span>Text PDF</span>
                  </div>
                  <div className="flex items-center space-x-1.5">
                    <span className="px-1.5 py-0.5 rounded bg-secondary-100 font-semibold text-secondary-700">
                      DOCX
                    </span>
                    <span>Word</span>
                  </div>
                  <div className="flex items-center space-x-1.5">
                    <span className="px-1.5 py-0.5 rounded bg-secondary-100 font-semibold text-secondary-700">
                      TXT
                    </span>
                    <span>Plain text</span>
                  </div>
                </div>
              </div>
            </div>
          </motion.div>
        ) : (
          <motion.div
            key="uploaded-file"
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -12 }}
            transition={{ duration: 0.2 }}
            className="bg-white border border-secondary-200 rounded-xl p-5 shadow-soft"
          >
            <div className="flex items-center justify-between gap-3">
              <div className="flex items-center space-x-3 min-w-0">
                <div className="w-11 h-11 rounded-lg bg-primary-50 border border-primary-200 flex items-center justify-center shrink-0">
                  <span className="text-xs font-bold text-primary-700">
                    {getFileBadge(uploadedFile.name)}
                  </span>
                </div>
                <div className="min-w-0">
                  <h4 className="font-medium text-secondary-900 truncate">
                    {uploadedFile.name}
                  </h4>
                  <div className="flex flex-wrap items-center gap-2 text-xs text-secondary-600 mt-0.5">
                    <span>{formatFileSize(uploadedFile.size)}</span>
                    <span>•</span>
                    <span className="inline-flex items-center space-x-1 text-success-700 font-medium">
                      <CheckCircleIcon className="w-4 h-4" />
                      <span>Ready to match</span>
                    </span>
                  </div>
                </div>
              </div>

              <button
                type="button"
                onClick={removeFile}
                disabled={disabled}
                aria-label={`Remove selected file ${uploadedFile.name}`}
                className="p-2 text-secondary-500 hover:text-error-600 hover:bg-error-50 rounded-lg transition-colors duration-200 focus:outline-none focus:ring-2 focus:ring-primary-500 disabled:opacity-50"
              >
                <XMarkIcon className="w-5 h-5" />
              </button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      <div className="mt-3 flex items-start space-x-2 text-xs text-secondary-600">
        <DocumentTextIcon className="w-4 h-4 text-primary-600 shrink-0 mt-0.5" />
        <span>
          Text-based PDFs, Word (.docx), and plain-text (.txt) CVs are extracted
          directly. Scanned image-only PDFs should be exported as text first.
        </span>
      </div>
    </div>
  );
};

export default FileUpload;
