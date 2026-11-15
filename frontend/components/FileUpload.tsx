import React, { useCallback, useState, useEffect } from 'react';
import { useDropzone, FileRejection } from 'react-dropzone';
import {
  ArrowUpTrayIcon,
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
      if (externalSelectedFile === null) {
        setError(null);
      }
    }
  }, [externalSelectedFile]);

  const onDrop = useCallback(
    (acceptedFiles: File[], rejectedFiles: FileRejection[]) => {
      setError(null);

      if (rejectedFiles.length > 0) {
        const rejection = rejectedFiles[0];
        const code = rejection.errors[0]?.code;
        if (code === 'file-too-large') {
          setError('File size exceeds the 5 MB limit.');
        } else if (code === 'file-invalid-type') {
          setError('Unsupported file format. Choose a .pdf, .docx, or .txt file.');
        } else {
          setError('Could not read the selected file. Try another file.');
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
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return `${parseFloat((bytes / Math.pow(k, i)).toFixed(1))} ${sizes[i]}`;
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
    <div className="w-full space-y-3">
      {!uploadedFile ? (
        <div
          {...getRootProps()}
          aria-label="Upload CV file (.pdf, .docx, or .txt up to 5 MB)"
          className={`rounded-panel border border-dashed p-6 text-center transition-colors ${
            disabled
              ? 'border-hairline bg-surface opacity-60 cursor-not-allowed'
              : isDragReject || error
                ? 'border-bad-line bg-bad-bg cursor-pointer'
                : isDragActive
                  ? 'border-accent bg-accent-soft cursor-pointer'
                  : 'border-hairline-strong bg-ground hover:border-ink hover:bg-surface cursor-pointer'
          }`}
        >
          <input {...getInputProps()} aria-label="CV file input" />

          <div className="flex flex-col items-center gap-3">
            <div
              className={`w-11 h-11 rounded-ctrl border flex items-center justify-center ${
                isDragReject || error
                  ? 'bg-card border-bad-line text-bad'
                  : isDragActive
                    ? 'bg-card border-accent-line text-accent-text'
                    : 'bg-card border-hairline text-ink'
              }`}
            >
              {error || isDragReject ? (
                <ExclamationTriangleIcon className="w-5 h-5" aria-hidden="true" />
              ) : (
                <ArrowUpTrayIcon className="w-5 h-5" aria-hidden="true" />
              )}
            </div>

            <div className="space-y-1">
              {error ? (
                <>
                  <p className="text-[15px] font-medium text-bad">{error}</p>
                  <p className="text-[13px] text-muted">
                    Select or drop another file to try again.
                  </p>
                </>
              ) : isDragActive ? (
                isDragReject ? (
                  <p className="text-[15px] font-medium text-bad">
                    Unsupported file format — drop a .pdf, .docx, or .txt file.
                  </p>
                ) : (
                  <p className="text-[15px] font-medium text-accent-text">
                    Release to attach this CV file.
                  </p>
                )
              ) : (
                <>
                  <p className="text-[15px] font-medium text-ink">
                    Drop a CV file here or{' '}
                    <span className="underline underline-offset-4 text-accent-text">
                      browse from disk
                    </span>
                  </p>
                  <p className="text-[13px] text-muted">
                    Accepts .pdf, .docx, and .txt up to 5 MB (200+ characters).
                  </p>
                </>
              )}
            </div>

            <div className="flex flex-wrap items-center justify-center gap-2 pt-1 text-[12px] font-mono text-muted">
              <span className="px-2 py-0.5 rounded bg-card border border-hairline text-ink">
                .pdf
              </span>
              <span className="px-2 py-0.5 rounded bg-card border border-hairline text-ink">
                .docx
              </span>
              <span className="px-2 py-0.5 rounded bg-card border border-hairline text-ink">
                .txt
              </span>
            </div>
          </div>
        </div>
      ) : (
        <div className="rounded-panel border border-hairline-strong bg-ground p-4">
          <div className="flex items-center justify-between gap-3">
            <div className="flex items-center gap-3 min-w-0">
              <div className="w-11 h-11 rounded-ctrl bg-accent-soft border border-accent-line flex items-center justify-center shrink-0">
                <span className="text-[12px] font-mono font-medium text-accent-text">
                  {getFileBadge(uploadedFile.name)}
                </span>
              </div>
              <div className="min-w-0">
                <p className="text-[15px] font-medium text-ink truncate">
                  {uploadedFile.name}
                </p>
                <div className="flex flex-wrap items-center gap-2 text-[13px] text-muted mt-0.5">
                  <span className="font-mono tabular-nums">
                    {formatFileSize(uploadedFile.size)}
                  </span>
                  <span aria-hidden="true">·</span>
                  <span className="inline-flex items-center gap-1 text-good font-medium">
                    <CheckCircleIcon className="w-4 h-4 shrink-0" aria-hidden="true" />
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
              className="inline-flex items-center justify-center w-[44px] h-[44px] rounded-ctrl border border-hairline bg-card text-muted hover:text-bad hover:border-bad-line transition-colors shrink-0 disabled:opacity-50"
            >
              <XMarkIcon className="w-5 h-5" aria-hidden="true" />
            </button>
          </div>
        </div>
      )}

      <div className="flex items-start gap-2 text-[13px] text-muted leading-relaxed">
        <DocumentTextIcon
          className="w-4 h-4 text-muted shrink-0 mt-0.5"
          aria-hidden="true"
        />
        <span>
          Text-based PDFs, Word (.docx), and plain-text (.txt) CVs are read
          directly. Scanned image-only PDFs are rejected; export selectable text
          first.
        </span>
      </div>
    </div>
  );
};

export default FileUpload;
