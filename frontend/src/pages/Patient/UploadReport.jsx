import React, { useState, useRef } from 'react';
import { useNavigate } from 'react-router-dom';

function UploadReport() {
  const [file, setFile] = useState(null);
  const [reportType, setReportType] = useState('');
  const [date, setDate] = useState('');
  const [uploading, setUploading] = useState(false);
  const [dragActive, setDragActive] = useState(false);
  const [errors, setErrors] = useState({});
  const fileInputRef = useRef(null);
  const navigate = useNavigate();

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      if (selectedFile.size > 10 * 1024 * 1024) {
        setErrors({ file: 'File size must be less than 10MB' });
        return;
      }
      setFile(selectedFile);
      setErrors({});
    }
  };

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const selectedFile = e.dataTransfer.files[0];
      if (selectedFile.size > 10 * 1024 * 1024) {
        setErrors({ file: 'File size must be less than 10MB' });
        return;
      }
      setFile(selectedFile);
      setErrors({});
    }
  };

  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    const newErrors = {};

    if (!file) newErrors.file = 'Please select a file';
    if (!reportType) newErrors.reportType = 'Please select a report type';
    if (!date) newErrors.date = 'Please select a date';

    if (Object.keys(newErrors).length > 0) {
      setErrors(newErrors);
      return;
    }

    setUploading(true);
    try {
      // Add upload logic here
      console.log('Uploading report:', { file, reportType, date });
      // await uploadReport(file, reportType, date);
      
      // Simulate upload delay
      await new Promise(resolve => setTimeout(resolve, 2000));
      
      // Success - navigate to view reports
      navigate('/patient/view-report');
    } catch (error) {
      setErrors({ submit: 'Error uploading report. Please try again.' });
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="flex-1 min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 px-4 py-8">
      <div className="max-w-3xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-4xl font-bold text-slate-900 mb-2">Upload Medical Report</h1>
          <p className="text-slate-600 text-lg">Upload your medical reports for AI-powered analysis and insights</p>
        </div>

        <form onSubmit={handleSubmit} className="bg-white p-8 rounded-xl shadow-lg border border-slate-200 space-y-6">
          {/* Report Type */}
          <div>
            <label className="block text-sm font-semibold text-slate-700 mb-2">
              Report Type <span className="text-red-500">*</span>
            </label>
            <select
              value={reportType}
              onChange={(e) => {
                setReportType(e.target.value);
                setErrors({ ...errors, reportType: '' });
              }}
              className={`w-full px-4 py-3 border rounded-lg focus:outline-none focus:ring-2 focus:ring-green-600 transition ${
                errors.reportType ? 'border-red-500' : 'border-slate-300'
              }`}
            >
              <option value="">Select report type</option>
              <option value="blood-test">Blood Test</option>
            </select>
            {errors.reportType && (
              <p className="mt-1 text-sm text-red-600">{errors.reportType}</p>
            )}
          </div>

          {/* Report Date */}
          <div>
            <label className="block text-sm font-semibold text-slate-700 mb-2">
              Report Date <span className="text-red-500">*</span>
            </label>
            <input
              type="date"
              value={date}
              onChange={(e) => {
                setDate(e.target.value);
                setErrors({ ...errors, date: '' });
              }}
              max={new Date().toISOString().split('T')[0]}
              className={`w-full px-4 py-3 border rounded-lg focus:outline-none focus:ring-2 focus:ring-green-600 transition ${
                errors.date ? 'border-red-500' : 'border-slate-300'
              }`}
            />
            {errors.date && (
              <p className="mt-1 text-sm text-red-600">{errors.date}</p>
            )}
          </div>

          {/* File Upload */}
          <div>
            <label className="block text-sm font-semibold text-slate-700 mb-2">
              Upload File <span className="text-red-500">*</span>
            </label>
            <div
              onDragEnter={handleDrag}
              onDragLeave={handleDrag}
              onDragOver={handleDrag}
              onDrop={handleDrop}
              className={`border-2 border-dashed rounded-xl p-8 text-center transition ${
                dragActive
                  ? 'border-green-500 bg-green-50'
                  : errors.file
                  ? 'border-red-500 bg-red-50'
                  : 'border-slate-300 bg-slate-50'
              }`}
            >
              <input
                ref={fileInputRef}
                type="file"
                onChange={handleFileChange}
                accept=".pdf,.jpg,.jpeg,.png"
                className="hidden"
                id="file-upload"
              />
              {file ? (
                <div className="space-y-2">
                  <i className="fa-solid fa-file-circle-check text-4xl text-green-600"></i>
                  <div>
                    <p className="font-semibold text-slate-900">{file.name}</p>
                    <p className="text-sm text-slate-600">{formatFileSize(file.size)}</p>
                  </div>
                  <button
                    type="button"
                    onClick={() => {
                      setFile(null);
                      if (fileInputRef.current) fileInputRef.current.value = '';
                    }}
                    className="text-red-600 hover:text-red-700 text-sm font-semibold"
                  >
                    Remove file
                  </button>
                </div>
              ) : (
                <div>
                  <i className="fa-solid fa-cloud-arrow-up text-5xl text-slate-400 mb-4"></i>
                  <label
                    htmlFor="file-upload"
                    className="cursor-pointer block"
                  >
                    <span className="text-green-600 font-semibold hover:text-green-700">
                      Click to upload
                    </span>
                    <span className="text-slate-600"> or drag and drop</span>
                  </label>
                  <p className="text-sm text-slate-500 mt-2">PDF, JPG, PNG (Max 10MB)</p>
                </div>
              )}
            </div>
            {errors.file && (
              <p className="mt-1 text-sm text-red-600">{errors.file}</p>
            )}
          </div>

          {/* Error Message */}
          {errors.submit && (
            <div className="p-4 bg-red-50 border border-red-200 rounded-lg">
              <p className="text-sm text-red-600">{errors.submit}</p>
            </div>
          )}

          {/* Submit Button */}
          <button
            type="submit"
            disabled={uploading}
            className="w-full bg-gradient-to-r from-green-600 to-green-500 text-white font-bold py-4 rounded-lg hover:from-green-700 hover:to-green-600 transition disabled:from-slate-400 disabled:to-slate-400 disabled:cursor-not-allowed shadow-lg hover:shadow-xl flex items-center justify-center gap-2"
          >
            {uploading ? (
              <>
                <i className="fa-solid fa-spinner fa-spin"></i>
                Uploading...
              </>
            ) : (
              <>
                <i className="fa-solid fa-cloud-arrow-up"></i>
                Upload Report
              </>
            )}
          </button>
        </form>
      </div>
    </div>
  );
}

export default UploadReport;
