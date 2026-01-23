import React, { useState, useRef } from 'react';
import { useNavigate } from 'react-router-dom';

function UploadReport() {
  const navigate = useNavigate();

  // Steps: 1 = Upload, 2 = Scanning, 3 = Verify
  const [step, setStep] = useState(1);
  const [file, setFile] = useState(null);
  const [date, setDate] = useState('');

  // Data extraction state
  const [extractedData, setExtractedData] = useState(null);

  // UI states
  const [dragActive, setDragActive] = useState(false);
  const [errors, setErrors] = useState({});
  const fileInputRef = useRef(null);

  // Mock extracted data
  const mockScanResult = {
    patientName: 'Guest User',
    reportDate: date || new Date().toISOString().split('T')[0],
    testType: 'Blood Test',
    values: [
      { test: 'Hemoglobin', value: '11.2', unit: 'g/dL', status: 'low' },
      { test: 'Glucose', value: '105', unit: 'mg/dL', status: 'high' }
    ]
  };

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

  const handleSubmit = (e) => {
    e.preventDefault();
    const newErrors = {};

    if (!file) newErrors.file = 'Please select a file';
    if (!date) newErrors.date = 'Please select a date';

    if (Object.keys(newErrors).length > 0) {
      setErrors(newErrors);
      return;
    }

    // Start Scanning Flow
    setStep(2);

    // Simulate OCR delay
    setTimeout(() => {
      setExtractedData({
        ...mockScanResult,
        reportDate: date,
        testType: 'Blood Test'
      });
      setStep(3);
    }, 2500);
  };

  const handleInputChange = (field, value) => {
    setExtractedData(prev => ({
      ...prev,
      [field]: value
    }));
  };

  const handleValueChange = (index, field, value) => {
    setExtractedData(prev => {
      const newValues = [...prev.values];
      newValues[index] = { ...newValues[index], [field]: value };
      return { ...prev, values: newValues };
    });
  };

  const handleAnalyze = () => {
    // In a real app, this is where we'd save the verified data to the backend
    // For now, we navigate to the analysis page with the verified data
    navigate('/analysis', { state: { reportData: extractedData } });
  };

  return (
    <div className="flex-1 min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 px-4 py-8">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <div className="mb-8 text-center md:text-left">
          <h1 className="text-3xl md:text-4xl font-bold text-slate-900 mb-2">
            {step === 1 ? 'Upload Medical Report' : step === 2 ? 'Scanning Document...' : 'Verify Extracted Data'}
          </h1>
          <p className="text-slate-600 text-lg">
            {step === 1
              ? 'Upload your medical reports for AI-powered analysis'
              : step === 2
                ? 'Our AI is extracting values from your report'
                : 'Please review the data below before final analysis'}
          </p>
        </div>

        {/* Step 1: Upload Form */}
        {step === 1 && (
          <form onSubmit={handleSubmit} className="bg-white p-8 rounded-xl shadow-lg border border-slate-200 space-y-6">
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
                className={`w-full px-4 py-3 border rounded-lg focus:outline-none focus:ring-2 focus:ring-green-600 transition ${errors.date ? 'border-red-500' : 'border-slate-300'
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
                className={`border-2 border-dashed rounded-xl p-8 text-center transition ${dragActive
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

            {/* Submit Button */}
            <button
              type="submit"
              className="w-full bg-gradient-to-r from-green-600 to-green-500 text-white font-bold py-4 rounded-lg hover:from-green-700 hover:to-green-600 transition shadow-lg hover:shadow-xl flex items-center justify-center gap-2"
            >
              <i className="fa-solid fa-search"></i>
              Scan & Analyze Report
            </button>
          </form>
        )}

        {/* Step 2: Scanning Animation */}
        {step === 2 && (
          <div className="bg-white rounded-2xl p-16 text-center shadow-sm border border-slate-100 flex flex-col items-center justify-center min-h-[400px]">
            <div className="loader mx-auto mb-6 !w-20 !h-20 !border-4 !border-t-green-600"></div>
            <h3 className="text-2xl font-bold text-slate-800 mb-2">Analyzing Report...</h3>
            <p className="text-slate-500">We are extracting the clinical data from your document.</p>
          </div>
        )}

        {/* Step 3: Verification Form */}
        {step === 3 && extractedData && (
          <div className="bg-white rounded-2xl shadow-lg border border-slate-200 overflow-hidden fade-in">
            <div className="bg-green-600 p-4 text-white flex justify-between items-center">
              <h3 className="font-bold flex items-center gap-2">
                <i className="fa-solid fa-check-circle"></i> Verify Extracted Data
              </h3>
              <span className="text-xs bg-white/20 px-3 py-1 rounded-full font-medium">Please correct any errors</span>
            </div>

            <div className="p-6 md:p-8 space-y-8">
              {/* Header Info */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div>
                  <label className="block text-xs text-slate-500 uppercase font-bold mb-2">Patient Name</label>
                  <input
                    value={extractedData.patientName}
                    onChange={(e) => handleInputChange('patientName', e.target.value)}
                    className="w-full bg-slate-50 border border-slate-200 rounded-lg px-4 py-3 font-medium focus:outline-none focus:ring-2 focus:ring-green-500 transition"
                  />
                </div>
                <div>
                  <label className="block text-xs text-slate-500 uppercase font-bold mb-2">Report Date</label>
                  <input
                    type="date"
                    value={extractedData.reportDate}
                    onChange={(e) => handleInputChange('reportDate', e.target.value)}
                    className="w-full bg-slate-50 border border-slate-200 rounded-lg px-4 py-3 font-medium focus:outline-none focus:ring-2 focus:ring-green-500 transition"
                  />
                </div>
                <div>
                  <label className="block text-xs text-slate-500 uppercase font-bold mb-2">Test Type</label>
                  <input
                    value={extractedData.testType}
                    onChange={(e) => handleInputChange('testType', e.target.value)}
                    className="w-full bg-slate-50 border border-slate-200 rounded-lg px-4 py-3 font-medium focus:outline-none focus:ring-2 focus:ring-green-500 transition"
                  />
                </div>
              </div>

              {/* Values Table */}
              <div className="bg-slate-50 rounded-xl p-6 border border-slate-200">
                <div className="flex justify-between items-center mb-4">
                  <h4 className="font-bold text-slate-700 flex items-center gap-2">
                    <i className="fa-solid fa-list-ul"></i> Extracted Outcomes
                  </h4>
                  <span className="text-xs text-slate-400">Edit values if mismatched</span>
                </div>

                <div className="space-y-3">
                  {/* Table Header */}
                  <div className="hidden md:flex text-xs font-bold text-slate-400 uppercase tracking-wide mb-2 px-2">
                    <div className="flex-1">Test / Biomarker</div>
                    <div className="w-24 text-center">Value</div>
                    <div className="w-20">Unit</div>
                  </div>

                  {extractedData.values.map((val, idx) => (
                    <div key={idx} className="flex flex-col md:flex-row md:items-center gap-3 bg-white p-3 rounded-lg border border-slate-200 shadow-sm">
                      <div className="flex-1">
                        <label className="md:hidden text-xs text-slate-400 font-bold mb-1 block">Test Name</label>
                        <input
                          value={val.test}
                          onChange={(e) => handleValueChange(idx, 'test', e.target.value)}
                          className="w-full font-medium text-slate-800 bg-transparent border-none p-0 focus:ring-0"
                          placeholder="Test Name"
                        />
                      </div>
                      <div className="flex gap-3">
                        <div className="w-full md:w-24">
                          <label className="md:hidden text-xs text-slate-400 font-bold mb-1 block">Value</label>
                          <input
                            value={val.value}
                            onChange={(e) => handleValueChange(idx, 'value', e.target.value)}
                            className="w-full font-bold text-slate-900 bg-slate-50 px-3 py-2 rounded border border-slate-200 focus:outline-none focus:border-green-500 text-center"
                            placeholder="0.0"
                          />
                        </div>
                        <div className="w-full md:w-20">
                          <label className="md:hidden text-xs text-slate-400 font-bold mb-1 block">Unit</label>
                          <input
                            value={val.unit}
                            onChange={(e) => handleValueChange(idx, 'unit', e.target.value)}
                            className="w-full text-slate-500 text-sm bg-transparent border-none p-2 focus:ring-0"
                            placeholder="Unit"
                          />
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Actions */}
              <div className="flex flex-col-reverse md:flex-row justify-end gap-4 pt-6 border-t border-slate-100">
                <button
                  onClick={() => {
                    setStep(1);
                    setExtractedData(null);
                  }}
                  className="px-6 py-3 text-slate-500 font-bold hover:text-slate-800 hover:bg-slate-100 rounded-lg transition"
                >
                  <i className="fa-solid fa-arrow-rotate-left mr-2"></i> Re-upload
                </button>
                <button
                  onClick={handleAnalyze}
                  className="bg-green-600 text-white px-8 py-3 rounded-xl font-bold hover:bg-green-700 transition shadow-lg hover:shadow-green-500/30 flex items-center justify-center gap-2"
                >
                  <i className="fa-solid fa-wand-magic-sparkles"></i>
                  Confirm & Generate Analysis
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default UploadReport;
