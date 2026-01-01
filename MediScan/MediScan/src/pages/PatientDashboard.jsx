import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
export default function PatientDashboard() {
  const navigate = useNavigate();
  const [uploading, setUploading] = useState(false);
  const handleUpload = () => {
    setUploading(true);
    setTimeout(() => {
      navigate('/analysis');
    }, 1800);
  };
  return (
    <div className="max-w-4xl mx-auto px-4 py-12 fade-in">
      <h2 className="text-2xl font-bold text-slate-900 mb-2">Dashboard</h2>
      <p className="text-slate-500 mb-8">Upload your medical reports to get instant AI insights.</p>
      {!uploading ? (
        <div
          onClick={handleUpload}
          className="bg-white rounded-2xl shadow-sm border-2 border-dashed border-slate-300 p-12 text-center hover:border-blue-500 transition group cursor-pointer"
        >
          <div className="w-16 h-16 bg-blue-50 text-blue-500 rounded-full flex items-center justify-center mx-auto mb-4 group-hover:scale-110 transition">
            <i className="fa-solid fa-cloud-arrow-up text-2xl"></i>
          </div>
          <h3 className="text-lg font-semibold text-slate-900">Click to upload or drag & drop</h3>
          <p className="text-slate-500 text-sm mt-2">Support for PDF, JPG, PNG (Max 10MB)</p>
        </div>
      ) : (
        <div className="bg-white p-8 rounded-xl shadow-lg border border-slate-100 flex items-center gap-4">
          <div className="loader"></div>
          <div>
            <h4 className="font-semibold text-slate-900">Analyzing Report...</h4>
            <p className="text-sm text-slate-500">Extracting values and checking for abnormalities.</p>
          </div>
        </div>
      )}
      <div className="mt-12">
        <h3 className="text-lg font-bold text-slate-900 mb-4">Recent Reports</h3>
        <div
          className="bg-white p-4 rounded-xl border border-slate-200 shadow-sm flex items-center justify-between hover:bg-slate-50 transition cursor-pointer"
          onClick={() => navigate('/analysis')}
        >
          <div className="flex items-center gap-4">
            <div className="bg-green-100 text-green-600 w-10 h-10 rounded-lg flex items-center justify-center">
              <i className="fa-solid fa-file-pdf"></i>
            </div>
            <div>
              <div className="font-medium text-slate-900">Thyroid Profile.pdf</div>
              <div className="text-xs text-slate-500">Uploaded 2 weeks ago • All Normal</div>
            </div>
          </div>
          <div className="text-slate-400">
            <i className="fa-solid fa-chevron-right"></i>
          </div>
        </div>
      </div>
    </div>
  );
}