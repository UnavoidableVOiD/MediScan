import React from 'react';
import { Link, useParams } from 'react-router-dom';
export default function DoctorReview() {
  const { id } = useParams();
  return (
    <div className="max-w-7xl mx-auto px-4 py-8 fade-in h-[calc(100vh-64px)] flex flex-col">
      <div className="flex justify-between items-center mb-6">
        <Link to="/doctor-dashboard" className="text-slate-500 hover:text-slate-800 flex items-center gap-2">
          <i className="fa-solid fa-arrow-left"></i> Back
        </Link>
        <button className="bg-green-600 text-white px-4 py-2 rounded-lg text-sm hover:bg-green-700 font-medium">
          <i className="fa-solid fa-check mr-2"></i> Approve Report
        </button>
      </div>
      <div className="text-xs text-slate-400 mb-4">Reviewing patient report ID: {id}</div>
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 flex-1 min-h-0">
        <div className="bg-slate-900 rounded-2xl shadow-lg flex items-center justify-center relative overflow-hidden text-slate-500">
          <div className="text-center">
            <i className="fa-solid fa-file-pdf text-6xl mb-4"></i>
            <p>PDF Viewer Simulation</p>
          </div>
        </div>
        <div className="bg-white rounded-2xl shadow-sm border border-slate-200 flex flex-col overflow-hidden">
          <div className="p-4 border-b border-slate-100 bg-slate-50 flex justify-between items-center">
            <h3 className="font-bold text-slate-900">AI Findings</h3>
            <span className="bg-blue-100 text-blue-700 text-xs px-2 py-1 rounded-full font-bold">Confidence: 98%</span>
          </div>
          <div className="p-6 overflow-y-auto space-y-6">
            <div className="bg-red-50 p-4 rounded-lg border border-red-100">
              <div className="font-bold text-red-800 text-sm mb-1">Hemoglobin: 11.2 g/dL (Low)</div>
              <p className="text-xs text-red-600">AI Suggestion: Signs of Anemia. Recommend Iron studies.</p>
            </div>
            <div>
              <label className="block text-xs font-bold text-slate-400 uppercase tracking-wider mb-2">Doctor's Remarks</label>
              <textarea
                className="w-full border border-slate-300 rounded-xl p-3 text-sm focus:outline-none focus:border-blue-500"
                rows="4"
                placeholder="Add clinical notes..."
              ></textarea>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
