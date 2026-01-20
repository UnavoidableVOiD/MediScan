import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';

function ViewReport() {
  const [reports, setReports] = useState([]);
  const [selectedReport, setSelectedReport] = useState(null);
  const [filter, setFilter] = useState('all');
  const [loading, setLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState('');

  useEffect(() => {
    // Fetch reports from API
    // setReports(data);
    setTimeout(() => setLoading(false), 500);
  }, [filter]);

  const getReportTypeIcon = (type) => {
    const icons = {
      'blood-test': 'fa-vial',
      'ct-scan': 'fa-image',
      'mri': 'fa-magnet',
      'other': 'fa-file-medical'
    };
    return icons[type] || 'fa-file-medical';
  };

  if (loading) {
    return (
      <div className="flex-1 min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 flex items-center justify-center">
        <div className="text-center">
          <i className="fa-solid fa-spinner fa-spin text-4xl text-green-600 mb-4"></i>
          <p className="text-slate-600">Loading reports...</p>
        </div>
      </div>
    );
  }

  const filteredReports = reports.filter(report => {
    const matchesFilter = filter === 'all' || report.type === filter;
    const matchesSearch = searchQuery === '' || 
      report.type.toLowerCase().includes(searchQuery.toLowerCase()) ||
      report.hospital?.toLowerCase().includes(searchQuery.toLowerCase());
    return matchesFilter && matchesSearch;
  });

  return (
    <div className="flex-1 min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 px-4 py-8">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-8 flex items-center justify-between">
          <div>
            <h1 className="text-4xl font-bold text-slate-900 mb-2">My Medical Reports</h1>
            <p className="text-slate-600 text-lg">View and manage your medical reports</p>
          </div>
          <Link
            to="/patient/upload-report"
            className="bg-green-600 text-white px-6 py-3 rounded-lg font-semibold hover:bg-green-700 transition shadow-lg hover:shadow-xl"
          >
            <i className="fa-solid fa-plus mr-2"></i>Upload Report
          </Link>
        </div>

        {/* Search and Filters */}
        <div className="mb-6 space-y-4">
          <div className="relative">
            <i className="fa-solid fa-magnifying-glass absolute left-4 top-1/2 transform -translate-y-1/2 text-slate-400"></i>
            <input
              type="text"
              placeholder="Search reports..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full pl-12 pr-4 py-3 border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-green-600 bg-white"
            />
          </div>
          <div className="flex flex-wrap gap-3">
            {['all', 'blood-test'].map((filterType) => (
              <button
                key={filterType}
                onClick={() => setFilter(filterType)}
                className={`px-5 py-2 rounded-lg font-semibold transition ${
                  filter === filterType
                    ? 'bg-green-600 text-white shadow-lg'
                    : 'bg-white text-slate-700 border border-slate-300 hover:border-green-600'
                }`}
              >
                {filterType === 'all' ? 'All Reports' : filterType.split('-').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ')}
              </button>
            ))}
          </div>
        </div>

        {/* Reports Grid */}
        {filteredReports.length === 0 ? (
          <div className="bg-white rounded-xl shadow-lg border border-slate-200 p-12 text-center">
            <i className="fa-solid fa-folder-open text-6xl text-slate-300 mb-4"></i>
            <h3 className="text-xl font-bold text-slate-900 mb-2">No reports found</h3>
            <p className="text-slate-600 mb-6">
              {reports.length === 0 
                ? "Upload your first report to get started"
                : "No reports match your search criteria"}
            </p>
            {reports.length === 0 && (
              <Link
                to="/patient/upload-report"
                className="inline-block bg-green-600 text-white px-6 py-3 rounded-lg font-semibold hover:bg-green-700 transition"
              >
                Upload Report
              </Link>
            )}
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {filteredReports.map((report) => (
              <div
                key={report.id}
                className="bg-white p-6 rounded-xl shadow-lg border border-slate-200 hover:shadow-2xl transition-all hover:-translate-y-1 cursor-pointer group"
                onClick={() => setSelectedReport(report)}
              >
                <div className="flex items-start justify-between mb-4">
                  <div className={`p-3 rounded-xl bg-gradient-to-br from-green-500 to-green-600`}>
                    <i className={`fa-solid ${getReportTypeIcon(report.type)} text-white text-xl`}></i>
                  </div>
                  <span className={`px-3 py-1 text-xs font-semibold rounded-full ${
                    report.status === 'analyzed' 
                      ? 'bg-green-100 text-green-700'
                      : report.status === 'pending'
                      ? 'bg-yellow-100 text-yellow-700'
                      : 'bg-slate-100 text-slate-700'
                  }`}>
                    {report.status || 'Pending'}
                  </span>
                </div>
                <h3 className="text-lg font-bold text-slate-900 mb-2 group-hover:text-green-600 transition-colors">
                  {report.type?.replace('-', ' ').replace(/\b\w/g, l => l.toUpperCase())}
                </h3>
                <div className="space-y-2 mb-4">
                  <p className="text-slate-600 text-sm">
                    <i className="fa-solid fa-calendar mr-2 text-slate-400"></i>
                    {report.date || 'N/A'}
                  </p>
                  {report.hospital && (
                    <p className="text-slate-600 text-sm">
                      <i className="fa-solid fa-hospital mr-2 text-slate-400"></i>
                      {report.hospital}
                    </p>
                  )}
                </div>
                {report.aiAnalysis && (
                  <div className="mt-4 pt-4 border-t border-slate-200">
                    <p className="text-xs font-semibold text-slate-700 mb-1 uppercase tracking-wide">AI Analysis</p>
                    <p className="text-sm text-slate-600 line-clamp-2">{report.aiAnalysis}</p>
                  </div>
                )}
              </div>
            ))}
          </div>
        )}

        {/* Report Detail Modal */}
        {selectedReport && (
          <div 
            className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50"
            onClick={() => setSelectedReport(null)}
          >
            <div 
              className="bg-white rounded-xl shadow-2xl max-w-4xl w-full max-h-[90vh] overflow-y-auto"
              onClick={(e) => e.stopPropagation()}
            >
              <div className="sticky top-0 bg-white border-b border-slate-200 p-6 flex items-center justify-between z-10">
                <h2 className="text-2xl font-bold text-slate-900">
                  {selectedReport.type?.replace('-', ' ').replace(/\b\w/g, l => l.toUpperCase())}
                </h2>
                <button
                  onClick={() => setSelectedReport(null)}
                  className="text-slate-500 hover:text-slate-700 text-2xl w-8 h-8 flex items-center justify-center rounded-full hover:bg-slate-100 transition"
                >
                  <i className="fa-solid fa-times"></i>
                </button>
              </div>
              <div className="p-6 space-y-6">
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <p className="text-sm font-semibold text-slate-700 mb-1">Report Date</p>
                    <p className="text-slate-900">{selectedReport.date || 'N/A'}</p>
                  </div>
                  {selectedReport.hospital && (
                    <div>
                      <p className="text-sm font-semibold text-slate-700 mb-1">Hospital/Clinic</p>
                      <p className="text-slate-900">{selectedReport.hospital}</p>
                    </div>
                  )}
                  <div>
                    <p className="text-sm font-semibold text-slate-700 mb-1">Status</p>
                    <span className={`px-3 py-1 text-xs font-semibold rounded-full ${
                      selectedReport.status === 'analyzed' 
                        ? 'bg-green-100 text-green-700'
                        : 'bg-yellow-100 text-yellow-700'
                    }`}>
                      {selectedReport.status || 'Pending'}
                    </span>
                  </div>
                </div>
                
                {selectedReport.aiAnalysis && (
                  <div className="bg-green-50 border border-green-200 rounded-lg p-4">
                    <p className="text-sm font-semibold text-green-900 mb-2 flex items-center gap-2">
                      <i className="fa-solid fa-robot"></i>
                      AI Analysis
                    </p>
                    <p className="text-slate-700">{selectedReport.aiAnalysis}</p>
                  </div>
                )}
                
                {selectedReport.fileUrl && (
                  <div>
                    <a
                      href={selectedReport.fileUrl}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="inline-flex items-center gap-2 bg-green-600 text-white px-6 py-3 rounded-lg font-semibold hover:bg-green-700 transition"
                    >
                      <i className="fa-solid fa-file-pdf"></i>
                      View Full Report
                      <i className="fa-solid fa-external-link text-sm"></i>
                    </a>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default ViewReport;
