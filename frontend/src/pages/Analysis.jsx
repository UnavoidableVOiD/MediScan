import React from 'react';
import { useLocation, useNavigate } from 'react-router-dom';

export default function Analysis() {
    const location = useLocation();
    const navigate = useNavigate();
    const { reportData } = location.state || {}; // Expecting { patientName, reportDate, testType, values: [...] }

    // Fallback if accessed directly without data
    if (!reportData) {
        return (
            <div className="min-h-screen flex flex-col items-center justify-center p-4 text-center">
                <div className="bg-red-50 text-red-600 p-4 rounded-full mb-4">
                    <i className="fa-solid fa-triangle-exclamation text-3xl"></i>
                </div>
                <h2 className="text-2xl font-bold text-slate-900 mb-2">No Report Data Found</h2>
                <p className="text-slate-600 mb-6 max-w-md">
                    It looks like you haven't uploaded or scanned a report yet. Please go back and upload a document to see the analysis.
                </p>
                <button
                    onClick={() => navigate('/check-reports')}
                    className="bg-green-600 text-white px-6 py-3 rounded-xl font-bold hover:bg-green-700 transition"
                >
                    Go to Check Reports
                </button>
            </div>
        );
    }

    return (
        <div className="max-w-5xl mx-auto px-4 py-8 fade-in">
            {/* Header / Summary Card */}
            <div className="bg-white rounded-3xl p-8 shadow-xl border border-slate-100 mb-8 relative overflow-hidden">
                <div className="absolute top-0 left-0 w-full h-2 bg-gradient-to-r from-green-400 to-blue-500"></div>
                <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
                    <div>
                        <div className="flex items-center gap-3 mb-2">
                            <span className="bg-blue-100 text-blue-700 text-xs font-bold px-3 py-1 rounded-full uppercase tracking-wider">
                                {reportData.testType || 'General Report'}
                            </span>
                            <span className="text-slate-400 text-sm font-medium">
                                <i className="fa-regular fa-calendar mr-1"></i> {reportData.reportDate || 'Unknown Date'}
                            </span>
                        </div>
                        <h1 className="text-3xl font-bold text-slate-900">Analysis Result</h1>
                        <p className="text-slate-500">
                            Patient: <span className="font-semibold text-slate-700">{reportData.patientName}</span>
                        </p>
                    </div>
                    <button
                        onClick={() => window.print()}
                        className="bg-slate-100 text-slate-700 px-5 py-2.5 rounded-xl font-bold hover:bg-slate-200 transition flex items-center gap-2"
                    >
                        <i className="fa-solid fa-print"></i> Print / Save PDF
                    </button>
                </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
                {/* Main Analysis Column */}
                <div className="lg:col-span-2 space-y-8">
                    {/* Insights Section */}
                    <div className="bg-white rounded-2xl p-6 shadow-sm border border-slate-100">
                        <h3 className="text-xl font-bold text-slate-900 mb-4 flex items-center gap-2">
                            <i className="fa-solid fa-stethoscope text-green-600"></i> AI Insights
                        </h3>
                        <div className="bg-green-50 border-l-4 border-green-500 p-4 rounded-r-xl">
                            <p className="text-green-800 font-medium leading-relaxed">
                                Based on the extracted values, the **Hemoglobin** levels appear to be slightly low, which may indicate mild anemia. However, **Glucose** levels are within a high-normal range. It is recommended to maintain a balanced diet rich in iron.
                            </p>
                        </div>
                    </div>

                    {/* Detailed Values Table */}
                    <div className="bg-white rounded-2xl p-6 shadow-sm border border-slate-100">
                        <h3 className="text-xl font-bold text-slate-900 mb-4">Detailed Breakdown</h3>
                        <div className="overflow-x-auto">
                            <table className="w-full text-left border-collapse">
                                <thead>
                                    <tr className="border-b border-slate-100 text-xs text-slate-400 uppercase tracking-wider">
                                        <th className="py-3 font-bold">Biomarker</th>
                                        <th className="py-3 font-bold">Value</th>
                                        <th className="py-3 font-bold">Reference Range</th>
                                        <th className="py-3 font-bold">Status</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {reportData.values && reportData.values.map((item, index) => (
                                        <tr key={index} className="border-b border-slate-50 last:border-0 hover:bg-slate-50 transition">
                                            <td className="py-4 font-medium text-slate-700">{item.test}</td>
                                            <td className="py-4">
                                                <span className="font-bold text-slate-900">{item.value}</span>
                                                <span className="text-xs text-slate-400 ml-1">{item.unit}</span>
                                            </td>
                                            <td className="py-4 text-slate-500 text-sm">
                                                {/* Mock ranges based on test names for demo */}
                                                {item.test.includes('Glucose') ? '70 - 99 mg/dL' : '13.5 - 17.5 g/dL'}
                                            </td>
                                            <td className="py-4">
                                                {/* Simple logic for demo status badges */}
                                                {item.status === 'high' ? (
                                                    <span className="bg-red-100 text-red-700 text-xs font-bold px-2 py-1 rounded">High</span>
                                                ) : item.status === 'low' ? (
                                                    <span className="bg-yellow-100 text-yellow-700 text-xs font-bold px-2 py-1 rounded">Low</span>
                                                ) : (
                                                    <span className="bg-green-100 text-green-700 text-xs font-bold px-2 py-1 rounded">Normal</span>
                                                )}
                                            </td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>

                {/* Sidebar Recommendations */}
                <div className="space-y-6">
                    <div className="bg-white rounded-2xl p-6 shadow-sm border border-slate-100">
                        <h3 className="text-lg font-bold text-slate-900 mb-4">Recommended Actions</h3>
                        <ul className="space-y-3">
                            <li className="flex items-start gap-3">
                                <div className="mt-1 w-5 h-5 rounded-full bg-blue-100 flex items-center justify-center shrink-0">
                                    <span className="text-blue-600 text-xs font-bold">1</span>
                                </div>
                                <p className="text-sm text-slate-600">Schedule a follow-up with a General Physician.</p>
                            </li>
                            <li className="flex items-start gap-3">
                                <div className="mt-1 w-5 h-5 rounded-full bg-blue-100 flex items-center justify-center shrink-0">
                                    <span className="text-blue-600 text-xs font-bold">2</span>
                                </div>
                                <p className="text-sm text-slate-600">Increase intake of leafy greens and iron-rich foods.</p>
                            </li>
                            <li className="flex items-start gap-3">
                                <div className="mt-1 w-5 h-5 rounded-full bg-blue-100 flex items-center justify-center shrink-0">
                                    <span className="text-blue-600 text-xs font-bold">3</span>
                                </div>
                                <p className="text-sm text-slate-600">Monitor blood sugar levels daily.</p>
                            </li>
                        </ul>
                        <button className="w-full mt-6 bg-blue-600 text-white py-3 rounded-xl font-bold hover:bg-blue-700 transition">
                            Find a Doctor Near Me
                        </button>
                    </div>

                    <div className="bg-gradient-to-br from-purple-600 to-indigo-700 rounded-2xl p-6 text-white text-center">
                        <div className="w-12 h-12 bg-white/20 rounded-full flex items-center justify-center mx-auto mb-4 backdrop-blur-sm">
                            <i className="fa-solid fa-robot text-2xl"></i>
                        </div>
                        <h3 className="font-bold text-lg mb-2">Have questions?</h3>
                        <p className="text-indigo-100 text-sm mb-4">Chat with our MediScan AI assistant to understand these results better.</p>
                        <button
                            onClick={() => navigate('/patient/chat')}
                            className="bg-white text-indigo-700 px-6 py-2 rounded-lg font-bold text-sm hover:bg-indigo-50 transition"
                        >
                            Start Chat
                        </button>
                    </div>
                </div>
            </div>
        </div>
    );
}
