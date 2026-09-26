import React, { useState, useEffect } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';

export default function CheckReports() {
    const navigate = useNavigate();
    const [searchParams] = useSearchParams();
    const isDemo = searchParams.get('demo') === 'true';

    const [step, setStep] = useState(1); // 1: Upload, 2: Scanning, 3: Verify
    const [file, setFile] = useState(null);
    const [extractedData, setExtractedData] = useState(null);

    // Mock extracted data
    const mockScanResult = {
        patientName: 'Guest User',
        reportDate: '2025-02-24',
        testType: 'CBC (Complete Blood Count)',
        values: [
            { test: 'Hemoglobin', value: '11.2', unit: 'g/dL', status: 'low' },
            { test: 'Glucose', value: '105', unit: 'mg/dL', status: 'high' }
        ]
    };

    const handleFileChange = (e) => {
        if (e.target.files[0]) {
            setFile(e.target.files[0]);
        }
    };

    const startScanning = () => {
        if (!file && !isDemo) return;
        setStep(2);
        // Simulate OCR scanning delay
        setTimeout(() => {
            setExtractedData(mockScanResult);
            setStep(3);
        }, 2500);
    };

    // For Demo: Auto-upload simulation if user clicks "Try Demo Report"
    const useDemoFile = () => {
        setFile({ name: 'demo_lab_report.pdf' });
        setStep(2);
        setTimeout(() => {
            setExtractedData(mockScanResult);
            setStep(3);
        }, 2000);
    };

    const handleAnalyze = () => {
        // In a real app, we would send 'extractedData' to the backend/analysis page
        // For now, we rely on the Analysis page reading MOCK_DATA or we pass state
        navigate('/analysis', { state: { reportData: extractedData } });
    };

    return (
        <div className="max-w-4xl mx-auto px-4 py-12 fade-in">
            <h1 className="text-3xl font-bold text-slate-900 text-center mb-2">
                {step === 1 ? 'Upload Your Report' : step === 2 ? 'Scanning Document...' : 'Verify Extracted Data'}
            </h1>
            <p className="text-center text-slate-500 mb-8">
                {step === 1
                    ? 'Upload a clear photo or PDF of your lab report.'
                    : step === 2
                        ? 'Our AI is extracting values from your report.'
                        : 'Please review the data below before analysis.'}
            </p>

            {/* Step 1: Upload */}
            {step === 1 && (
                <div className="bg-white border-2 border-dashed border-slate-300 rounded-2xl p-10 text-center hover:border-green-500 transition cursor-pointer">
                    <div className="mb-4 text-green-100 bg-green-600 w-16 h-16 rounded-full flex items-center justify-center mx-auto">
                        <i className="fa-solid fa-cloud-arrow-up text-3xl text-white"></i>
                    </div>
                    <h3 className="text-xl font-bold text-slate-800 mb-2">Drag & drop your file here</h3>
                    <p className="text-slate-400 mb-6">Supported formats: JPG, PNG, PDF</p>

                    <input
                        type="file"
                        id="fileInput"
                        className="hidden"
                        onChange={handleFileChange}
                        accept=".jpg,.jpeg,.png,.pdf"
                    />

                    {!file ? (
                        <div className="space-y-3">
                            <button
                                onClick={() => document.getElementById('fileInput').click()}
                                className="bg-green-600 text-white px-8 py-3 rounded-xl font-bold hover:bg-green-700 transition shadow-lg"
                            >
                                Select File
                            </button>
                            {isDemo && (
                                <div>
                                    <span className="text-slate-400 text-sm">or</span><br />
                                    <button
                                        onClick={useDemoFile}
                                        className="mt-2 text-green-600 font-bold hover:underline"
                                    >
                                        Use Demo Report
                                    </button>
                                </div>
                            )}
                        </div>
                    ) : (
                        <div className="space-y-4">
                            <div className="bg-green-50 text-green-700 p-4 rounded-xl inline-block font-medium">
                                <i className="fa-solid fa-file-lines mr-2"></i> {file.name}
                            </div>
                            <br />
                            <button
                                onClick={startScanning}
                                className="bg-green-600 text-white px-8 py-3 rounded-xl font-bold hover:bg-green-700 transition shadow-lg"
                            >
                                Scan & Analyze <i className="fa-solid fa-arrow-right ml-2"></i>
                            </button>
                        </div>
                    )}
                </div>
            )}

            {/* Step 2: Scanning Animation */}
            {step === 2 && (
                <div className="bg-white rounded-2xl p-12 text-center shadow-sm border border-slate-100">
                    <div className="loader mx-auto mb-6 !w-16 !h-16 !border-4 !border-t-green-600"></div>
                    <p className="text-lg font-medium text-slate-700">Analyzing medical terms...</p>
                    <p className="text-sm text-slate-400 mt-2">Extracting vital markers...</p>
                </div>
            )}

            {/* Step 3: Verification */}
            {step === 3 && extractedData && (
                <div className="bg-white rounded-2xl shadow-lg border border-slate-200 overflow-hidden">
                    <div className="bg-green-600 p-4 text-white flex justify-between items-center">
                        <h3 className="font-bold flex items-center gap-2">
                            <i className="fa-solid fa-check-circle"></i> Verification
                        </h3>
                        <span className="text-xs bg-green-500 px-2 py-1 rounded">Wait! Is this correct?</span>
                    </div>

                    <div className="p-6 space-y-6">
                        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                            <div>
                                <label className="block text-xs text-slate-500 uppercase font-bold mb-1">Patient Name</label>
                                <input
                                    defaultValue={extractedData.patientName}
                                    className="w-full bg-slate-50 border border-slate-200 rounded-lg px-3 py-2 font-medium"
                                />
                            </div>
                            <div>
                                <label className="block text-xs text-slate-500 uppercase font-bold mb-1">Report Date</label>
                                <input
                                    defaultValue={extractedData.reportDate}
                                    className="w-full bg-slate-50 border border-slate-200 rounded-lg px-3 py-2 font-medium"
                                />
                            </div>
                            <div>
                                <label className="block text-xs text-slate-500 uppercase font-bold mb-1">Type</label>
                                <input
                                    defaultValue={extractedData.testType}
                                    className="w-full bg-slate-50 border border-slate-200 rounded-lg px-3 py-2 font-medium"
                                />
                            </div>
                        </div>

                        <div className="bg-slate-50 rounded-xl p-4 border border-slate-100">
                            <h4 className="font-bold text-slate-700 mb-3">Extracted Values</h4>
                            {extractedData.values.map((val, idx) => (
                                <div key={idx} className="flex items-center gap-3 mb-2 last:mb-0">
                                    <span className="flex-1 font-medium text-slate-800">{val.test}</span>
                                    <span className="font-bold text-slate-900 bg-white px-2 py-1 rounded border border-slate-200">{val.value}</span>
                                    <span className="text-slate-500 text-sm">{val.unit}</span>
                                </div>
                            ))}
                        </div>

                        <div className="flex justify-end gap-3 pt-4 border-t border-slate-100">
                            <button
                                onClick={() => setStep(1)}
                                className="px-6 py-2 text-slate-500 font-bold hover:text-slate-800"
                            >
                                Re-upload
                            </button>
                            <button
                                onClick={handleAnalyze}
                                className="bg-green-600 text-white px-8 py-3 rounded-xl font-bold hover:bg-green-700 transition shadow-lg flex items-center gap-2"
                            >
                                <i className="fa-solid fa-wand-magic-sparkles"></i> Generate Analysis
                            </button>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}
