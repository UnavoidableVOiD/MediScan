import React, { useState, useEffect, useRef } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';

export default function CheckReports() {
    const navigate = useNavigate();
    // const [searchParams] = useSearchParams();
    const isDemo = true; // Always allow demo button to work locally for user understanding


    const [step, setStep] = useState(1); // 1: Upload, 2: Scanning, 3: Verify
    const [file, setFile] = useState(null);
    const [extractedData, setExtractedData] = useState(null);

    // Camera states
    const [isCameraOpen, setIsCameraOpen] = useState(false);
    const videoRef = useRef(null);
    const canvasRef = useRef(null);

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

    const startCamera = async () => {
        setIsCameraOpen(true);
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ video: true });
            if (videoRef.current) {
                videoRef.current.srcObject = stream;
            }
        } catch (err) {
            console.error("Error accessing camera:", err);
            alert("Could not access camera. Please ensure you have granted permission.");
            setIsCameraOpen(false);
        }
    };

    const stopCamera = () => {
        if (videoRef.current && videoRef.current.srcObject) {
            const tracks = videoRef.current.srcObject.getTracks();
            tracks.forEach(track => track.stop());
            videoRef.current.srcObject = null;
        }
        setIsCameraOpen(false);
    };

    const captureImage = () => {
        if (videoRef.current && canvasRef.current) {
            const context = canvasRef.current.getContext('2d');
            // Set canvas dimensions to match video
            canvasRef.current.width = videoRef.current.videoWidth;
            canvasRef.current.height = videoRef.current.videoHeight;

            context.drawImage(videoRef.current, 0, 0, canvasRef.current.width, canvasRef.current.height);

            canvasRef.current.toBlob((blob) => {
                const capturedFile = new File([blob], "captured_report.jpg", { type: "image/jpeg" });
                setFile(capturedFile);
                stopCamera();
            }, 'image/jpeg');
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
        setFile({ name: 'demo_lab_report.png', preview: '/demo_report.png' });
        setStep(2);
        setTimeout(() => {
            setExtractedData(mockScanResult);
            setStep(3);
        }, 500);
    };



    const handleAnalyze = () => {
        // In a real app, we would send 'extractedData' to the backend/analysis page
        // For now, we rely on the Analysis page reading MOCK_DATA or we pass state
        navigate('/analysis', { state: { reportData: extractedData } });
    };

    useEffect(() => {
        return () => {
            stopCamera(); // Cleanup on unmount
        };
    }, []);

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

                    {!isCameraOpen ? (
                        <>
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
                                    <div className="flex justify-center gap-3">
                                        <button
                                            onClick={() => document.getElementById('fileInput').click()}
                                            className="bg-green-600 text-white px-6 py-3 rounded-xl font-bold hover:bg-green-700 transition shadow-lg"
                                        >
                                            <i className="fa-solid fa-file-arrow-up mr-2"></i> Select File
                                        </button>
                                        <button
                                            onClick={startCamera}
                                            className="bg-blue-600 text-white px-6 py-3 rounded-xl font-bold hover:bg-blue-700 transition shadow-lg lg:hidden"
                                        >
                                            <i className="fa-solid fa-camera mr-2"></i> Take Photo
                                        </button>
                                    </div>

                                    <div className="mt-6 border-t border-slate-100 pt-4">
                                        <p className="text-xs text-slate-400 mb-2">New here? See how it works:</p>
                                        <button
                                            onClick={useDemoFile}
                                            className="text-sm text-green-600 font-bold hover:underline bg-green-50 px-4 py-2 rounded-lg"
                                        >
                                            <i className="fa-solid fa-flask mr-2"></i> Try Demo Report
                                        </button>
                                    </div>
                                </div>

                            ) : (
                                <div className="space-y-4">
                                    <div className="bg-green-50 text-green-700 p-4 rounded-xl inline-block font-medium">
                                        <i className="fa-solid fa-file-lines mr-2"></i> {file.name}
                                    </div>
                                    <br />
                                    <div className="flex justify-center gap-3">
                                        <button
                                            onClick={() => setFile(null)}
                                            className="bg-slate-200 text-slate-700 px-6 py-3 rounded-xl font-bold hover:bg-slate-300 transition"
                                        >
                                            Change File
                                        </button>
                                        <button
                                            onClick={startScanning}
                                            className="bg-green-600 text-white px-8 py-3 rounded-xl font-bold hover:bg-green-700 transition shadow-lg"
                                        >
                                            Scan & Analyze <i className="fa-solid fa-arrow-right ml-2"></i>
                                        </button>
                                    </div>
                                </div>
                            )}
                        </>
                    ) : (
                        <div className="relative max-w-lg mx-auto">
                            <video
                                ref={videoRef}
                                autoPlay
                                playsInline
                                className="w-full rounded-xl border-2 border-slate-200 shadow-lg"
                                onLoadedMetadata={() => videoRef.current.play()}
                            />
                            <canvas ref={canvasRef} className="hidden" />
                            <div className="flex justify-center gap-3 mt-4">
                                <button
                                    onClick={stopCamera}
                                    className="bg-slate-500 text-white px-6 py-2 rounded-lg font-bold hover:bg-slate-600 transition"
                                >
                                    Cancel
                                </button>
                                <button
                                    onClick={captureImage}
                                    className="bg-green-600 text-white px-6 py-2 rounded-lg font-bold hover:bg-green-700 transition shadow-lg"
                                >
                                    <i className="fa-solid fa-camera mr-2"></i> Capture
                                </button>
                            </div>
                        </div>
                    )}
                </div>
            )
            }

            {/* Step 2: Scanning Animation */}
            {
                step === 2 && (
                    <div className="bg-white rounded-2xl p-12 text-center shadow-sm border border-slate-100">
                        <div className="loader mx-auto mb-6 !w-16 !h-16 !border-4 !border-t-green-600"></div>
                        <p className="text-lg font-medium text-slate-700">Analyzing medical terms...</p>
                        <p className="text-sm text-slate-400 mt-2">Extracting vital markers...</p>
                    </div>
                )
            }

            {/* Step 3: Verification */}
            {
                step === 3 && extractedData && (
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
                                        value={extractedData.patientName}
                                        onChange={(e) => handleInputChange('patientName', e.target.value)}
                                        className="w-full bg-slate-50 border border-slate-200 rounded-lg px-3 py-2 font-medium focus:outline-none focus:border-green-500"
                                    />
                                </div>
                                <div>
                                    <label className="block text-xs text-slate-500 uppercase font-bold mb-1">Report Date</label>
                                    <input
                                        value={extractedData.reportDate}
                                        onChange={(e) => handleInputChange('reportDate', e.target.value)}
                                        className="w-full bg-slate-50 border border-slate-200 rounded-lg px-3 py-2 font-medium focus:outline-none focus:border-green-500"
                                    />
                                </div>
                                <div>
                                    <label className="block text-xs text-slate-500 uppercase font-bold mb-1">Type</label>
                                    <input
                                        value={extractedData.testType}
                                        onChange={(e) => handleInputChange('testType', e.target.value)}
                                        className="w-full bg-slate-50 border border-slate-200 rounded-lg px-3 py-2 font-medium focus:outline-none focus:border-green-500"
                                    />
                                </div>
                            </div>

                            <div className="bg-slate-50 rounded-xl p-4 border border-slate-100">
                                <h4 className="font-bold text-slate-700 mb-3">Extracted Values</h4>
                                {extractedData.values.map((val, idx) => (
                                    <div key={idx} className="flex items-center gap-3 mb-2 last:mb-0">
                                        <input
                                            value={val.test}
                                            onChange={(e) => handleValueChange(idx, 'test', e.target.value)}
                                            className="flex-1 font-medium text-slate-800 bg-transparent border-b border-transparent focus:border-green-500 focus:outline-none"
                                        />
                                        <input
                                            value={val.value}
                                            onChange={(e) => handleValueChange(idx, 'value', e.target.value)}
                                            className="w-20 font-bold text-slate-900 bg-white px-2 py-1 rounded border border-slate-200 focus:border-green-500 focus:outline-none text-center"
                                        />
                                        <input
                                            value={val.unit}
                                            onChange={(e) => handleValueChange(idx, 'unit', e.target.value)}
                                            className="w-16 text-slate-500 text-sm bg-transparent border-b border-transparent focus:border-green-500 focus:outline-none"
                                        />
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
                )
            }
        </div >
    );
}
