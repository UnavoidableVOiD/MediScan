import React, { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { UploadCloud, FileText, CheckCircle, Loader2, X } from 'lucide-react';
import { useNavigate } from 'react-router-dom';

const UploadReport = () => {
    const navigate = useNavigate();
    const [dragActive, setDragActive] = useState(false);
    const [file, setFile] = useState(null);
    const [analyzing, setAnalyzing] = useState(false);
    const [uploaded, setUploaded] = useState(false);

    const handleDrag = (e) => {
        e.preventDefault();
        e.stopPropagation();
        if (e.type === "dragenter" || e.type === "dragover") {
            setDragActive(true);
        } else if (e.type === "dragleave") {
            setDragActive(false);
        }
    };

    const handleDrop = (e) => {
        e.preventDefault();
        e.stopPropagation();
        setDragActive(false);
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            setFile(e.dataTransfer.files[0]);
        }
    };

    const handleChange = (e) => {
        e.preventDefault();
        if (e.target.files && e.target.files[0]) {
            setFile(e.target.files[0]);
        }
    };

    const handleAnalyze = () => {
        if (!file) return;
        setAnalyzing(true);
        // Simulate upload and analysis
        setTimeout(() => {
            setAnalyzing(false);
            setUploaded(true);
            // In a real app, this would navigate to the new report ID
            setTimeout(() => navigate('/patient/report/R004'), 1500);
        }, 2500);
    };

    const removeFile = () => {
        setFile(null);
        setUploaded(false);
    };

    return (
        <div className="max-w-3xl mx-auto space-y-8">
            <div>
                <h1 className="text-3xl font-bold tracking-tight text-primary">Upload Medical Report</h1>
                <p className="text-muted-foreground">Upload your lab reports, X-rays, or prescriptions for AI analysis.</p>
            </div>

            <Card>
                <CardContent className="pt-6">
                    {!file && (
                        <div
                            className={`border-2 border-dashed rounded-lg p-12 text-center transition-colors ${dragActive ? 'border-primary bg-primary/5' : 'border-muted-foreground/25 hover:bg-muted/50'
                                }`}
                            onDragEnter={handleDrag}
                            onDragLeave={handleDrag}
                            onDragOver={handleDrag}
                            onDrop={handleDrop}
                        >
                            <div className="flex flex-col items-center justify-center space-y-4">
                                <div className="p-4 bg-primary/10 rounded-full">
                                    <UploadCloud className="w-8 h-8 text-primary" />
                                </div>
                                <div className="space-y-1">
                                    <p className="font-medium">Click to upload or drag and drop</p>
                                    <p className="text-xs text-muted-foreground">PDF, PNG, JPG up to 10MB</p>
                                </div>
                                <input
                                    type="file"
                                    className="hidden"
                                    id="file-upload"
                                    onChange={handleChange}
                                    accept=".pdf,.png,.jpg,.jpeg"
                                />
                                <Button variant="outline" onClick={() => document.getElementById('file-upload').click()}>
                                    Select File
                                </Button>
                            </div>
                        </div>
                    )}

                    {file && !uploaded && !analyzing && (
                        <div className="space-y-6">
                            <div className="flex items-center justify-between p-4 border rounded-lg bg-muted/20">
                                <div className="flex items-center gap-3">
                                    <FileText className="w-8 h-8 text-primary" />
                                    <div>
                                        <p className="font-medium text-sm">{file.name}</p>
                                        <p className="text-xs text-muted-foreground">{(file.size / 1024 / 1024).toFixed(2)} MB</p>
                                    </div>
                                </div>
                                <Button variant="ghost" size="icon" onClick={removeFile}>
                                    <X className="w-4 h-4" />
                                </Button>
                            </div>

                            <div className="flex justify-end gap-3">
                                <Button variant="outline" onClick={removeFile}>Cancel</Button>
                                <Button onClick={handleAnalyze}>Analyze Report</Button>
                            </div>
                        </div>
                    )}

                    {analyzing && (
                        <div className="py-12 flex flex-col items-center justify-center space-y-4 text-center">
                            <Loader2 className="w-10 h-10 text-primary animate-spin" />
                            <div>
                                <h3 className="text-lg font-medium">Analyzing Report...</h3>
                                <p className="text-sm text-muted-foreground">MediBot AI is extracting values and checking for anomalies.</p>
                            </div>
                        </div>
                    )}

                    {uploaded && (
                        <div className="py-12 flex flex-col items-center justify-center space-y-4 text-center animate-in fade-in zoom-in duration-300">
                            <div className="w-12 h-12 bg-green-100 text-green-600 rounded-full flex items-center justify-center">
                                <CheckCircle className="w-6 h-6" />
                            </div>
                            <div>
                                <h3 className="text-lg font-medium">Analysis Complete!</h3>
                                <p className="text-sm text-muted-foreground">Redirecting to your results...</p>
                            </div>
                        </div>
                    )}
                </CardContent>
            </Card>

            {!file && (
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                    <Card>
                        <CardHeader>
                            <FileText className="w-5 h-5 text-primary mb-2" />
                            <CardTitle className="text-base">Lab Reports</CardTitle>
                            <CardDescription>Blood tests, Urine analysis, etc.</CardDescription>
                        </CardHeader>
                    </Card>
                    <Card>
                        <CardHeader>
                            <FileText className="w-5 h-5 text-primary mb-2" />
                            <CardTitle className="text-base">Radiology</CardTitle>
                            <CardDescription>X-Rays, MRI, CT Scans</CardDescription>
                        </CardHeader>
                    </Card>
                    <Card>
                        <CardHeader>
                            <FileText className="w-5 h-5 text-primary mb-2" />
                            <CardTitle className="text-base">Prescriptions</CardTitle>
                            <CardDescription>Handwritten or printed notes</CardDescription>
                        </CardHeader>
                    </Card>
                </div>
            )}
        </div>
    );
};

export default UploadReport;
