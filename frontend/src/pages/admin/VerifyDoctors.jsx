import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Check, X, FileText } from 'lucide-react';

const VerifyDoctors = () => {
    const pendingDoctors = [
        { id: 1, name: 'Dr. Emily Chen', license: 'MD-2023-445', date: '2024-12-24' },
        { id: 2, name: 'Dr. Mark Wilson', license: 'MD-2023-998', date: '2024-12-25' },
    ];

    return (
        <div className="space-y-6">
            <h1 className="text-3xl font-bold tracking-tight text-primary">Verify Medical Licenses</h1>

            <div className="grid gap-4">
                {pendingDoctors.map((doc) => (
                    <Card key={doc.id}>
                        <CardContent className="p-6 flex items-center justify-between">
                            <div>
                                <h3 className="font-bold text-lg">{doc.name}</h3>
                                <div className="flex items-center gap-2 text-sm text-muted-foreground">
                                    <FileText className="w-4 h-4" />
                                    License: {doc.license}
                                </div>
                                <p className="text-xs text-muted-foreground mt-1">Uploaded: {doc.date}</p>
                            </div>
                            <div className="flex gap-2">
                                <Button variant="outline" className="text-destructive border-destructive hover:bg-destructive/10">
                                    <X className="w-4 h-4 mr-2" /> Reject
                                </Button>
                                <Button className="bg-green-600 hover:bg-green-700">
                                    <Check className="w-4 h-4 mr-2" /> Approve
                                </Button>
                            </div>
                        </CardContent>
                    </Card>
                ))}
                {pendingDoctors.length === 0 && <p className="text-muted-foreground">No pending verifications.</p>}
            </div>
        </div>
    );
};

export default VerifyDoctors;
