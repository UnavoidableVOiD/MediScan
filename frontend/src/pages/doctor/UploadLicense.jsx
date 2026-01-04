import React from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';

const UploadLicense = () => {
    return (
        <Card className="max-w-2xl mx-auto mt-10">
            <CardHeader>
                <CardTitle>Upload Medical License</CardTitle>
            </CardHeader>
            <CardContent>
                <div className="border-2 border-dashed p-10 text-center rounded-lg">
                    <p className="text-muted-foreground mb-4">Upload your medical license for verification.</p>
                    <Button>Select File</Button>
                </div>
            </CardContent>
        </Card>
    );
};
export default UploadLicense;
