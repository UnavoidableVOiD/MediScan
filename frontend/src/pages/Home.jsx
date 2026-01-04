import React from 'react';
import { Button } from '@/components/ui/button';
import { Link } from 'react-router-dom';

const Home = () => {
    return (
        <div className="flex flex-col items-center justify-center min-h-screen bg-background text-foreground space-y-8">
            <h1 className="text-4xl font-bold text-primary">MediScan System</h1>
            <p className="text-xl text-muted-foreground">AI-Powered Medical Report Analyzer</p>
            <div className="flex gap-4">
                <Button asChild size="lg">
                    <Link to="/patient/login">Patient Portal</Link>
                </Button>
                <Button asChild variant="outline" size="lg">
                    <Link to="/doctor/login">Doctor Portal</Link>
                </Button>
                <Button asChild variant="ghost" size="lg">
                    <Link to="/admin/login">Admin Access</Link>
                </Button>
            </div>
        </div>
    );
};

export default Home;
