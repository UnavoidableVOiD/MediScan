import React from 'react';
import { Link } from 'react-router-dom';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '@/components/ui/card';

const PatientSignup = () => {
    return (
        <div className="flex items-center justify-center min-h-[calc(100vh-64px)] bg-muted/20">
            <Card className="w-full max-w-md">
                <CardHeader>
                    <CardTitle className="text-2xl text-primary">Create Account</CardTitle>
                    <CardDescription>Join MediScan to manage your health records</CardDescription>
                </CardHeader>
                <CardContent>
                    <form className="space-y-4">
                        <div className="grid grid-cols-2 gap-4">
                            <div className="space-y-2">
                                <Label htmlFor="fname">First Name</Label>
                                <Input id="fname" placeholder="John" required />
                            </div>
                            <div className="space-y-2">
                                <Label htmlFor="lname">Last Name</Label>
                                <Input id="lname" placeholder="Doe" required />
                            </div>
                        </div>
                        <div className="space-y-2">
                            <Label htmlFor="email">Email</Label>
                            <Input id="email" type="email" placeholder="yours@example.com" required />
                        </div>
                        <div className="space-y-2">
                            <Label htmlFor="password">Password</Label>
                            <Input id="password" type="password" required />
                        </div>
                        <Button className="w-full">Create Account</Button>
                    </form>
                </CardContent>
                <CardFooter className="flex justify-center">
                    <p className="text-sm text-muted-foreground">
                        Already have an account?{' '}
                        <Link to="/patient/login" className="text-primary hover:underline">
                            Login
                        </Link>
                    </p>
                </CardFooter>
            </Card>
        </div>
    );
};

export default PatientSignup;
