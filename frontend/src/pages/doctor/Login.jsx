import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuthStore } from '@/store/authStore';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '@/components/ui/card';
import { Loader2, Stethoscope } from 'lucide-react';

const DoctorLogin = () => {
    const navigate = useNavigate();
    const { mockLogin } = useAuthStore();
    const [loading, setLoading] = useState(false);
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');

    const handleLogin = async (e) => {
        e.preventDefault();
        setLoading(true);
        await mockLogin('doctor');
        setLoading(false);
        navigate('/doctor');
    };

    return (
        <div className="flex items-center justify-center min-h-[calc(100vh-64px)] bg-muted/20">
            <Card className="w-full max-w-md border-t-4 border-t-primary">
                <CardHeader>
                    <div className="flex items-center gap-2 mb-2">
                        <div className="p-2 bg-primary/10 rounded-full text-primary">
                            <Stethoscope className="w-6 h-6" />
                        </div>
                        <span className="font-bold text-lg">MediScan Pro</span>
                    </div>
                    <CardTitle className="text-2xl">Doctor Portal</CardTitle>
                    <CardDescription>Secure access for medical professionals</CardDescription>
                </CardHeader>
                <CardContent>
                    <form onSubmit={handleLogin} className="space-y-4">
                        <div className="space-y-2">
                            <Label htmlFor="email">Medical ID / Email</Label>
                            <Input
                                id="email"
                                type="email"
                                placeholder="dr.smith@hospital.com"
                                value={email}
                                onChange={(e) => setEmail(e.target.value)}
                                required
                            />
                        </div>
                        <div className="space-y-2">
                            <Label htmlFor="password">Password</Label>
                            <Input
                                id="password"
                                type="password"
                                value={password}
                                onChange={(e) => setPassword(e.target.value)}
                                required
                            />
                        </div>
                        <Button type="submit" className="w-full" disabled={loading}>
                            {loading && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
                            Access Dashboard
                        </Button>
                    </form>
                </CardContent>
                <CardFooter className="flex justify-center border-t pt-4">
                    <p className="text-xs text-muted-foreground">
                        Protected Information System. Unauthorized access is prohibited.
                    </p>
                </CardFooter>
            </Card>
        </div>
    );
};

export default DoctorLogin;
