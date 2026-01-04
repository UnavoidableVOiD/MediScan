import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuthStore } from '@/store/authStore';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '@/components/ui/card';
import { Loader2, ShieldCheck } from 'lucide-react';

const AdminLogin = () => {
    const navigate = useNavigate();
    const { mockLogin } = useAuthStore();
    const [loading, setLoading] = useState(false);
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');

    const handleLogin = async (e) => {
        e.preventDefault();
        setLoading(true);
        await mockLogin('admin');
        setLoading(false);
        navigate('/admin');
    };

    return (
        <div className="flex items-center justify-center min-h-[calc(100vh-64px)] bg-muted/20">
            <Card className="w-full max-w-md border-t-4 border-t-destructive">
                <CardHeader>
                    <div className="flex items-center gap-2 mb-2">
                        <div className="p-2 bg-destructive/10 rounded-full text-destructive">
                            <ShieldCheck className="w-6 h-6" />
                        </div>
                        <span className="font-bold text-lg">System Administration</span>
                    </div>
                    <CardTitle className="text-2xl">Admin Login</CardTitle>
                    <CardDescription>Enter credentials to manage the MediScan system</CardDescription>
                </CardHeader>
                <CardContent>
                    <form onSubmit={handleLogin} className="space-y-4">
                        <div className="space-y-2">
                            <Label htmlFor="email">Admin Email</Label>
                            <Input
                                id="email"
                                type="email"
                                placeholder="admin@mediscan.com"
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
                        <Button type="submit" variant="destructive" className="w-full" disabled={loading}>
                            {loading && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
                            Login as Admin
                        </Button>
                    </form>
                </CardContent>
                <CardFooter className="flex justify-center border-t pt-4">
                    <p className="text-xs text-muted-foreground">Authorized Personnel Only</p>
                </CardFooter>
            </Card>
        </div>
    );
};

export default AdminLogin;
