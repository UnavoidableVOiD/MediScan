import React from 'react';
import { Link } from 'react-router-dom';
import { Button } from '@/components/ui/button';
import { HeartPulse } from 'lucide-react';

const Navbar = () => {
    return (
        <nav className="border-b bg-card text-card-foreground">
            <div className="container mx-auto flex items-center justify-between h-16 px-4">
                <Link to="/" className="flex items-center gap-2 font-bold text-xl text-primary">
                    <img src="/logo2.png" alt="MediScan" className="h-25 w-auto object-contain" />


                </Link>

                <div className="flex items-center gap-4">
                    <Link to="/patient/login" className="text-sm font-medium hover:text-primary">Patients</Link>
                    <Link to="/doctor/login" className="text-sm font-medium hover:text-primary">Doctors</Link>
                    <Button asChild size="sm">
                        <Link to="/patient/signup">Get Started</Link>
                    </Button>
                </div>
            </div>
        </nav>
    );
};

export default Navbar;
