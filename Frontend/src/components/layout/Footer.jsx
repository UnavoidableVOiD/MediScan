import React from 'react';
import { Link } from 'react-router-dom';
import { HeartPulse } from 'lucide-react';

const Footer = () => {
    return (
        <footer className="bg-white border-t border-medic-light/50 py-12 text-gray-600 mt-20">
            <div className="max-w-7xl mx-auto px-6 flex flex-col md:flex-row justify-between items-center gap-8">
                <div className="flex flex-col items-center md:items-start gap-4 text-center md:text-left">
                    <div className="flex items-center gap-2">
                        <HeartPulse className="w-6 h-6 text-medic-dark" />
                        <span className="text-xl font-bold text-medic-dark">Mediscan</span>
                    </div>
                    <p className="max-w-xs text-sm">
                        Empowering patients with AI-driven medical report analysis for better understanding and peace of mind.
                    </p>
                </div>

                <div className="flex gap-12 text-sm font-medium">
                    <div className="flex flex-col gap-3">
                        <span className="text-gray-900 font-bold uppercase tracking-wider text-xs">Product</span>
                        <Link to="/about" className="hover:text-medic-dark transition-colors">About</Link>
                        <Link to="/services" className="hover:text-medic-dark transition-colors">Services</Link>
                        <Link to="/demo" className="hover:text-medic-dark transition-colors">Try Demo</Link>
                    </div>
                    <div className="flex flex-col gap-3">
                        <span className="text-gray-900 font-bold uppercase tracking-wider text-xs">Legal</span>
                        <Link to="/privacy" className="hover:text-medic-dark transition-colors">Privacy</Link>
                        <Link to="/terms" className="hover:text-medic-dark transition-colors">Terms</Link>
                    </div>
                    <div className="flex flex-col gap-3">
                        <span className="text-gray-900 font-bold uppercase tracking-wider text-xs">Support</span>
                        <Link to="/contact" className="hover:text-medic-dark transition-colors">Contact</Link>
                        <Link to="/faq" className="hover:text-medic-dark transition-colors">FAQ</Link>
                    </div>
                </div>
            </div>
            <div className="max-w-7xl mx-auto px-6 mt-12 pt-8 border-t border-medic-light/30 text-center text-xs opacity-60">
                &copy; {new Date().getFullYear()} Mediscan. All rights reserved.
            </div>
        </footer>
    );
};

export default Footer;
