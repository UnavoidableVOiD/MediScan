<<<<<<< Updated upstream
import React, { useState } from 'react';
import { Link, useNavigate, useLocation } from 'react-router-dom';
import { HeartPulse, LogOut, User as UserIcon, Menu, X } from 'lucide-react';
import { useSelector, useDispatch } from 'react-redux';
import { logoutUser } from '../../store/slices/authSlice';
import { motion, AnimatePresence } from 'framer-motion';

const Navbar = () => {
    const { isAuthenticated, user } = useSelector(state => state.auth);
    const dispatch = useDispatch();
    const navigate = useNavigate();
    const location = useLocation();
    const [isOpen, setIsOpen] = useState(false);

    const handleLogout = () => {
        dispatch(logoutUser());
        navigate('/login');
        setIsOpen(false);
    };

    const navLinks = [
        { name: 'Home', path: '/', show: !isAuthenticated },
        { name: 'Dashboard', path: '/dashboard', show: isAuthenticated },
        { name: 'Check Reports', path: '/check-reports', show: isAuthenticated && user?.role?.toLowerCase() === 'patient' },
        { name: 'About', path: '/about', show: true },
        { name: 'Contact', path: '/contact', show: true },
        { name: 'Services', path: '/services', show: true },
    ];

    const toggleMenu = () => setIsOpen(!isOpen);

    const activeLinkClass = "text-medic-dark font-bold";
    const inactiveLinkClass = "hover:text-medic-dark transition-colors";

    return (
        <nav className="sticky top-0 z-50 bg-white/80 backdrop-blur-md border-b border-medic-light/50">
            <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
                <Link to="/" className="flex items-center gap-2 group z-50">
                    <HeartPulse className="w-8 h-8 text-medic-dark transition-transform group-hover:scale-110" />
                    <span className="text-2xl font-bold text-medic-dark tracking-tight">Mediscan</span>
                </Link>

                {/* Desktop Links */}
                <div className="hidden md:flex items-center gap-8 text-gray-600 font-medium">
                    {navLinks.filter(link => link.show).map((link) => (
                        <Link
                            key={link.path}
                            to={link.path}
                            className={location.pathname === link.path ? activeLinkClass : inactiveLinkClass}
                        >
                            {link.name}
                        </Link>
                    ))}
                </div>

                <div className="flex items-center gap-4">
                    {/* Desktop Auth */}
                    <div className="hidden md:flex items-center gap-4">
                        {isAuthenticated ? (
                            <div className="flex items-center gap-4">
                                <Link to="/profile">
                                    <div className="flex items-center gap-2 px-3 py-1.5 bg-medic-light/50 rounded-full border border-medic-dark/10 hover:bg-medic-light transition-colors">
                                        <div className="w-8 h-8 rounded-full bg-medic-dark text-white flex items-center justify-center">
                                            <UserIcon className="w-4 h-4" />
                                        </div>
                                        <span className="text-sm font-bold text-medic-dark">{user?.first_name}</span>
                                    </div>
                                </Link>

                                <button
                                    onClick={handleLogout}
                                    className="p-2 hover:bg-red-50 text-gray-400 hover:text-red-500 rounded-full transition-colors"
                                    title="Logout"
                                >
                                    <LogOut className="w-5 h-5" />
                                </button>
                            </div>
                        ) : (
                            <>
                                <Link to="/login" className="text-medic-dark font-semibold hover:opacity-80 transition-opacity">Login</Link>
                                <Link to="/signup" className="bg-medic-dark text-white px-6 py-2.5 rounded-full font-semibold shadow-md shadow-medic-dark/20 hover:bg-medic-primary transition-all active:scale-95">
                                    Sign Up
                                </Link>
                            </>
                        )}
                    </div>

                    {/* Mobile Menu Toggle */}
                    <button
                        onClick={toggleMenu}
                        className="md:hidden p-2 text-medic-dark hover:bg-medic-light/50 rounded-xl transition-colors z-50"
                    >
                        {isOpen ? <X className="w-7 h-7" /> : <Menu className="w-7 h-7" />}
                    </button>
                </div>
            </div>

            {/* Mobile Menu & Backdrop */}
            <AnimatePresence>
                {isOpen && (
                    <>
                        {/* Blurred Backdrop */}
                        <motion.div
                            initial={{ opacity: 0 }}
                            animate={{ opacity: 1 }}
                            exit={{ opacity: 0 }}
                            onClick={() => setIsOpen(false)}
                            className="fixed inset-0 bg-medic-dark/20 backdrop-blur-sm z-30 md:hidden"
                        />

                        {/* Dropdown Menu */}
                        <motion.div
                            initial={{ opacity: 0, y: -10 }}
                            animate={{ opacity: 1, y: 0 }}
                            exit={{ opacity: 0, y: -10 }}
                            className="absolute top-full left-0 right-0 bg-white z-40 md:hidden border-b border-medic-light/50 shadow-2xl overflow-hidden"
                        >
                            <div className="p-6 flex flex-col gap-4 max-h-[80vh] overflow-y-auto">
                                {navLinks.filter(link => link.show).map((link) => (
                                    <Link
                                        key={link.name}
                                        to={link.path}
                                        onClick={() => setIsOpen(false)}
                                        className={`text-lg font-bold py-3 px-4 rounded-xl transition-all ${location.pathname === link.path
                                                ? 'bg-medic-light/30 text-medic-dark'
                                                : 'text-gray-600 hover:bg-neutral-soft'
                                            }`}
                                    >
                                        {link.name}
                                    </Link>
                                ))}

                                <hr className="border-gray-100 my-2" />

                                {isAuthenticated ? (
                                    <div className="space-y-4 pt-2">
                                        <Link
                                            to="/profile"
                                            onClick={() => setIsOpen(false)}
                                            className="flex items-center gap-4 p-4 bg-medic-light/20 rounded-2xl border border-medic-light/10"
                                        >
                                            <div className="w-12 h-12 rounded-full bg-medic-dark text-white flex items-center justify-center">
                                                <UserIcon className="w-6 h-6" />
                                            </div>
                                            <div>
                                                <p className="font-bold text-medic-dark">{user?.first_name} {user?.last_name}</p>
                                                <p className="text-xs text-gray-500 capitalize">{user?.role}</p>
                                            </div>
                                        </Link>
                                        <button
                                            onClick={handleLogout}
                                            className="w-full py-4 flex items-center justify-center gap-2 text-red-500 font-bold bg-red-50 rounded-2xl"
                                        >
                                            <LogOut className="w-5 h-5" />
                                            Logout Account
                                        </button>
                                    </div>
                                ) : (
                                    <div className="flex flex-col gap-3 pt-2">
                                        <Link
                                            to="/login"
                                            onClick={() => setIsOpen(false)}
                                            className="w-full py-4 text-center font-bold text-medic-dark border-2 border-medic-dark rounded-2xl"
                                        >
                                            Login
                                        </Link>
                                        <Link
                                            to="/signup"
                                            onClick={() => setIsOpen(false)}
                                            className="w-full py-4 text-center font-bold text-white bg-medic-dark rounded-2xl shadow-lg shadow-medic-dark/20"
                                        >
                                            Sign Up Free
                                        </Link>
                                    </div>
                                )}
                            </div>
                        </motion.div>
                    </>
                )}
            </AnimatePresence>
        </nav>
    );
};

export default Navbar;
=======
import React, { useState } from 'react';
import { Link, useNavigate, useLocation } from 'react-router-dom';
import { HeartPulse, LogOut, User as UserIcon, Menu, X } from 'lucide-react';
import { useSelector, useDispatch } from 'react-redux';
import { logoutUser } from '../../store/slices/authSlice';
import { motion, AnimatePresence } from 'framer-motion';

const Navbar = () => {
    const { isAuthenticated, user } = useSelector(state => state.auth);
    const dispatch = useDispatch();
    const navigate = useNavigate();
    const location = useLocation();
    const [isOpen, setIsOpen] = useState(false);

    const handleLogout = () => {
        dispatch(logoutUser());
        navigate('/login');
        setIsOpen(false);
    };

    const isUnverifiedDoctor = user?.role === 'DOCTOR' && user?.doctor_status === 'UNVERIFIED';
    const isPendingDoctor = user?.role === 'DOCTOR' && user?.doctor_status === 'PENDING';
    const isVerifiedDoctor = user?.role === 'DOCTOR' && user?.doctor_status === 'VERIFIED';
    const isRestrictedDoctor = !isVerifiedDoctor && user?.role === 'DOCTOR';

    const navLinks = [
        { name: 'Home', path: '/', show: !isAuthenticated },
        {
            name: user?.role === 'ADMIN' ? 'Admin Portal' : 'Dashboard',
            path: user?.role === 'ADMIN' ? '/admin/dashboard' : (user?.role === 'DOCTOR' ? '/doctor-dashboard' : '/dashboard'),
            show: isAuthenticated,
            disabled: isRestrictedDoctor
        },
        {
            name: 'Appointments',
            path: '/appointments',
            show: isAuthenticated && user?.role === 'DOCTOR',
            disabled: isRestrictedDoctor
        },
        {
            name: 'Patients',
            path: '/patients',
            show: isAuthenticated && user?.role === 'DOCTOR',
            disabled: isRestrictedDoctor
        },
        // {
        //     name: 'Reports',
        //     path: '/doctor-reports',
        //     show: isAuthenticated && user?.role === 'DOCTOR',
        //     disabled: isRestrictedDoctor
        // },
        {
            name: 'Check Reports',
            path: '/check-reports',
            show: isAuthenticated && user?.role === 'PATIENT'
        },
        { name: 'About', path: '/about', show: user?.role !== 'ADMIN' },
        { name: 'Contact', path: '/contact', show: user?.role !== 'ADMIN' },
        { name: 'Services', path: '/services', show: user?.role !== 'ADMIN' },
    ];

    const toggleMenu = () => setIsOpen(!isOpen);

    const activeLinkClass = "text-medic-dark font-bold";
    const inactiveLinkClass = "hover:text-medic-dark transition-colors";

    return (
        <nav className="sticky top-0 z-50 bg-white/80 backdrop-blur-md border-b border-medic-light/50">
            <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
                <Link to="/" className="flex items-center gap-2 group z-50">
                    <HeartPulse className="w-8 h-8 text-medic-dark transition-transform group-hover:scale-110" />
                    <span className="text-2xl font-bold text-medic-dark tracking-tight">Mediscan</span>
                </Link>

                {/* Desktop Links */}
                <div className="hidden md:flex items-center gap-8 text-gray-600 font-medium">
                    {navLinks.filter(link => link.show).map((link) => (
                        link.disabled ? (
                            <div key={link.path} className="relative group cursor-not-allowed">
                                <span className="text-gray-400">
                                    {link.name}
                                </span>
                                <div className="absolute top-full left-1/2 -translate-x-1/2 mt-2 px-3 py-1 bg-gray-800 text-white text-xs rounded-lg opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap pointer-events-none z-[60]">
                                    {isPendingDoctor ? "Access will be available after verification approval" : "Verify your profile first to get access"}
                                    <div className="absolute -top-1 left-1/2 -translate-x-1/2 border-4 border-transparent border-b-gray-800" />
                                </div>
                            </div>
                        ) : (
                            <Link
                                key={link.path}
                                to={link.path}
                                className={location.pathname === link.path ? activeLinkClass : inactiveLinkClass}
                            >
                                {link.name}
                            </Link>
                        )
                    ))}
                </div>

                <div className="flex items-center gap-4">
                    {/* Desktop Auth */}
                    <div className="hidden md:flex items-center gap-4">
                        {isAuthenticated ? (
                            <div className="flex items-center gap-4">
                                <Link to="/profile">
                                    <div className="flex items-center gap-2 px-3 py-1.5 bg-medic-light/50 rounded-full border border-medic-dark/10 hover:bg-medic-light transition-colors">
                                        <div className="w-8 h-8 rounded-full bg-medic-dark text-white flex items-center justify-center">
                                            <UserIcon className="w-4 h-4" />
                                        </div>
                                        <span className="text-sm font-bold text-medic-dark">{user?.first_name}</span>
                                    </div>
                                </Link>

                                <button
                                    onClick={handleLogout}
                                    className="p-2 hover:bg-red-50 text-gray-400 hover:text-red-500 rounded-full transition-colors"
                                    title="Logout"
                                >
                                    <LogOut className="w-5 h-5" />
                                </button>
                            </div>
                        ) : (
                            <>
                                <Link to="/login" className="text-medic-dark font-semibold hover:opacity-80 transition-opacity">Login</Link>
                                <Link to="/signup" className="bg-medic-dark text-white px-6 py-2.5 rounded-full font-semibold shadow-md shadow-medic-dark/20 hover:bg-medic-primary transition-all active:scale-95">
                                    Sign Up
                                </Link>
                            </>
                        )}
                    </div>

                    {/* Mobile Menu Toggle */}
                    <button
                        onClick={toggleMenu}
                        className="md:hidden p-2 text-medic-dark hover:bg-medic-light/50 rounded-xl transition-colors z-50"
                    >
                        {isOpen ? <X className="w-7 h-7" /> : <Menu className="w-7 h-7" />}
                    </button>
                </div>
            </div>

            {/* Mobile Menu & Backdrop */}
            <AnimatePresence>
                {isOpen && (
                    <>
                        {/* Blurred Backdrop */}
                        <motion.div
                            initial={{ opacity: 0 }}
                            animate={{ opacity: 1 }}
                            exit={{ opacity: 0 }}
                            onClick={() => setIsOpen(false)}
                            className="fixed inset-0 bg-medic-dark/20 backdrop-blur-sm z-30 md:hidden"
                        />

                        {/* Dropdown Menu */}
                        <motion.div
                            initial={{ opacity: 0, y: -10 }}
                            animate={{ opacity: 1, y: 0 }}
                            exit={{ opacity: 0, y: -10 }}
                            className="absolute top-full left-0 right-0 bg-white z-40 md:hidden border-b border-medic-light/50 shadow-2xl overflow-hidden"
                        >
                            <div className="p-6 flex flex-col gap-4 max-h-[80vh] overflow-y-auto">
                                {navLinks.filter(link => link.show).map((link) => (
                                    link.disabled ? (
                                        <div
                                            key={link.name}
                                            className="text-lg font-bold py-3 px-4 rounded-xl text-gray-400 bg-gray-50/50 flex flex-col gap-1"
                                        >
                                            <span>{link.name}</span>
                                            <span className="text-[10px] text-orange-500 uppercase tracking-widest leading-none">
                                                {isPendingDoctor ? "Approval Pending" : "Verification Required"}
                                            </span>
                                        </div>
                                    ) : (
                                        <Link
                                            key={link.name}
                                            to={link.path}
                                            onClick={() => setIsOpen(false)}
                                            className={`text-lg font-bold py-3 px-4 rounded-xl transition-all ${location.pathname === link.path
                                                ? 'bg-medic-light/30 text-medic-dark'
                                                : 'text-gray-600 hover:bg-neutral-soft'
                                                }`}
                                        >
                                            {link.name}
                                        </Link>
                                    )
                                ))}

                                <hr className="border-gray-100 my-2" />

                                {isAuthenticated ? (
                                    <div className="space-y-4 pt-2">
                                        <Link
                                            to="/profile"
                                            onClick={() => setIsOpen(false)}
                                            className="flex items-center gap-4 p-4 bg-medic-light/20 rounded-2xl border border-medic-light/10"
                                        >
                                            <div className="w-12 h-12 rounded-full bg-medic-dark text-white flex items-center justify-center">
                                                <UserIcon className="w-6 h-6" />
                                            </div>
                                            <div>
                                                <p className="font-bold text-medic-dark">{user?.first_name} {user?.last_name}</p>
                                                <p className="text-xs text-gray-500 capitalize">{user?.role}</p>
                                            </div>
                                        </Link>
                                        <button
                                            onClick={handleLogout}
                                            className="w-full py-4 flex items-center justify-center gap-2 text-red-500 font-bold bg-red-50 rounded-2xl"
                                        >
                                            <LogOut className="w-5 h-5" />
                                            Logout Account
                                        </button>
                                    </div>
                                ) : (
                                    <div className="flex flex-col gap-3 pt-2">
                                        <Link
                                            to="/login"
                                            onClick={() => setIsOpen(false)}
                                            className="w-full py-4 text-center font-bold text-medic-dark border-2 border-medic-dark rounded-2xl"
                                        >
                                            Login
                                        </Link>
                                        <Link
                                            to="/signup"
                                            onClick={() => setIsOpen(false)}
                                            className="w-full py-4 text-center font-bold text-white bg-medic-dark rounded-2xl shadow-lg shadow-medic-dark/20"
                                        >
                                            Sign Up Free
                                        </Link>
                                    </div>
                                )}
                            </div>
                        </motion.div>
                    </>
                )}
            </AnimatePresence>
        </nav>
    );
};

export default Navbar;
>>>>>>> Stashed changes
