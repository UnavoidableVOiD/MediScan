import React from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { useDispatch, useSelector } from 'react-redux';
import { logout } from '../store/slices/authSlice';
import { Shield, Activity, Menu, X, LogOut, User } from 'lucide-react';

const Navbar = () => {
    const [isOpen, setIsOpen] = React.useState(false);
    const dispatch = useDispatch();
    const navigate = useNavigate();
    const { token, user } = useSelector((state) => state.auth);

    const handleLogout = () => {
        dispatch(logout());
        setIsOpen(false);
        navigate('/login');
    };

    return (
        <nav className="bg-white border-b border-gray-100 fixed w-full z-50 top-0 left-0">
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                <div className="flex justify-between h-20">
                    <div className="flex items-center">
                        <Link to="/" className="flex items-center gap-2 group">
                            <div className="bg-gradient-to-br from-blue-600 to-emerald-500 p-2 rounded-lg transition-transform group-hover:scale-105">
                                <Shield className="h-6 w-6 text-white" />
                            </div>
                            <span className="text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-700 to-emerald-600">
                                MediScan
                            </span>
                        </Link>
                    </div>

                    {/* Desktop Menu */}
                    <div className="hidden md:flex items-center space-x-8">
                        {!token && <Link to="/" className="text-gray-600 hover:text-blue-600 font-semibold transition-colors">Home</Link>}
                        {token && <Link to="/dashboard" className="text-gray-600 hover:text-blue-600 font-semibold transition-colors">Dashboard</Link>}
                        {!token && <a href="#features" className="text-gray-600 hover:text-blue-600 font-semibold transition-colors">Features</a>}
                        {!token && <Link to="/demo" className="text-gray-600 hover:text-emerald-600 font-semibold transition-colors">Demo</Link>}

                        <div className="flex items-center gap-4 ml-6 pl-6 border-l border-gray-100">
                            {token ? (
                                <div className="flex items-center gap-4">
                                    <Link to="/profile" className="flex items-center gap-2 px-3 py-1.5 bg-gray-50 rounded-lg border border-gray-100 hover:bg-gray-100 transition-colors">
                                        <div className="h-8 w-8 bg-blue-100 rounded-full flex items-center justify-center">
                                            <User className="h-4 w-4 text-blue-600" />
                                        </div>
                                        <span className="text-sm font-bold text-gray-700 capitalize">{user?.first_name || 'User'}</span>
                                    </Link>
                                    <button
                                        onClick={handleLogout}
                                        className="flex items-center gap-2 text-gray-600 hover:text-red-500 font-bold px-4 py-2 transition-colors"
                                    >
                                        <LogOut className="h-4 w-4" />
                                        Logout
                                    </button>
                                </div>
                            ) : (
                                <>
                                    <Link to="/login" className="text-gray-700 hover:text-blue-600 font-bold px-4 py-2 transition-colors">
                                        Login
                                    </Link>
                                    <Link to="/signup" className="bg-gradient-to-r from-blue-600 to-emerald-500 text-white px-7 py-2.5 rounded-full font-bold shadow-lg shadow-blue-200 hover:shadow-xl hover:-translate-y-0.5 transition-all">
                                        Get Started
                                    </Link>
                                </>
                            )}
                        </div>
                    </div>

                    {/* Mobile Menu Button */}
                    <div className="md:hidden flex items-center">
                        <button
                            onClick={() => setIsOpen(!isOpen)}
                            className="p-2 rounded-md text-gray-600 hover:bg-gray-50 focus:outline-none"
                        >
                            {isOpen ? <X className="h-6 w-6" /> : <Menu className="h-6 w-6" />}
                        </button>
                    </div>
                </div>
            </div>

            {/* Mobile Menu */}
            {isOpen && (
                <div className="md:hidden bg-white border-b border-gray-100 animate-in fade-in slide-in-from-top-4 duration-200">
                    <div className="px-4 pt-2 pb-6 space-y-2">
                        {!token && <Link to="/" onClick={() => setIsOpen(false)} className="block px-3 py-3 text-gray-700 font-semibold hover:bg-gray-50 rounded-lg">Home</Link>}
                        {token && <Link to="/dashboard" onClick={() => setIsOpen(false)} className="block px-3 py-3 text-gray-700 font-semibold hover:bg-gray-50 rounded-lg">Dashboard</Link>}
                        {!token && <a href="#features" onClick={() => setIsOpen(false)} className="block px-3 py-3 text-gray-700 font-semibold hover:bg-gray-50 rounded-lg">Features</a>}
                        {!token && <Link to="/demo" onClick={() => setIsOpen(false)} className="block px-3 py-3 text-gray-700 font-semibold hover:bg-gray-50 rounded-lg">Demo</Link>}

                        {token ? (
                            <>
                                <div className="px-3 py-3 flex items-center gap-3 border-t border-gray-50 mt-2">
                                    <div className="h-10 w-10 bg-blue-100 rounded-full flex items-center justify-center">
                                        <User className="h-5 w-5 text-blue-600" />
                                    </div>
                                    <div>
                                        <p className="font-bold text-gray-900 capitalize">{user?.first_name} {user?.last_name}</p>
                                        <p className="text-xs text-gray-500">{user?.email}</p>
                                    </div>
                                </div>
                                <Link to="/profile" onClick={() => setIsOpen(false)} className="block px-3 py-3 text-gray-700 font-semibold hover:bg-gray-50 rounded-lg">Profile</Link>
                                <button
                                    onClick={handleLogout}
                                    className="w-full text-left px-3 py-3 text-red-600 font-bold hover:bg-red-50 rounded-lg flex items-center gap-2"
                                >
                                    <LogOut className="h-5 w-5" />
                                    Logout
                                </button>
                            </>
                        ) : (
                            <>
                                <Link to="/login" onClick={() => setIsOpen(false)} className="block px-3 py-3 text-gray-700 font-bold hover:bg-gray-50 rounded-lg">Login</Link>
                                <Link to="/signup" onClick={() => setIsOpen(false)} className="block px-3 py-4 bg-gradient-to-r from-blue-600 to-emerald-500 text-white text-center rounded-xl font-bold">Get Started</Link>
                            </>
                        )}
                    </div>
                </div>
            )}
        </nav>
    );
};

export default Navbar;
