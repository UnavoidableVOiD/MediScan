import React from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext.jsx';

export default function Navbar() {
  const { user, logout } = useAuth();
  const navigate = useNavigate();

  const getDashboardLink = () => {
    if (!user) return '/';
    if (user.role === 'admin') return '/admin-dashboard';
    if (user.role === 'doctor') return '/doctor-dashboard';
    return '/dashboard';
  };

  return (
    <nav className="bg-white border-b border-slate-200 shadow-sm z-50 sticky top-0">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between h-20">
          <div className="flex items-center cursor-pointer" onClick={() => navigate('/')}>
            <div className="shrink-0 flex items-center">
              <img src="/logo.jpg" alt="MediScan Logo" className="h-20 w-auto" />
            </div>
          </div>
          <div className="flex items-center gap-4">
            {!user ? (
              <Link
                to="/login"
                className="px-6 py-2 text-sm font-medium bg-green-600 text-white rounded-full hover:bg-green-700 transition shadow-sm"
              >
                Login / Register
              </Link>
            ) : (
              <div className="flex items-center gap-4">
                {user.role === 'patient' && user.plan === 'premium' && (
                  <span className="bg-slate-100 text-yellow-600 text-xs font-bold px-3 py-1.5 rounded-full flex items-center gap-1 border border-yellow-200">
                    <i className="fa-solid fa-crown"></i> Pro Member
                  </span>
                )}

                <Link
                  to={getDashboardLink()}
                  className="px-3 py-2 text-sm font-medium text-slate-500 flex items-center gap-2 hover:text-green-600"
                >
                  {user.profileImage ? (
                    <img
                      src={user.profileImage}
                      alt="Profile"
                      className="w-8 h-8 rounded-full object-cover border border-slate-200"
                    />
                  ) : (
                    <i
                      className={`fa-solid ${user.role === 'admin'
                        ? 'fa-shield-halved'
                        : user.role === 'doctor'
                          ? 'fa-user-doctor'
                          : 'fa-circle-user'
                        } text-xl`}
                    ></i>
                  )}
                </Link>

                <Link
                  to="/profile"
                  className="px-3 py-2 text-sm font-medium text-slate-500 flex items-center gap-2 hover:text-blue-600"
                >
                  <i className="fa-solid fa-gear"></i>
                </Link>

                <button
                  onClick={() => {
                    logout();
                    navigate('/');
                  }}
                  className="px-4 py-2 text-sm font-medium text-red-500 hover:bg-red-50 rounded-lg transition"
                >
                  Logout
                </button>
              </div>
            )}
          </div>
        </div>
      </div>
    </nav>
  );
}