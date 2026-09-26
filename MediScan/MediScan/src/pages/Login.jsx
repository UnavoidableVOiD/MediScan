import React, { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext.jsx';
export default function Login() {
  const [role, setRole] = useState('patient');
  const [loading, setLoading] = useState(false);
  const { login } = useAuth();
  const navigate = useNavigate();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState('');

  const handleLogin = (e) => {
    e.preventDefault();
    setError('');

    if (!email.includes('@') || !email.includes('.')) {
      setError('Please enter a valid email address.');
      return;
    }
    const passwordRegex = /^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$/;

    if (!passwordRegex.test(password)) {
      setError('Password must be at least 8 chars, include uppercase, lowercase, number & symbol.');
      return;
    }

    setLoading(true);
    // Simulate API login
    setTimeout(() => {
      login(role);
      if (role === 'admin') navigate('/admin-dashboard');
      else if (role === 'doctor') navigate('/doctor-dashboard');
      else navigate('/dashboard');
    }, 1200);
  };

  const handleGoogleLogin = () => {
    setLoading(true);
    setTimeout(() => {
      login('patient');
      navigate('/dashboard');
    }, 1500);
  };

  const getRoleIcon = () => {
    if (role === 'patient') return 'fa-user';
    if (role === 'doctor') return 'fa-user-doctor';
    return 'fa-shield-halved';
  };
  return (
    <div className="flex-1 flex items-center justify-center p-4 min-h-[calc(100vh-64px)]">
      <div className="max-w-md w-full space-y-8 bg-white p-10 rounded-2xl shadow-xl border border-slate-100 fade-in">
        <div className="text-center">
          <div className="mx-auto flex justify-center mb-6">
            <img src="/logo.jpg" alt="MediScan Logo" className="h-28 w-auto" />
          </div>
          <h2 className="text-3xl font-extrabold text-slate-900">Welcome Back</h2>
          <p className="mt-2 text-sm text-slate-600">Sign in to access your {role} panel</p>
        </div>
        <div className="flex p-1 bg-slate-100 rounded-xl">
          <button
            onClick={() => setRole('patient')}
            className={`flex-1 py-2 text-sm font-medium rounded-lg transition-all ${role === 'patient' ? 'bg-white text-slate-900 shadow-sm' : 'text-slate-500 hover:text-slate-900'
              }`}
          >
            Patient
          </button>
          <button
            onClick={() => setRole('doctor')}
            className={`flex-1 py-2 text-sm font-medium rounded-lg transition-all ${role === 'doctor' ? 'bg-white text-slate-900 shadow-sm' : 'text-slate-500 hover:text-slate-900'
              }`}
          >
            Doctor
          </button>
          <button
            onClick={() => setRole('admin')}
            className={`flex-1 py-2 text-sm font-medium rounded-lg transition-all ${role === 'admin' ? 'bg-white text-slate-900 shadow-sm' : 'text-slate-500 hover:text-slate-900'
              }`}
          >
            Admin
          </button>
        </div>
        {error && (
          <div className="bg-red-50 text-red-600 p-3 rounded-lg text-sm mb-6 flex items-center gap-2">
            <i className="fa-solid fa-circle-exclamation"></i>
            {error}
          </div>
        )}

        <form onSubmit={handleLogin} className="mt-8 space-y-4">
          <div className="space-y-4">
            <input
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
              className="w-full px-4 py-3 rounded-lg border border-slate-300 focus:ring-2 focus:ring-green-500 focus:outline-none"
              placeholder="Email address"
            />
            <div className="relative">
              <input
                type={showPassword ? "text" : "password"}
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
                className="w-full px-4 py-3 rounded-lg border border-slate-300 focus:ring-2 focus:ring-green-500 focus:outline-none pr-10"
                placeholder="Password"
              />
              <button
                type="button"
                onClick={() => setShowPassword(!showPassword)}
                className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-400 hover:text-slate-600"
              >
                <i className={`fa-solid ${showPassword ? 'fa-eye-slash' : 'fa-eye'}`}></i>
              </button>
            </div>
          </div>

          <div className="flex items-center justify-between text-sm">
            <label className="flex items-center text-slate-500">
              <input type="checkbox" className="mr-2 rounded text-green-600 focus:ring-green-500" />
              Remember me
            </label>
            <a href="#" className="font-medium text-green-600 hover:text-green-500">
              Forgot password?
            </a>
          </div>

          <button
            type="submit"
            disabled={loading}
            className="w-full py-3 px-4 text-sm font-bold rounded-xl text-white bg-green-600 hover:bg-green-700 focus:outline-none transition shadow-lg"
          >
            {loading ? <span className="loader w-5 h-5 border-2 border-white/30 border-t-white inline-block"></span> : 'Sign in'}
          </button>
        </form>

        <div className="mt-6">
          <div className="relative">
            <div className="absolute inset-0 flex items-center">
              <div className="w-full border-t border-slate-200"></div>
            </div>
            <div className="relative flex justify-center text-sm">
              <span className="px-2 bg-white text-slate-500">Or continue with</span>
            </div>
          </div>

          <button
            onClick={handleGoogleLogin}
            disabled={loading}
            className="mt-6 w-full flex items-center justify-center gap-3 px-4 py-3 border border-slate-300 shadow-sm text-sm font-medium rounded-xl text-slate-700 bg-white hover:bg-slate-50 transition"
          >
            <img src="https://www.google.com/favicon.ico" alt="Google" className="h-5 w-5" />
            Sign in with Google
          </button>
        </div>

        <p className="text-center text-sm text-slate-500 mt-8">
          Don't have an account?{' '}
          <Link to="/signup" className="text-green-600 font-bold hover:underline">
            Sign up
          </Link>
        </p>
      </div>
    </div>
  );
}
