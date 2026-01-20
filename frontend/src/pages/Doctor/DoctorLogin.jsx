import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { isIOS } from '../../utils/platform';

function DoctorLogin({ isEmbedded = false }) {
  const [formData, setFormData] = useState({
    email: '',
    password: ''
  });
  const navigate = useNavigate();

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData({ ...formData, [name]: value });
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    console.log('Doctor Login:', formData);
    // In a real app: await login(formData);
    navigate('/doctor/dashboard', { state: { welcomeMessage: 'Welcome Back, Dr. Sharma!' } });
  };

  const formContent = (
    <form onSubmit={handleSubmit} className="space-y-4">
      <div>
        <label className="block text-sm font-semibold text-slate-700 mb-2">Email</label>
        <input
          type="email"
          name="email"
          value={formData.email}
          onChange={handleChange}
          placeholder="Enter your email"
          className="w-full px-4 py-2 border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-green-600"
          required
        />
      </div>

      <div>
        <label className="block text-sm font-semibold text-slate-700 mb-2">Password</label>
        <input
          type="password"
          name="password"
          value={formData.password}
          onChange={handleChange}
          placeholder="Enter your password"
          className="w-full px-4 py-2 border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-green-600"
          required
        />
      </div>

      <button
        type="submit"
        className="w-full bg-green-600 text-white font-bold py-2 rounded-lg hover:bg-green-700 transition"
      >
        Login
      </button>
    </form>
  );

  const handleSocialLogin = (provider) => {
    console.log(`Doctor ${provider} login clicked`);
  };

  if (isEmbedded) {
    return (
      <div className="space-y-6">
        {formContent}

        <div className="relative">
          <div className="absolute inset-0 flex items-center">
            <div className="w-full border-t border-slate-200"></div>
          </div>
          <div className="relative flex justify-center text-sm">
            <span className="px-2 bg-white text-slate-500">Or continue with</span>
          </div>
        </div>

        <div className="grid grid-cols-2 gap-3">
          {!isIOS() && (
            <button
              type="button"
              onClick={() => handleSocialLogin('Google')}
              className="flex items-center justify-center py-2.5 border border-slate-200 rounded-lg hover:bg-slate-50 transition"
              title="Sign in with Google"
            >
              <svg className="w-5 h-5" viewBox="0 0 24 24">
                <path fill="#EA4335" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z" />
                <path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z" />
                <path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z" />
                <path fill="#4285F4" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z" />
              </svg>
            </button>
          )}

          {isIOS() && (
            <button
              type="button"
              onClick={() => handleSocialLogin('Apple')}
              className="flex items-center justify-center py-2.5 border border-slate-200 rounded-lg hover:bg-slate-50 transition"
              title="Sign in with Apple"
            >
              <svg className="w-5 h-5 text-slate-900" fill="currentColor" viewBox="0 0 24 24">
                <path d="M17.05 20.28c-.98.95-2.05 2.34-3.41 2.34-1.2 0-2.46-.9-3.69-2.34-.39-.45-.68-.85-.88-1.18-.74-1.22-1.28-2.91-1.28-4.78 0-3.32 2.34-5.24 4.54-5.24 1.13 0 2.22.56 2.83.56.55 0 1.93-.65 3.09-.65.25 0 .58.02.94.07 1.48.23 2.76.99 3.56 2.06-.09.06-2.02 1.13-2.02 4.3 0 2.94 2.13 4.29 2.18 4.32-.04.14-.3.94-1.19 2.22-.59.85-1.45 2.15-2.21 2.15l-2.46.17zm-4.32-15.5c.61-.75.98-1.74.98-2.78 0-.15 0-.29-.02-.44-1.12.06-2.3.62-2.97 1.41-.53.63-.9 1.55-.9 2.5 0 .15.01.3.02.41 1.02-.07 2.26-.59 2.89-1.1z" />
              </svg>
            </button>
          )}

          <button
            type="button"
            onClick={() => handleSocialLogin('Phone')}
            className="flex items-center justify-center py-2.5 border border-slate-200 rounded-lg hover:bg-slate-50 transition"
            title="Sign in with Phone"
          >
            <svg className="w-5 h-5 text-green-600" fill="currentColor" viewBox="0 0 24 24">
              <path d="M20.01 15.38c-1.23 0-2.42-.2-3.53-.56a.977.977 0 00-1.01.24l-1.57 1.97c-2.83-1.35-5.48-3.9-6.89-6.83l1.95-1.66c.27-.28.35-.67.24-1.02-.37-1.11-.56-2.3-.56-3.53 0-.54-.45-.99-.99-.99H4.19C3.65 3 3 3.24 3 3.99 3 13.28 10.73 21 20.01 21c.71 0 .99-.63.99-1.18v-3.45c0-.54-.45-.99-.99-.99z" />
            </svg>
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="flex-1 min-h-screen flex items-center justify-center bg-slate-50 px-4">
      <div className="w-full max-w-md">
        <div className="text-center mb-8">
          <img src="/logo.jpg" alt="MediScan Logo" className="h-16 w-auto mx-auto mb-4 mix-blend-multiply" />
          <h1 className="text-3xl font-bold text-slate-900 mb-2">Doctor Login</h1>
          <p className="text-slate-600">Access your professional dashboard</p>
        </div>

        {formContent}

        <div className="relative mt-6">
          <div className="absolute inset-0 flex items-center">
            <div className="w-full border-t border-slate-200"></div>
          </div>
          <div className="relative flex justify-center text-sm">
            <span className="px-2 bg-white text-slate-500">Or continue with</span>
          </div>
        </div>

        <div className="grid grid-cols-2 gap-3 mt-6">
          {!isIOS() && (
            <button
              type="button"
              onClick={() => handleSocialLogin('Google')}
              className="flex items-center justify-center py-2.5 border border-slate-200 rounded-lg hover:bg-slate-50 transition"
              title="Sign in with Google"
            >
              <i className="fa-brands fa-google text-red-500 text-xl"></i>
            </button>
          )}

          {isIOS() && (
            <button
              type="button"
              onClick={() => handleSocialLogin('Apple')}
              className="flex items-center justify-center py-2.5 border border-slate-200 rounded-lg hover:bg-slate-50 transition"
              title="Sign in with Apple"
            >
              <i className="fa-brands fa-apple text-slate-900 text-xl"></i>
            </button>
          )}


        </div>

        <div className="mt-6 text-center">
          <p className="text-slate-600">
            Don't have an account?{' '}
            <Link to="/signup?role=doctor" className="text-green-600 font-bold hover:underline">
              Sign up
            </Link>
          </p>
        </div>
      </div>
    </div>
  );
}

export default DoctorLogin;
