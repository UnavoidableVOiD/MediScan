import React, { useState, useEffect } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { useAuth } from '../../contexts/AuthContext';
import { isIOS } from '../../utils/platform';

function PatientSignup({ isEmbedded = false }) {
  const [formData, setFormData] = useState({
    firstName: '',
    lastName: '',
    email: '',
    password: '',
    confirmPassword: ''
  });
  const navigate = useNavigate();

  useEffect(() => {
    // Load Google Identity Services script
    const script = document.createElement('script');
    script.src = 'https://accounts.google.com/gsi/client';
    script.async = true;
    script.defer = true;
    document.body.appendChild(script);

    return () => {
      // Cleanup script on unmount
      const existingScript = document.querySelector('script[src="https://accounts.google.com/gsi/client"]');
      if (existingScript) {
        document.body.removeChild(existingScript);
      }
    };
  }, []);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData({ ...formData, [name]: value });
  };

  const handleSubmit = (e) => {
    e.preventDefault();

    if (formData.password !== formData.confirmPassword) {
      alert('Passwords do not match');
      return;
    }

    console.log('Patient Signup:', formData);
    // Add signup logic here - after successful signup:
    // Redirect to login page for the user to log in
    navigate('/patient/login');
  };

  const handleGoogleSignup = async () => {
    try {
      // Initialize Google Identity Services
      if (window.google?.accounts) {
        window.google.accounts.id.initialize({
          client_id: process.env.REACT_APP_GOOGLE_CLIENT_ID || '',
          callback: handleGoogleResponse,
        });

        window.google.accounts.id.prompt((notification) => {
          if (notification.isNotDisplayed() || notification.isSkippedMoment()) {
            // Use popup flow if prompt is not displayed
            window.google.accounts.oauth2.initTokenClient({
              client_id: process.env.REACT_APP_GOOGLE_CLIENT_ID || '',
              scope: 'email profile',
              callback: (response) => {
                if (response.access_token) {
                  fetchGoogleUserInfo(response.access_token);
                }
              },
            }).requestAccessToken();
          }
        });
      } else {
        // Fallback: Use OAuth popup if GIS is not fully loaded
        const clientId = process.env.REACT_APP_GOOGLE_CLIENT_ID || '';
        if (clientId) {
          const authUrl = `https://accounts.google.com/o/oauth2/v2/auth?client_id=${clientId}&redirect_uri=${window.location.origin}&response_type=code&scope=email profile&access_type=offline`;
          window.location.href = authUrl;
        } else {
          // Simulate Google signup flow (for development without API key)
          console.log('Google signup initiated - please configure REACT_APP_GOOGLE_CLIENT_ID');
          alert('Google Sign-Up: Please configure your Google OAuth Client ID. For now, you can sign up with email and password.');
        }
      }
    } catch (error) {
      console.error('Google signup error:', error);
      alert('Google signup is not available. Please use email and password to sign up.');
    }
  };

  const handleGoogleResponse = async (response) => {
    try {
      // Decode the JWT token to get user info
      const base64Url = response.credential.split('.')[1];
      const base64 = base64Url.replace(/-/g, '+').replace(/_/g, '/');
      const jsonPayload = decodeURIComponent(atob(base64).split('').map(c => {
        return '%' + ('00' + c.charCodeAt(0).toString(16)).slice(-2);
      }).join(''));
      
      const userData = JSON.parse(jsonPayload);
      
      console.log('Google Signup Success:', userData);
      
      // Extract name from Google response
      const fullName = userData.name || '';
      const nameParts = fullName.split(' ');
      const firstName = nameParts[0] || '';
      const lastName = nameParts.slice(1).join(' ') || '';

      // After successful Google signup, redirect to login
      // In a real app, you would save this user data to your backend
      navigate('/patient/login', { 
        state: { 
          message: 'Account created successfully! Please login with your Google account.',
          email: userData.email 
        } 
      });
    } catch (error) {
      console.error('Error processing Google response:', error);
      alert('Error processing Google signup. Please try again.');
    }
  };

  const fetchGoogleUserInfo = async (accessToken) => {
    try {
      const response = await fetch('https://www.googleapis.com/oauth2/v2/userinfo', {
        headers: {
          Authorization: `Bearer ${accessToken}`,
        },
      });
      const userData = await response.json();
      
      console.log('Google Signup Success:', userData);
      
      const fullName = userData.name || '';
      const nameParts = fullName.split(' ');
      const firstName = nameParts[0] || '';
      const lastName = nameParts.slice(1).join(' ') || '';

      navigate('/patient/login', { 
        state: { 
          message: 'Account created successfully! Please login with your Google account.',
          email: userData.email 
        } 
      });
    } catch (error) {
      console.error('Error fetching user info:', error);
      alert('Error processing Google signup. Please try again.');
    }
  };

  const formContent = (
    <form onSubmit={handleSubmit} className="space-y-4">
      <div className="grid grid-cols-2 gap-4">
        <div>
          <label className="block text-sm font-semibold text-slate-700 mb-2">First Name</label>
          <input
            type="text"
            name="firstName"
            value={formData.firstName}
            onChange={handleChange}
            placeholder="Enter first name"
            className="w-full px-4 py-2 border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-green-600"
            required
          />
        </div>

        <div>
          <label className="block text-sm font-semibold text-slate-700 mb-2">Last Name</label>
          <input
            type="text"
            name="lastName"
            value={formData.lastName}
            onChange={handleChange}
            placeholder="Enter last name"
            className="w-full px-4 py-2 border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-green-600"
            required
          />
        </div>
      </div>

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
          placeholder="Create a password"
          className="w-full px-4 py-2 border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-green-600"
          required
        />
      </div>

      <div>
        <label className="block text-sm font-semibold text-slate-700 mb-2">Confirm Password</label>
        <input
          type="password"
          name="confirmPassword"
          value={formData.confirmPassword}
          onChange={handleChange}
          placeholder="Confirm your password"
          className="w-full px-4 py-2 border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-green-600"
          required
        />
      </div>

      <button
        type="submit"
        className="w-full bg-green-600 text-white font-bold py-2 rounded-lg hover:bg-green-700 transition"
      >
        Sign up
      </button>

      <div className="text-center text-slate-600 font-medium">or</div>

      <div className="flex gap-3 justify-center">
        {!isIOS() && (
          <button
            type="button"
            onClick={handleGoogleSignup}
            className="flex items-center justify-center gap-2 px-6 py-2.5 border border-slate-200 rounded-lg hover:bg-slate-50 transition w-full"
            title="Sign up with Google"
          >
            <svg className="w-5 h-5" viewBox="0 0 24 24">
              <path fill="#EA4335" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z" />
              <path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z" />
              <path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z" />
              <path fill="#4285F4" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z" />
            </svg>
            <span className="text-sm font-medium text-slate-700">Continue with Google</span>
          </button>
        )}

        {isIOS() && (
          <button
            type="button"
            onClick={() => console.log('Apple signup')}
            className="flex items-center justify-center gap-2 px-6 py-2.5 border border-slate-200 rounded-lg hover:bg-slate-50 transition w-full"
            title="Sign up with Apple"
          >
            <svg className="w-5 h-5 text-slate-900" fill="currentColor" viewBox="0 0 24 24">
              <path d="M17.05 20.28c-.98.95-2.05 2.34-3.41 2.34-1.2 0-2.46-.9-3.69-2.34-.39-.45-.68-.85-.88-1.18-.74-1.22-1.28-2.91-1.28-4.78 0-3.32 2.34-5.24 4.54-5.24 1.13 0 2.22.56 2.83.56.55 0 1.93-.65 3.09-.65.25 0 .58.02.94.07 1.48.23 2.76.99 3.56 2.06-.09.06-2.02 1.13-2.02 4.3 0 2.94 2.13 4.29 2.18 4.32-.04.14-.3.94-1.19 2.22-.59.85-1.45 2.15-2.21 2.15l-2.46.17zm-4.32-15.5c.61-.75.98-1.74.98-2.78 0-.15 0-.29-.02-.44-1.12.06-2.3.62-2.97 1.41-.53.63-.9 1.55-.9 2.5 0 .15.01.3.02.41 1.02-.07 2.26-.59 2.89-1.1z" />
            </svg>
            <span className="text-sm font-medium text-slate-700">Continue with Apple</span>
          </button>
        )}
      </div>
    </form>
  );

  if (isEmbedded) {
    return formContent;
  }

  return (
    <div className="flex-1 min-h-screen flex items-center justify-center bg-slate-50 px-4">
      <div className="w-full max-w-md">
        <div className="text-center mb-8">
          <img src="/logo.jpg" alt="MediScan Logo" className="h-16 w-auto mx-auto mb-4 mix-blend-multiply" />
          <h1 className="text-3xl font-bold text-slate-900 mb-2">Patient Sign Up</h1>
          <p className="text-slate-600">Create your account to access your medical reports</p>
        </div>

        {formContent}

        <div className="mt-6 text-center">
          <p className="text-slate-600">
            Already have an account?{' '}
            <Link to="/patient/login" className="text-green-600 font-bold hover:underline">
              Login
            </Link>
          </p>
        </div>
      </div>
    </div>
  );
}

export default PatientSignup;
