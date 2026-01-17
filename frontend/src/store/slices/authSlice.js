import { createSlice, createAsyncThunk } from '@reduxjs/toolkit';
import axiosInstance from '../../api/axiosInstance';

export const signup = createAsyncThunk(
    'auth/signup',
    async (userData, { rejectWithValue }) => {
        try {
            const response = await axiosInstance.post('auth/register/', userData);
            return response.data;
        } catch (error) {
            return rejectWithValue(error.response?.data || 'Signup failed');
        }
    }
);

export const login = createAsyncThunk(
    'auth/login',
    async (credentials, { rejectWithValue }) => {
        try {
            const response = await axiosInstance.post('auth/login/', credentials);
            return response.data;
        } catch (error) {
            return rejectWithValue(error.response?.data || 'Login failed');
        }
    }
);

export const verifyOTP = createAsyncThunk(
    'auth/verifyOTP',
    async (otpData, { rejectWithValue }) => {
        try {
            const response = await axiosInstance.post('auth/verify-otp/', otpData);
            return response.data;
        } catch (error) {
            return rejectWithValue(error.response?.data || 'OTP verification failed');
        }
    }
);

export const googleLogin = createAsyncThunk(
    'auth/googleLogin',
    async (token, { rejectWithValue }) => {
        try {
            const response = await axiosInstance.post('auth/google-login/', { token });
            return response.data;
        } catch (error) {
            return rejectWithValue(error.response?.data || 'Google login failed');
        }
    }
);

export const updateProfile = createAsyncThunk(
    'auth/updateProfile',
    async (profileData, { rejectWithValue }) => {
        try {
            const response = await axiosInstance.patch('auth/profile/', profileData);
            return response.data;
        } catch (error) {
            return rejectWithValue(error.response?.data || 'Profile update failed');
        }
    }
);

const authSlice = createSlice({
    name: 'auth',
    initialState: {
        user: (() => {
            const u = localStorage.getItem('user');
            try {
                return u && u !== 'undefined' ? JSON.parse(u) : null;
            } catch {
                return null;
            }
        })(),
        token: (() => {
            const t = localStorage.getItem('token');
            return t && t !== 'undefined' ? t : null;
        })(),
        loading: false,
        error: null,
        tempEmail: null,
        otpSent: false,
        authType: null, // 'login' or 'register'
    },
    reducers: {
        logout: (state) => {
            state.user = null;
            state.token = null;
            state.otpSent = false;
            state.tempEmail = null;
            state.authType = null;
            localStorage.removeItem('token');
            localStorage.removeItem('user');
        },
        clearError: (state) => {
            state.error = null;
        },
    },
    extraReducers: (builder) => {
        builder
            // Signup
            .addCase(signup.pending, (state) => {
                state.loading = true;
                state.error = null;
            })
            .addCase(signup.fulfilled, (state, action) => {
                state.loading = false;
                state.otpSent = true;
                state.tempEmail = action.meta.arg.email;
                state.authType = 'register';
            })
            .addCase(signup.rejected, (state, action) => {
                state.loading = false;
                state.error = action.payload;
            })
            // Login
            .addCase(login.pending, (state) => {
                state.loading = true;
                state.error = null;
            })
            .addCase(login.fulfilled, (state, action) => {
                state.loading = false;
                state.otpSent = true;
                state.tempEmail = action.meta.arg.email;
                state.authType = 'login';
            })
            .addCase(login.rejected, (state, action) => {
                state.loading = false;
                state.error = action.payload;
            })
            // Verify OTP
            .addCase(verifyOTP.pending, (state) => {
                state.loading = true;
                state.error = null;
            })
            .addCase(verifyOTP.fulfilled, (state, action) => {
                state.loading = false;
                state.user = action.payload.user;
                state.token = action.payload.access;
                localStorage.setItem('token', action.payload.access);
                localStorage.setItem('user', JSON.stringify(action.payload.user));
                state.otpSent = false;
                state.tempEmail = null;
                state.authType = null;
            })
            .addCase(verifyOTP.rejected, (state, action) => {
                state.loading = false;
                state.error = action.payload;
            })
            // Google Login
            .addCase(googleLogin.pending, (state) => {
                state.loading = true;
                state.error = null;
            })
            .addCase(googleLogin.fulfilled, (state, action) => {
                state.loading = false;
                state.user = action.payload.user;
                state.token = action.payload.access;
                localStorage.setItem('token', action.payload.access);
                localStorage.setItem('user', JSON.stringify(action.payload.user));
            })
            .addCase(googleLogin.rejected, (state, action) => {
                state.loading = false;
                state.error = action.payload;
            })
            // Update Profile
            .addCase(updateProfile.pending, (state) => {
                state.loading = true;
                state.error = null;
            })
            .addCase(updateProfile.fulfilled, (state, action) => {
                state.loading = false;
                state.user = action.payload.user;
                localStorage.setItem('user', JSON.stringify(action.payload.user));
            })
            .addCase(updateProfile.rejected, (state, action) => {
                state.loading = false;
                state.error = action.payload;
            });
    },
});

export const { logout, clearError } = authSlice.actions;
export default authSlice.reducer;
