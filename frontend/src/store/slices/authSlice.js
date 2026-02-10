import { createSlice, createAsyncThunk } from '@reduxjs/toolkit';
import api, { authApi } from '../../services/api';
import { toast } from 'react-toastify';

export const submitVerification = createAsyncThunk(
    'auth/submitVerification',
    async (formData, { rejectWithValue }) => {
        try {
            const response = await authApi.submitDoctorVerification(formData);
            return response.data;
        } catch (error) {
            return rejectWithValue(error.response.data);
        }
    }
);

const initialState = {
    user: null,
    isAuthenticated: false,
    tempEmail: null,
    isVerifying: false,
    flowType: null, // 'login' or 'register'
    loading: false,
    error: null,
    initialized: false, // For initial auth check
};

// Async Thunks
export const registerUser = createAsyncThunk(
    'auth/register',
    async (userData, { rejectWithValue }) => {
        try {
            const response = await api.post('/auth/register/', userData);
            return response.data;
        } catch (error) {
            return rejectWithValue(error.response.data);
        }
    }
);

export const loginUser = createAsyncThunk(
    'auth/login',
    async (loginData, { rejectWithValue }) => {
        try {
            const response = await api.post('/auth/login/', loginData);
            return response.data;
        } catch (error) {
            return rejectWithValue(error.response.data);
        }
    }
);

export const verifyOTP = createAsyncThunk(
    'auth/verifyOTP',
    async (otpData, { rejectWithValue }) => {
        try {
            const response = await api.post('/auth/verify-otp/', otpData);
            return response.data;
        } catch (error) {
            return rejectWithValue(error.response.data);
        }
    }
);

export const checkAuth = createAsyncThunk(
    'auth/checkAuth',
    async (_, { rejectWithValue }) => {
        try {
            const response = await api.get('/auth/profile/');
            return response.data;
        } catch (error) {
            return rejectWithValue(error.response?.data);
        }
    }
);

export const googleLogin = createAsyncThunk(
    'auth/googleLogin',
    async (token, { rejectWithValue }) => {
        try {
            const response = await api.post('/auth/google-login/', { token });
            return response.data;
        } catch (error) {
            return rejectWithValue(error.response.data);
        }
    }
);

export const logoutUser = createAsyncThunk(
    'auth/logout',
    async (_, { rejectWithValue }) => {
        try {
            const response = await api.post('/auth/logout/');
            return response.data;
        } catch (error) {
            return rejectWithValue(error.response?.data);
        }
    }
);

export const updateProfile = createAsyncThunk(
    'auth/updateProfile',
    async (profileData, { rejectWithValue }) => {
        try {
            const response = await api.patch('/auth/profile/', profileData);
            return response.data;
        } catch (error) {
            return rejectWithValue(error.response.data);
        }
    }
);

const authSlice = createSlice({
    name: 'auth',
    initialState,
    reducers: {
        setCredentials: (state, action) => {
            const { user } = action.payload;
            state.user = user;
            state.isAuthenticated = true;
        },
        setTempEmail: (state, action) => {
            state.tempEmail = action.payload;
            state.isVerifying = true;
        },
        clearVerification: (state) => {
            state.isVerifying = false;
            state.tempEmail = null;
        },
        resetError: (state) => {
            state.error = null;
        }
    },
    extraReducers: (builder) => {
        builder
            // Register
            .addCase(registerUser.pending, (state) => {
                state.loading = true;
                state.error = null;
                state.flowType = 'register';
            })
            .addCase(registerUser.fulfilled, (state, action) => {
                state.loading = false;
                state.isVerifying = true;
                state.tempEmail = action.meta.arg.email;
                toast.success(action.payload.message || "OTP sent to your email!");
            })
            .addCase(registerUser.rejected, (state, action) => {
                state.loading = false;
                state.error = action.payload;
                state.flowType = null;
                toast.error(action.payload?.message || "Registration failed");
            })
            // Login
            .addCase(loginUser.pending, (state) => {
                state.loading = true;
                state.error = null;
                state.flowType = 'login';
            })
            .addCase(loginUser.fulfilled, (state, action) => {
                state.loading = false;
                state.isVerifying = true;
                state.tempEmail = action.meta.arg.email;
                toast.success(action.payload.message || "OTP sent to your email!");
            })
            .addCase(loginUser.rejected, (state, action) => {
                state.loading = false;
                state.error = action.payload;
                state.flowType = null;
                toast.error(action.payload?.message || "Login failed");
            })
            // Verify OTP
            .addCase(verifyOTP.pending, (state) => {
                state.loading = true;
                state.error = null;
            })
            .addCase(verifyOTP.fulfilled, (state, action) => {
                state.loading = false;
                state.isVerifying = false;
                state.tempEmail = null;
                state.flowType = null;
                if (action.payload.user) {
                    state.isAuthenticated = true;
                    state.user = action.payload.user;
                    toast.success("Login successful!");
                } else {
                    toast.success(action.payload.message || "Verification successful! Please login.");
                }
            })
            .addCase(verifyOTP.rejected, (state, action) => {
                state.loading = false;
                state.error = action.payload;
                toast.error(action.payload?.error || action.payload?.message || "Verification failed");
            })
            // Google Login
            .addCase(googleLogin.pending, (state) => {
                state.loading = true;
                state.error = null;
            })
            .addCase(googleLogin.fulfilled, (state, action) => {
                state.loading = false;
                state.isAuthenticated = true;
                state.user = action.payload.user;
                toast.success("Google Login successful!");
            })
            .addCase(googleLogin.rejected, (state, action) => {
                state.loading = false;
                state.error = action.payload;
                toast.error(action.payload?.error || "Google Login failed");
            })
            // Logout
            .addCase(logoutUser.fulfilled, (state) => {
                state.user = null;
                state.isAuthenticated = false;
                toast.info("Logged out successfully");
            })
            .addCase(logoutUser.rejected, (state) => {
                // Force logout anyway
                state.user = null;
                state.isAuthenticated = false;
            })
            // Submit Verification
            .addCase(submitVerification.pending, (state) => {
                state.loading = true;
                state.error = null;
            })
            .addCase(submitVerification.fulfilled, (state, action) => {
                state.loading = false;
                if (state.user) {
                    state.user.doctor_status = 'PENDING';
                }
                toast.success(action.payload.message || "Documents submitted successfully! Your profile is now under review.");
            })
            .addCase(submitVerification.rejected, (state, action) => {
                state.loading = false;
                state.error = action.payload;
                toast.error(action.payload?.message || action.payload?.error || "Submission failed");
            })
            // Update Profile
            .addCase(updateProfile.pending, (state) => {
                state.loading = true;
                state.error = null;
            })
            .addCase(updateProfile.fulfilled, (state, action) => {
                state.loading = false;
                state.user = action.payload.user;
                toast.success(action.payload.message || "Profile updated successfully!");
            })
            .addCase(updateProfile.rejected, (state, action) => {
                state.loading = false;
                state.error = action.payload;

                // Handle field-specific errors from DRF
                const errorData = action.payload;
                let errorMessage = "Profile update failed";

                if (errorData) {
                    if (typeof errorData === 'object') {
                        const firstKey = Object.keys(errorData)[0];
                        if (Array.isArray(errorData[firstKey])) {
                            errorMessage = `${firstKey.replace('_', ' ')}: ${errorData[firstKey][0]}`;
                        } else if (errorData.message) {
                            errorMessage = errorData.message;
                        }
                    }
                }

                toast.error(errorMessage);
            })
            // Check Auth
            .addCase(checkAuth.pending, (state) => {
                state.loading = true;
            })
            .addCase(checkAuth.fulfilled, (state, action) => {
                state.loading = false;
                state.initialized = true;
                state.isAuthenticated = true;
                state.user = action.payload;
            })
            .addCase(checkAuth.rejected, (state) => {
                state.loading = false;
                state.initialized = true;
                state.isAuthenticated = false;
                state.user = null;
            });
    },
});

export const { setTempEmail, clearVerification, resetError, setCredentials } = authSlice.actions;
export default authSlice.reducer;
