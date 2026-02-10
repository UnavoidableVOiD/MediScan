import { createSlice, createAsyncThunk } from '@reduxjs/toolkit';
import { adminApi } from '../../services/api';
import { toast } from 'react-toastify';

// --- Async Thunks ---

export const adminLogin = createAsyncThunk(
    'admin/login',
    async (credentials, { rejectWithValue }) => {
        try {
            const response = await adminApi.login(credentials);
            return response.data;
        } catch (error) {
            return rejectWithValue(error.response?.data || 'Login failed');
        }
    }
);

export const fetchAdminDoctors = createAsyncThunk(
    'admin/fetchDoctors',
    async (status, { rejectWithValue }) => {
        try {
            const response = await adminApi.getDoctors(status);
            return response.data;
        } catch (error) {
            return rejectWithValue(error.response?.data || 'Failed to fetch doctors');
        }
    }
);

export const verifyDoctor = createAsyncThunk(
    'admin/verifyDoctor',
    async ({ id, data }, { rejectWithValue }) => {
        try {
            const response = await adminApi.verifyDoctor(id, data);
            return response.data;
        } catch (error) {
            return rejectWithValue(error.response?.data || 'Verification failed');
        }
    }
);

export const updateDoctor = createAsyncThunk(
    'admin/updateDoctor',
    async ({ id, data }, { rejectWithValue }) => {
        try {
            const response = await adminApi.updateDoctor(id, data);
            return response.data;
        } catch (error) {
            return rejectWithValue(error.response?.data || 'Update failed');
        }
    }
);

export const deleteDoctor = createAsyncThunk(
    'admin/deleteDoctor',
    async (id, { rejectWithValue }) => {
        try {
            await adminApi.deleteDoctor(id);
            return id;
        } catch (error) {
            return rejectWithValue(error.response?.data || 'Delete failed');
        }
    }
);

export const unverifyDoctor = createAsyncThunk(
    'admin/unverifyDoctor',
    async (id, { rejectWithValue }) => {
        try {
            const response = await adminApi.unverifyDoctor(id);
            return response.data;
        } catch (error) {
            return rejectWithValue(error.response?.data || 'Unverify failed');
        }
    }
);

export const fetchAdminPatients = createAsyncThunk(
    'admin/fetchPatients',
    async (_, { rejectWithValue }) => {
        try {
            const response = await adminApi.getPatients();
            return response.data;
        } catch (error) {
            return rejectWithValue(error.response?.data || 'Failed to fetch patients');
        }
    }
);

export const updatePatient = createAsyncThunk(
    'admin/updatePatient',
    async ({ id, data }, { rejectWithValue }) => {
        try {
            const response = await adminApi.updatePatient(id, data);
            return response.data;
        } catch (error) {
            return rejectWithValue(error.response?.data || 'Update failed');
        }
    }
);

export const deletePatient = createAsyncThunk(
    'admin/deletePatient',
    async (id, { rejectWithValue }) => {
        try {
            await adminApi.deletePatient(id);
            return id;
        } catch (error) {
            return rejectWithValue(error.response?.data || 'Delete failed');
        }
    }
);

export const createAdmin = createAsyncThunk(
    'admin/createAdmin',
    async (data, { rejectWithValue }) => {
        try {
            const response = await adminApi.createAdmin(data);
            return response.data;
        } catch (error) {
            return rejectWithValue(error.response?.data || 'Failed to create admin');
        }
    }
);

// --- Slice ---

const initialState = {
    doctors: [],
    patients: [],
    loading: false,
    actionLoading: false,
    error: null,
};

const adminSlice = createSlice({
    name: 'admin',
    initialState,
    reducers: {
        clearAdminState: (state) => {
            Object.assign(state, initialState);
        },
    },
    extraReducers: (builder) => {
        builder
            // Login
            .addCase(adminLogin.pending, (state) => {
                state.loading = true;
                state.error = null;
            })
            .addCase(adminLogin.fulfilled, (state) => {
                state.loading = false;
                toast.success("Welcome back, Admin!");
            })
            .addCase(adminLogin.rejected, (state, action) => {
                state.loading = false;
                state.error = action.payload;
                toast.error(action.payload?.error || "Login failed");
            })
            // Fetch Doctors
            .addCase(fetchAdminDoctors.pending, (state) => {
                state.loading = true;
                state.error = null;
            })
            .addCase(fetchAdminDoctors.fulfilled, (state, action) => {
                state.loading = false;
                state.doctors = action.payload;
            })
            .addCase(fetchAdminDoctors.rejected, (state, action) => {
                state.loading = false;
                state.error = action.payload;
                toast.error("Failed to fetch doctors");
            })
            // Verify Doctor
            .addCase(verifyDoctor.pending, (state) => {
                state.actionLoading = true;
            })
            .addCase(verifyDoctor.fulfilled, (state, action) => {
                state.actionLoading = false;
                const idx = state.doctors.findIndex(d => d.id === action.payload.id);
                if (idx !== -1) state.doctors[idx] = action.payload;
                toast.success("Doctor status updated");
            })
            .addCase(verifyDoctor.rejected, (state, action) => {
                state.actionLoading = false;
                toast.error(action.payload?.error || "Verification action failed");
            })
            // Update Doctor
            .addCase(updateDoctor.pending, (state) => {
                state.actionLoading = true;
            })
            .addCase(updateDoctor.fulfilled, (state, action) => {
                state.actionLoading = false;
                const idx = state.doctors.findIndex(d => d.id === action.payload.id);
                if (idx !== -1) state.doctors[idx] = action.payload;
                toast.success("Account updated successfully");
            })
            .addCase(updateDoctor.rejected, (state, action) => {
                state.actionLoading = false;
                toast.error("Failed to update account");
            })
            // Delete Doctor
            .addCase(deleteDoctor.pending, (state) => {
                state.actionLoading = true;
            })
            .addCase(deleteDoctor.fulfilled, (state, action) => {
                state.actionLoading = false;
                state.doctors = state.doctors.filter(d => d.id !== action.payload);
                toast.success("Doctor deleted successfully");
            })
            .addCase(deleteDoctor.rejected, (state, action) => {
                state.actionLoading = false;
                toast.error(action.payload?.error || "Failed to delete doctor");
            })
            // Unverify Doctor
            .addCase(unverifyDoctor.pending, (state) => {
                state.actionLoading = true;
            })
            .addCase(unverifyDoctor.fulfilled, (state, action) => {
                state.actionLoading = false;
                const idx = state.doctors.findIndex(d => d.id === action.payload.user?.id);
                if (idx !== -1) state.doctors[idx] = action.payload.user;
                toast.success(action.payload.message || "Doctor unverified successfully");
            })
            .addCase(unverifyDoctor.rejected, (state, action) => {
                state.actionLoading = false;
                toast.error(action.payload?.message || "unverify acton failed");
            })
            // Fetch Patients
            .addCase(fetchAdminPatients.pending, (state) => {
                state.loading = true;
                state.error = null;
            })
            .addCase(fetchAdminPatients.fulfilled, (state, action) => {
                state.loading = false;
                state.patients = action.payload;
            })
            .addCase(fetchAdminPatients.rejected, (state, action) => {
                state.loading = false;
                state.error = action.payload;
                toast.error("Failed to fetch patients");
            })
            // Update Patient
            .addCase(updatePatient.pending, (state) => {
                state.actionLoading = true;
            })
            .addCase(updatePatient.fulfilled, (state, action) => {
                state.actionLoading = false;
                const idx = state.patients.findIndex(p => p.id === action.payload.id);
                if (idx !== -1) state.patients[idx] = action.payload;
                toast.success("Account updated successfully");
            })
            .addCase(updatePatient.rejected, (state, action) => {
                state.actionLoading = false;
                toast.error("Failed to update account");
            })
            // Delete Patient
            .addCase(deletePatient.pending, (state) => {
                state.actionLoading = true;
            })
            .addCase(deletePatient.fulfilled, (state, action) => {
                state.actionLoading = false;
                state.patients = state.patients.filter(p => p.id !== action.payload);
                toast.success("Patient deleted successfully");
            })
            .addCase(deletePatient.rejected, (state, action) => {
                state.actionLoading = false;
                toast.error(action.payload?.error || "Failed to delete patient");
            })
            // Create Admin
            .addCase(createAdmin.pending, (state) => {
                state.actionLoading = true;
            })
            .addCase(createAdmin.fulfilled, (state) => {
                state.actionLoading = false;
                toast.success("New Admin created successfully!");
            })
            .addCase(createAdmin.rejected, (state, action) => {
                state.actionLoading = false;
                toast.error(action.payload?.error || "Failed to create admin");
            });
    },
});

export const { clearAdminState } = adminSlice.actions;
export default adminSlice.reducer;
