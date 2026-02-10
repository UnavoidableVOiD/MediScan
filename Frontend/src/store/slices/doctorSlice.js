import { createSlice, createAsyncThunk } from '@reduxjs/toolkit';
import { doctorApi } from '../../services/api';
import { toast } from 'react-toastify';

// --- Async Thunks ---

export const fetchDoctorStats = createAsyncThunk(
    'doctor/fetchStats',
    async (_, { rejectWithValue }) => {
        try {
            const response = await doctorApi.getStats();
            return response.data;
        } catch (error) {
            return rejectWithValue(error.response?.data || 'Failed to fetch stats');
        }
    }
);

export const fetchMyPatients = createAsyncThunk(
    'doctor/fetchPatients',
    async (_, { rejectWithValue }) => {
        try {
            const response = await doctorApi.getPatients();
            return response.data;
        } catch (error) {
            return rejectWithValue(error.response?.data || 'Failed to fetch patients');
        }
    }
);

export const fetchPatientReports = createAsyncThunk(
    'doctor/fetchPatientReports',
    async (patientId, { rejectWithValue }) => {
        try {
            const response = await doctorApi.getPatientReports(patientId);
            return response.data;
        } catch (error) {
            return rejectWithValue(error.response?.data || 'Failed to fetch patient reports');
        }
    }
);

export const updatePatientNotes = createAsyncThunk(
    'doctor/updatePatientNotes',
    async ({ patientId, notes }, { rejectWithValue }) => {
        try {
            const response = await doctorApi.updatePatientNotes(patientId, notes);
            return response.data;
        } catch (error) {
            return rejectWithValue(error.response?.data || 'Failed to update notes');
        }
    }
);

// --- Slice ---

const initialState = {
    stats: null,
    patients: [],
    currentPatientReports: [],
    loading: false,
    statsLoading: false,
    reportsLoading: false,
    notesLoading: false,
    error: null,
};

const doctorSlice = createSlice({
    name: 'doctor',
    initialState,
    reducers: {
        clearDoctorState: (state) => {
            Object.assign(state, initialState);
        },
        clearPatientReports: (state) => {
            state.currentPatientReports = [];
        },
    },
    extraReducers: (builder) => {
        builder
            // Stats
            .addCase(fetchDoctorStats.pending, (state) => {
                state.statsLoading = true;
                state.error = null;
            })
            .addCase(fetchDoctorStats.fulfilled, (state, action) => {
                state.statsLoading = false;
                state.stats = action.payload;
            })
            .addCase(fetchDoctorStats.rejected, (state, action) => {
                state.statsLoading = false;
                state.error = action.payload;
            })
            // Patients
            .addCase(fetchMyPatients.pending, (state) => {
                state.loading = true;
                state.error = null;
            })
            .addCase(fetchMyPatients.fulfilled, (state, action) => {
                state.loading = false;
                state.patients = action.payload;
            })
            .addCase(fetchMyPatients.rejected, (state, action) => {
                state.loading = false;
                state.error = action.payload;
            })
            // Patient Reports
            .addCase(fetchPatientReports.pending, (state) => {
                state.reportsLoading = true;
                state.error = null;
            })
            .addCase(fetchPatientReports.fulfilled, (state, action) => {
                state.reportsLoading = false;
                state.currentPatientReports = action.payload;
            })
            .addCase(fetchPatientReports.rejected, (state, action) => {
                state.reportsLoading = false;
                state.error = action.payload;
            })
            // Update Notes
            .addCase(updatePatientNotes.pending, (state) => {
                state.notesLoading = true;
            })
            .addCase(updatePatientNotes.fulfilled, (state, action) => {
                state.notesLoading = false;
                toast.success("Notes saved successfully");
            })
            .addCase(updatePatientNotes.rejected, (state, action) => {
                state.notesLoading = false;
                toast.error(action.payload?.error || "Failed to save notes");
            });
    },
});

export const { clearDoctorState, clearPatientReports } = doctorSlice.actions;
export default doctorSlice.reducer;
