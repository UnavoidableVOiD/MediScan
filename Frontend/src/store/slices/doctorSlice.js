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

export const submitDoctorComment = createAsyncThunk(
    'doctor/submitComment',
    async (data, { rejectWithValue }) => {
        try {
            const response = await doctorApi.addComment(data);
            return response.data;
        } catch (error) {
            return rejectWithValue(error.response?.data || 'Failed to submit comment');
        }
    }
);

export const fetchPatientTrends = createAsyncThunk(
    'doctor/fetchTrends',
    async (patientId, { rejectWithValue }) => {
        try {
            const response = await doctorApi.getPatientTrends(patientId);
            return response.data;
        } catch (error) {
            return rejectWithValue(error.response?.data || 'Failed to fetch trends');
        }
    }
);

export const markPatientCompleted = createAsyncThunk(
    'doctor/markCompleted',
    async (patientId, { rejectWithValue }) => {
        try {
            const response = await doctorApi.markPatientCompleted(patientId);
            return { patientId, ...response.data };
        } catch (error) {
            return rejectWithValue(error.response?.data || 'Failed to mark as completed');
        }
    }
);

export const updateClinicalObservations = createAsyncThunk(
    'doctor/updateObservations',
    async ({ patientId, observations }, { rejectWithValue }) => {
        try {
            const response = await doctorApi.updateClinicalObservations(patientId, observations);
            return response.data;
        } catch (error) {
            return rejectWithValue(error.response?.data || 'Failed to update observations');
        }
    }
);

// --- Slice ---

const initialState = {
    stats: null,
    patients: [],
    currentPatientReports: [],
    currentPatientTrends: [],
    loading: false,
    statsLoading: false,
    reportsLoading: false,
    trendsLoading: false,
    notesLoading: false,
    commentLoading: false,
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
                // Update in patients array
                const patient = state.patients.find(p => p.id === parseInt(action.meta.arg.patientId));
                if (patient) {
                    patient.notes = action.meta.arg.notes;
                }
            })
            .addCase(updatePatientNotes.rejected, (state, action) => {
                state.notesLoading = false;
                toast.error(action.payload?.error || "Failed to save notes");
            })
            // Submit Comment
            .addCase(submitDoctorComment.pending, (state) => {
                state.commentLoading = true;
            })
            .addCase(submitDoctorComment.fulfilled, (state, action) => {
                state.commentLoading = false;
                toast.success("Comment submitted successfully");
                // Update the report in currentPatientReports if it's there
                const reportIndex = state.currentPatientReports.findIndex(r => r.id === action.payload.report);
                if (reportIndex !== -1) {
                    state.currentPatientReports[reportIndex].doctor_comment = action.payload;
                }
            })
            .addCase(submitDoctorComment.rejected, (state, action) => {
                state.commentLoading = false;
                toast.error(action.payload?.error || "Failed to submit comment");
            })
            // Fetch Trends
            .addCase(fetchPatientTrends.pending, (state) => {
                state.trendsLoading = true;
            })
            .addCase(fetchPatientTrends.fulfilled, (state, action) => {
                state.trendsLoading = false;
                state.currentPatientTrends = action.payload;
            })
            .addCase(fetchPatientTrends.rejected, (state, action) => {
                state.trendsLoading = false;
            })
            // Mark Completed
            .addCase(markPatientCompleted.fulfilled, (state, action) => {
                toast.success("Patient marked as completed");
                const patient = state.patients.find(p => p.id === parseInt(action.payload.patientId));
                if (patient) {
                    patient.status = 'COMPLETED';
                }
            })
            // Update Observations
            .addCase(updateClinicalObservations.pending, (state) => {
                state.notesLoading = true;
            })
            .addCase(updateClinicalObservations.fulfilled, (state, action) => {
                state.notesLoading = false;
                toast.success("Observations updated successfully");
                // Update in patients array
                const patient = state.patients.find(p => p.id === parseInt(action.meta.arg.patientId));
                if (patient) {
                    patient.clinical_observations = action.meta.arg.observations;
                }
            })
            .addCase(updateClinicalObservations.rejected, (state, action) => {
                state.notesLoading = false;
                toast.error(action.payload?.error || "Failed to update observations");
            });
    },
});

export const { clearDoctorState, clearPatientReports } = doctorSlice.actions;
export default doctorSlice.reducer;
