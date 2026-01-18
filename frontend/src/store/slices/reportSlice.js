import { createSlice, createAsyncThunk } from '@reduxjs/toolkit';
import axiosInstance from '../../api/axiosInstance';

export const fetchReports = createAsyncThunk(
    'reports/fetchReports',
    async (_, { rejectWithValue }) => {
        try {
            const response = await axiosInstance.get('reports/');
            return response.data;
        } catch (error) {
            return rejectWithValue(error.response?.data || 'Failed to fetch reports');
        }
    }
);

export const uploadReport = createAsyncThunk(
    'reports/uploadReport',
    async (formData, { rejectWithValue }) => {
        try {
            const response = await axiosInstance.post('reports/', formData, {
                headers: {
                    'Content-Type': 'multipart/form-data',
                },
            });
            return response.data;
        } catch (error) {
            return rejectWithValue(error.response?.data || 'Failed to upload report');
        }
    }
);

const reportSlice = createSlice({
    name: 'reports',
    initialState: {
        reports: [],
        loading: false,
        uploading: false,
        error: null,
        stats: {
            total: 0,
            pending: 0,
            completed: 0,
            failed: 0,
        },
    },
    reducers: {
        clearReportError: (state) => {
            state.error = null;
        },
    },
    extraReducers: (builder) => {
        builder
            // Fetch Reports
            .addCase(fetchReports.pending, (state) => {
                state.loading = true;
                state.error = null;
            })
            .addCase(fetchReports.fulfilled, (state, action) => {
                state.loading = false;
                // Handle both direct array and paginated response (DRF default)
                const reportsArray = Array.isArray(action.payload)
                    ? action.payload
                    : action.payload?.results || [];

                state.reports = reportsArray;

                // Calculate stats safely
                state.stats = {
                    total: reportsArray.length,
                    pending: reportsArray.filter(r => r.status === 'PENDING').length,
                    completed: reportsArray.filter(r => r.status === 'PROCESSED').length,
                    failed: reportsArray.filter(r => r.status === 'FAILED').length,
                };
            })
            .addCase(fetchReports.rejected, (state, action) => {
                state.loading = false;
                state.error = action.payload;
            })
            // Upload Report
            .addCase(uploadReport.pending, (state) => {
                state.uploading = true;
                state.error = null;
            })
            .addCase(uploadReport.fulfilled, (state, action) => {
                state.uploading = false;
                state.reports.unshift(action.payload);
                state.stats.total += 1;
                state.stats.pending += 1;
            })
            .addCase(uploadReport.rejected, (state, action) => {
                state.uploading = false;
                state.error = action.payload;
            });
    },
});

export const { clearReportError } = reportSlice.actions;
export default reportSlice.reducer;
