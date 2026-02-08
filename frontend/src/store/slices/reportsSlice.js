import { createSlice, createAsyncThunk } from '@reduxjs/toolkit';
import api from '../../services/api';

export const fetchReports = createAsyncThunk(
    'reports/fetchAll',
    async (_, { rejectWithValue }) => {
        try {
            const response = await api.get('/reports/');
            return response.data;
        } catch (error) {
            return rejectWithValue(error.response?.data || 'Failed to fetch reports');
        }
    }
);

export const uploadReport = createAsyncThunk(
    'reports/upload',
    async (formData, { rejectWithValue }) => {
        try {
            const response = await api.post('/reports/', formData, {
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

export const processReport = createAsyncThunk(
    'reports/process',
    async (id, { rejectWithValue }) => {
        try {
            const response = await api.post(`/reports/${id}/process/`);
            return response.data;
        } catch (error) {
            return rejectWithValue(error.response?.data || 'Extraction failed');
        }
    }
);


export const correctReportData = createAsyncThunk(
    'reports/correct',
    async ({ id, final_data }, { rejectWithValue }) => {
        try {
            const response = await api.put(`/reports/${id}/correct/`, { final_data });
            return response.data;
        } catch (error) {
            return rejectWithValue(error.response?.data || 'Failed to correct report data');
        }
    }
);

export const fetchReportDetail = createAsyncThunk(
    'reports/fetchDetail',
    async (id, { rejectWithValue }) => {
        try {
            const response = await api.get(`/reports/${id}/`);
            return response.data;
        } catch (error) {
            return rejectWithValue(error.response?.data || 'Failed to fetch report detail');
        }
    }
);

export const fetchReportResult = createAsyncThunk(
    'reports/fetchResult',
    async (id, { rejectWithValue }) => {
        try {
            const response = await api.get(`/reports/${id}/result/`);
            return response.data;
        } catch (error) {
            return rejectWithValue(error.response?.data || 'Failed to fetch analysis result');
        }
    }
);


export const deleteReport = createAsyncThunk(
    'reports/delete',
    async (id, { rejectWithValue }) => {
        try {
            await api.delete(`/reports/${id}/`);
            return id;
        } catch (error) {
            return rejectWithValue(error.response?.data || 'Failed to delete report');
        }
    }
);


const initialState = {
    reports: [],
    currentReport: null,
    currentResult: null, // New state for AI analysis
    loading: false,
    uploading: false,
    processing: false, // New state for extraction
    correcting: false,
    error: null,
    success: false,
};



const reportsSlice = createSlice({
    name: 'reports',
    initialState,
    reducers: {
        resetStatus: (state) => {
            state.success = false;
            state.error = null;
        },
        clearCurrentReport: (state) => {
            state.currentReport = null;
            state.currentResult = null;
        },

        setCurrentReport: (state, action) => {
            state.currentReport = action.payload;
        }
    },

    extraReducers: (builder) => {
        builder
            // Fetch all reports
            .addCase(fetchReports.pending, (state) => {
                state.loading = true;
                state.error = null;
            })
            .addCase(fetchReports.fulfilled, (state, action) => {
                state.loading = false;
                state.reports = action.payload;
            })
            .addCase(fetchReports.rejected, (state, action) => {
                state.loading = false;
                state.error = action.payload;
            })
            // Upload report
            .addCase(uploadReport.pending, (state) => {
                state.uploading = true;
                state.error = null;
                state.success = false;
            })
            .addCase(uploadReport.fulfilled, (state, action) => {
                console.log("[Redux] uploadReport.fulfilled:", action.payload);
                state.uploading = false;
                state.reports.unshift(action.payload);
                state.currentReport = action.payload;
                // We don't set success=true here yet, extraction is next
            })

            .addCase(uploadReport.rejected, (state, action) => {
                state.uploading = false;
                state.error = action.payload;
            })
            // Process (Extraction)
            .addCase(processReport.pending, (state) => {
                state.processing = true;
                state.error = null;
                state.success = false;
            })
            .addCase(processReport.fulfilled, (state, action) => {
                state.processing = false;
                state.currentReport = action.payload;
                // Extraction succeeded
            })
            .addCase(processReport.rejected, (state, action) => {
                state.processing = false;
                state.error = action.payload;
            })

            // Correct data
            .addCase(correctReportData.pending, (state) => {
                state.correcting = true;
                state.error = null;
            })
            .addCase(correctReportData.fulfilled, (state, action) => {
                state.correcting = false;
                state.currentReport = action.payload;
                // Update in list as well
                const index = state.reports.findIndex(r => r.id === action.payload.id);
                if (index !== -1) {
                    state.reports[index] = action.payload;
                }
            })
            .addCase(correctReportData.rejected, (state, action) => {
                state.correcting = false;
                state.error = action.payload;
            })
            // Fetch detail
            .addCase(fetchReportDetail.pending, (state) => {
                state.loading = true;
                state.error = null;
            })
            .addCase(fetchReportDetail.fulfilled, (state, action) => {
                state.loading = false;
                state.currentReport = action.payload;
            })
            .addCase(fetchReportDetail.rejected, (state, action) => {
                state.loading = false;
                state.error = action.payload;
            })
            // Delete report
            .addCase(deleteReport.pending, (state) => {
                state.loading = true;
                state.error = null;
            })
            .addCase(deleteReport.fulfilled, (state, action) => {
                state.loading = false;
                state.reports = state.reports.filter(r => r.id !== action.payload);
                if (state.currentReport?.id === action.payload) {
                    state.currentReport = null;
                }
            })
            .addCase(deleteReport.rejected, (state, action) => {
                state.loading = false;
                state.error = action.payload;
            })
            // Fetch result
            .addCase(fetchReportResult.pending, (state) => {
                state.loading = true;
                state.error = null;
            })
            .addCase(fetchReportResult.fulfilled, (state, action) => {
                state.loading = false;
                state.currentResult = action.payload;
            })
            .addCase(fetchReportResult.rejected, (state, action) => {
                state.loading = false;
                state.error = action.payload;
            });
    },
});



export const { resetStatus, clearCurrentReport, setCurrentReport } = reportsSlice.actions;

export default reportsSlice.reducer;
