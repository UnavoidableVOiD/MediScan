import { createSlice, createAsyncThunk } from '@reduxjs/toolkit';
import { appointmentApi } from '../../services/api.js';
import { toast } from 'react-toastify';

export const fetchAvailability = createAsyncThunk(
    'appointment/fetchAvailability',
    async (doctorId, { rejectWithValue }) => {
        try {
            const response = await appointmentApi.getAvailability(doctorId);
            return response.data;
        } catch (error) {
            return rejectWithValue(error.response?.data || 'Failed to fetch availability');
        }
    }
);

export const fetchAppointments = createAsyncThunk(
    'appointment/fetchAppointments',
    async (_, { rejectWithValue }) => {
        try {
            const response = await appointmentApi.getAppointments();
            return response.data;
        } catch (error) {
            return rejectWithValue(error.response?.data || 'Failed to fetch appointments');
        }
    }
);

export const bookAppointment = createAsyncThunk(
    'appointment/book',
    async (data, { rejectWithValue }) => {
        try {
            const response = await appointmentApi.bookAppointment(data);
            return response.data;
        } catch (error) {
            return rejectWithValue(error.response?.data || 'Failed to book appointment');
        }
    }
);

export const verifyPayment = createAsyncThunk(
    'appointment/verifyPayment',
    async ({ appointmentId, data }, { rejectWithValue }) => {
        try {
            const response = await appointmentApi.verifyPayment(appointmentId, data);
            return response.data;
        } catch (error) {
            return rejectWithValue(error.response?.data || 'Payment verification failed');
        }
    }
);

export const initiateKhaltiPayment = createAsyncThunk(
    'appointment/initiateKhalti',
    async (data, { rejectWithValue }) => {
        try {
            const response = await appointmentApi.initiateKhaltiPayment(data);
            return response.data;
        } catch (error) {
            return rejectWithValue(error.response?.data || 'Failed to initiate payment');
        }
    }
);

export const verifyKhaltiPayment = createAsyncThunk(
    'appointment/verifyKhalti',
    async (data, { rejectWithValue }) => {
        try {
            const response = await appointmentApi.verifyKhaltiPayment(data);
            return response.data;
        } catch (error) {
            return rejectWithValue(error.response?.data || 'Payment verification failed');
        }
    }
);

export const fetchRecommendedDoctors = createAsyncThunk(
    'appointment/fetchRecommended',
    async (params, { rejectWithValue }) => {
        try {
            const response = await appointmentApi.getRecommendedDoctors(params);
            return response.data;
        } catch (error) {
            return rejectWithValue(error.response?.data || 'Failed to fetch recommended doctors');
        }
    }
);

export const fetchDoctorById = createAsyncThunk(
    'appointment/fetchDoctorById',
    async (id, { rejectWithValue }) => {
        try {
            const response = await appointmentApi.getDoctorById(id);
            return response.data;
        } catch (error) {
            return rejectWithValue(error.response?.data || 'Failed to fetch doctor details');
        }
    }
);

export const cancelAppointment = createAsyncThunk(
    'appointment/cancel',
    async (id, { rejectWithValue }) => {
        try {
            const response = await appointmentApi.cancelAppointment(id);
            return { id, data: response.data };
        } catch (error) {
            return rejectWithValue(error.response?.data || 'Failed to cancel appointment');
        }
    }
);

const initialState = {
    appointments: [],
    availability: [],
    recommendedDoctors: [],
    loading: false,
    availabilityLoading: false,
    recommendedLoading: false,
    bookingLoading: false,
    verifyingLoading: false,
    currentDoctor: null,
    error: null,
};

const appointmentSlice = createSlice({
    name: 'appointment',
    initialState,
    reducers: {
        clearAppointmentError: (state) => {
            state.error = null;
        },
    },
    extraReducers: (builder) => {
        builder
            // Appointments
            .addCase(fetchAppointments.pending, (state) => {
                state.loading = true;
                state.error = null;
            })
            .addCase(fetchAppointments.fulfilled, (state, action) => {
                state.loading = false;
                state.appointments = action.payload;
            })
            .addCase(fetchAppointments.rejected, (state, action) => {
                state.loading = false;
                state.error = action.payload;
            })
            // Availability
            .addCase(fetchAvailability.pending, (state) => {
                state.availabilityLoading = true;
                state.error = null;
            })
            .addCase(fetchAvailability.fulfilled, (state, action) => {
                state.availabilityLoading = false;
                state.availability = action.payload;
            })
            .addCase(fetchAvailability.rejected, (state, action) => {
                state.availabilityLoading = false;
                state.error = action.payload;
            })
            // Recommended Doctors
            .addCase(fetchRecommendedDoctors.pending, (state) => {
                state.recommendedLoading = true;
            })
            .addCase(fetchRecommendedDoctors.fulfilled, (state, action) => {
                state.recommendedLoading = false;
                state.recommendedDoctors = action.payload;
            })
            .addCase(fetchRecommendedDoctors.rejected, (state) => {
                state.recommendedLoading = false;
            })
            // Fetch Doctor By Id
            .addCase(fetchDoctorById.pending, (state) => {
                state.recommendedLoading = true;
            })
            .addCase(fetchDoctorById.fulfilled, (state, action) => {
                state.recommendedLoading = false;
                state.currentDoctor = action.payload;
                // Also update in recommended list if exists
                const index = state.recommendedDoctors.findIndex(d => d.id === action.payload.id);
                if (index !== -1) {
                    state.recommendedDoctors[index] = action.payload;
                } else {
                    state.recommendedDoctors.push(action.payload);
                }
            })
            .addCase(fetchDoctorById.rejected, (state) => {
                state.recommendedLoading = false;
            })
            // Book Appointment
            .addCase(bookAppointment.pending, (state) => {
                state.bookingLoading = true;
            })
            .addCase(bookAppointment.fulfilled, (state) => {
                state.bookingLoading = false;
                // Defer success notification until payment success
            })
            .addCase(bookAppointment.rejected, (state, action) => {
                state.bookingLoading = false;
                toast.error(action.payload?.error || "Booking failed");
            })
            // Verify Payment
            .addCase(verifyPayment.pending, (state) => {
                state.verifyingLoading = true;
            })
            .addCase(verifyPayment.fulfilled, (state) => {
                state.verifyingLoading = false;
                toast.success("Payment verified! Appointment confirmed.");
            })
            .addCase(verifyPayment.rejected, (state, action) => {
                state.verifyingLoading = false;
                toast.error(action.payload?.error || "Payment verification failed");
            })
            // Initiate Khalti
            .addCase(initiateKhaltiPayment.pending, (state) => {
                state.bookingLoading = true;
            })
            .addCase(initiateKhaltiPayment.fulfilled, (state) => {
                state.bookingLoading = false;
                // No toast here as we redirect
            })
            .addCase(initiateKhaltiPayment.rejected, (state, action) => {
                state.bookingLoading = false;
                toast.error(action.payload?.error || "Failed to initiate payment");
            })
             // Verify Khalti
            .addCase(verifyKhaltiPayment.pending, (state) => {
                state.verifyingLoading = true;
            })
            .addCase(verifyKhaltiPayment.fulfilled, (state) => {
                state.verifyingLoading = false;
                toast.success("Payment verified! Appointment confirmed.");
            })
            .addCase(verifyKhaltiPayment.rejected, (state, action) => {
                state.verifyingLoading = false;
                toast.error(action.payload?.error || "Payment verification failed");
            })
            // Cancel Appointment
            .addCase(cancelAppointment.pending, (state) => {
                state.loading = true;
            })
            .addCase(cancelAppointment.fulfilled, (state, action) => {
                state.loading = false;
                const index = state.appointments.findIndex(a => a.id === action.payload.id);
                if (index !== -1) {
                    state.appointments[index].status = 'CANCELLED';
                    state.appointments[index].refund_amount = action.payload.data.refund_amount;
                }
                toast.success(action.payload.data.message || "Appointment cancelled successfully");
            })
            .addCase(cancelAppointment.rejected, (state, action) => {
                state.loading = false;
                toast.error(action.payload?.error || "Cancellation failed");
            })
            // Clear state on logout
            .addCase('auth/logout/fulfilled', (state) => {
                state.appointments = [];
                state.availability = [];
                state.currentDoctor = null;
                state.recommendedDoctors = [];
                state.error = null;
            });
    },
});

export const { clearAppointmentError } = appointmentSlice.actions;
export default appointmentSlice.reducer;
