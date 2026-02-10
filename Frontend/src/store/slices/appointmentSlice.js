import { createSlice, createAsyncThunk } from '@reduxjs/toolkit';
import { appointmentApi } from '../../services/api';
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

export const fetchRecommendedDoctors = createAsyncThunk(
    'appointment/fetchRecommended',
    async (specialization, { rejectWithValue }) => {
        try {
            const response = await appointmentApi.getRecommendedDoctors(specialization);
            return response.data;
        } catch (error) {
            return rejectWithValue(error.response?.data || 'Failed to fetch recommended doctors');
        }
    }
);

const initialState = {
    appointments: [],
    availability: [],
    recommendedDoctors: [],
    loading: false,
    recommendedLoading: false,
    bookingLoading: false,
    verifyingLoading: false,
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
            // Book Appointment
            .addCase(bookAppointment.pending, (state) => {
                state.bookingLoading = true;
            })
            .addCase(bookAppointment.fulfilled, (state) => {
                state.bookingLoading = false;
                toast.success("Appointment booked! Please complete payment.");
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
            });
    },
});

export const { clearAppointmentError } = appointmentSlice.actions;
export default appointmentSlice.reducer;
