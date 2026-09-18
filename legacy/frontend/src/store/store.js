import { configureStore } from '@reduxjs/toolkit';
import uiReducer from './slices/uiSlice';
import authReducer from './slices/authSlice';
import reportsReducer from './slices/reportsSlice';
import doctorReducer from './slices/doctorSlice';
import adminReducer from './slices/adminSlice';
import appointmentReducer from './slices/appointmentSlice';

export const store = configureStore({
    reducer: {
        ui: uiReducer,
        auth: authReducer,
        reports: reportsReducer,
        doctor: doctorReducer,
        admin: adminReducer,
        appointment: appointmentReducer,
    },
});

