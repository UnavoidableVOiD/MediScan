import { configureStore } from '@reduxjs/toolkit';
import uiReducer from './slices/uiSlice';
import authReducer from './slices/authSlice';
import reportsReducer from './slices/reportsSlice';

export const store = configureStore({
    reducer: {
        ui: uiReducer,
        auth: authReducer,
        reports: reportsReducer,
    },
});

