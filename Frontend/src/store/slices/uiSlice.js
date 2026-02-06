import { createSlice } from '@reduxjs/toolkit';

const initialState = {
    isDarkMode: false,
    isLoading: false,
};

const uiSlice = createSlice({
    name: 'ui',
    initialState,
    reducers: {
        toggleDarkMode: (state) => {
            state.isDarkMode = !state.isDarkMode;
        },
        setLoading: (state, action) => {
            state.isLoading = action.payload;
        },
    },
});

export const { toggleDarkMode, setLoading } = uiSlice.actions;
export default uiSlice.reducer;
