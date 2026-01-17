export const parseError = (errorData) => {
    if (!errorData) return "An unexpected error occurred";

    if (typeof errorData === 'string') return errorData;

    if (typeof errorData === 'object') {
        // Handle "detail" key
        if (errorData.detail && typeof errorData.detail === 'string') {
            return errorData.detail;
        }

        // Handle field errors and non_field_errors
        const messages = [];
        Object.keys(errorData).forEach((key) => {
            const val = errorData[key];
            if (Array.isArray(val)) {
                messages.push(...val);
            } else if (typeof val === 'string') {
                messages.push(val);
            } else if (typeof val === 'object' && val !== null) {
                // Nested errors or something unusual
                messages.push(JSON.stringify(val));
            }
        });

        if (messages.length > 0) {
            return messages.join('\n');
        }
    }

    return JSON.stringify(errorData);
};
