/**
 * search_common.js
 * Common search logic for the AgriSetu platform.
 */

export function setupSearch(inputSelector, callback) {
    const searchInputs = document.querySelectorAll(inputSelector);
    
    searchInputs.forEach(input => {
        // Handle 'Enter' key
        input.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                const query = input.value.trim();
                handleSearch(query, callback);
            }
        });

        // Handle search icon click if applicable
        const parent = input.parentElement;
        const icon = parent.querySelector('.material-symbols-outlined');
        if (icon && icon.textContent === 'search') {
            icon.style.cursor = 'pointer';
            icon.addEventListener('click', () => {
                const query = input.value.trim();
                handleSearch(query, callback);
            });
        }
    });
}

function handleSearch(query, callback) {
    if (!query) return;

    // If a local callback is provided (e.g., for marketplace filtering), use it
    if (callback) {
        callback(query);
    } else {
        // Otherwise, redirect to marketplace with the search query
        window.location.href = `marketplace.html?search=${encodeURIComponent(query)}`;
    }
}
