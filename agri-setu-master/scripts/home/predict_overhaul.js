/**
 * predict_overhaul.js
 * Logic for the 'Predict' form page.
 * Handles form submission and results storage.
 */

const API_BASE = window.location.origin;
import { setupSearch } from './search_common.js';

let selectedSoil = 'Loamy';

function setupSoilButtons() {
    const buttons = document.querySelectorAll('.soil-btn');
    buttons.forEach(btn => {
        btn.addEventListener('click', () => {
            // Remove active classes from all
            buttons.forEach(b => {
                b.classList.remove('bg-primary', 'text-white');
                b.classList.add('bg-surface-container-highest', 'text-on-surface-variant');
            });
            // Add active classes to selected
            btn.classList.add('bg-primary', 'text-white');
            btn.classList.remove('bg-surface-container-highest', 'text-on-surface-variant');
            selectedSoil = btn.getAttribute('data-value');
        });
    });
}

async function handlePredict() {
    const district = document.getElementById('location').value.trim();
    const budget = parseFloat(document.getElementById('budget').value);
    const duration = parseFloat(document.getElementById('duration').value);
    const predictBtn = document.getElementById('predict-btn');

    if (!district || isNaN(budget) || isNaN(duration)) {
        alert('Please fill all fields correctly.');
        return;
    }

    predictBtn.disabled = true;
    predictBtn.textContent = 'Analyzing...';

    try {
        const res = await fetch(`${API_BASE}/api/recommend`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                district: district,
                budget: budget,
                duration_months: duration,
                soil_type: selectedSoil
            })
        });

        const data = await res.json();

        if (data.success) {
            // Store results in session storage for the details page
            sessionStorage.setItem('agrisetu_recommendation_results', JSON.stringify(data));
            // Redirect to results page
            window.location.href = 'crop_recommend.html';
        } else {
            alert('Error: ' + (data.error || 'Could not fetch recommendations'));
            predictBtn.disabled = false;
            predictBtn.textContent = 'Analyze ROI Potential';
        }
    } catch (err) {
        console.error('Prediction error:', err);
        alert('Could not reach the server. Please check your connection.');
        predictBtn.disabled = false;
        predictBtn.textContent = 'Analyze ROI Potential';
    }
}

document.addEventListener('DOMContentLoaded', () => {
    setupSoilButtons();
    document.getElementById('predict-btn').addEventListener('click', handlePredict);
    setupSearch('input[placeholder*="Search"]');
});
