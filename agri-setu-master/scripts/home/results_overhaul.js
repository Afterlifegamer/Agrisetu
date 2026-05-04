/**
 * results_overhaul.js
 * Logic for the 'Detailed Recommendation' (Results) page.
 * Renders data from sessionStorage and initializes charts.
 */

function renderResults() {
    const rawData = sessionStorage.getItem('agrisetu_recommendation_results');
    if (!rawData) {
        console.error('No recommendation data found in session storage.');
        // Optional: Redirect back to predict
        // window.location.href = 'predict.html';
        return;
    }

    const data = JSON.parse(rawData);
    if (!data.success || !data.recommendations || data.recommendations.length === 0) {
        return;
    }

    const top = data.recommendations[0];

    // -- Hero Section --
    document.getElementById('crop-name-main').innerHTML = top.crop_name;
    document.getElementById('crop-desc-main').textContent = top.recommendation_text || `Optimized for your soil profile and climate conditions. High ROI potential based on 10-year yield averages.`;
    
    const heroImg = document.getElementById('hero-crop-image');
    if (heroImg) {
        // Map 'Rice' to 'Paddy' image for farmers
        const imgName = top.crop_name.toLowerCase() === 'rice' ? 'paddy' : top.crop_name.toLowerCase();
        heroImg.src = `images/${imgName}.jpg`;
        heroImg.onerror = () => { heroImg.src = 'https://images.unsplash.com/photo-1500382017468-9049fed747ef?auto=format&fit=crop&q=80&w=2000'; };
    }

    // -- Metrics --
    document.getElementById('roi-value').textContent = `+${top.est_roi}%`;
    document.getElementById('market-price').innerHTML = `Rs. ${top.predicted_price.toLocaleString('en-IN')}<span class="text-2xl">/unit</span>`;
    document.getElementById('duration-value').innerHTML = `${top.approx_months * 30} <span class="text-2xl">Days</span>`;

    // Financial Breakdown
    const totalCost = top.annual_cost;
    document.getElementById('expense-seeds').textContent = `Rs. ${(totalCost * 0.15).toLocaleString('en-IN')}`;
    document.getElementById('expense-labor').textContent = `Rs. ${(totalCost * 0.45).toLocaleString('en-IN')}`;
    document.getElementById('expense-soil').textContent = `Rs. ${(totalCost * 0.25).toLocaleString('en-IN')}`;
    document.getElementById('expense-irrigation').textContent = `Rs. ${(totalCost * 0.15).toLocaleString('en-IN')}`;

    // -- Intercrop --
    const intercropSuggestion = top.companion_crops.length > 0 ? top.companion_crops.join(' & ') : "Azolla (Bio-fertilizer)";
    document.getElementById('intercrop-text').innerHTML = `Maximize your yield by planting <span class="text-primary-fixed font-bold">${intercropSuggestion}</span> alongside your ${top.crop_name}. This practice stabilizes soil nutrients and naturally suppresses weed growth.`;

    // -- Chart --
    renderChart(top);

    // -- Alternatives --
    renderAlternatives(data.recommendations.slice(1));
}

function renderChart(top) {
    const ctx = document.getElementById('yieldChart').getContext('2d');
    
    // Aesthetic colors from the brand palette
    const primaryGreen = '#173B1E';
    const secondaryBrown = '#75584D';
    const accentOrange = '#641800';

    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['Total Revenue', 'Net Profit', 'Total Expenses'],
            datasets: [{
                label: 'Financial Projection (INR)',
                data: [top.revenue, top.profit_cycle, top.annual_cost],
                backgroundColor: [primaryGreen, accentOrange, secondaryBrown],
                borderRadius: 12,
                borderSkipped: false,
                barThickness: 60
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: '#fafaf5',
                    titleColor: '#1a1c19',
                    bodyColor: '#1a1c19',
                    borderColor: '#e3e3de',
                    borderWidth: 1,
                    padding: 12,
                    displayColors: false,
                    callbacks: {
                        label: (context) => ` Rs. ${context.parsed.y.toLocaleString('en-IN')}`
                    }
                }
            },
            scales: {
                x: {
                    grid: { display: false },
                    ticks: { font: { family: 'Manrope', weight: 'bold' } }
                },
                y: {
                    beginAtZero: true,
                    grid: { color: '#e3e3de', drawBorder: false },
                    ticks: {
                        callback: (value) => '₹' + value / 1000 + 'k',
                        font: { family: 'Work Sans' }
                    }
                }
            }
        }
    });
}

function renderAlternatives(alternatives) {
    const container = document.getElementById('alternatives-container');
    container.innerHTML = '';

    alternatives.forEach(alt => {
        const card = document.createElement('div');
        card.className = 'bg-surface-container-lowest p-6 rounded-xl group cursor-pointer hover:shadow-xl transition-all';
        
        const imgName = alt.crop_name.toLowerCase() === 'rice' ? 'paddy' : alt.crop_name.toLowerCase();
        card.innerHTML = `
            <div class="h-40 w-full rounded-lg overflow-hidden mb-6">
                <img alt="${alt.crop_name}" class="w-full h-full object-cover group-hover:scale-110 transition-transform duration-500" 
                     src="images/${imgName}.jpg"
                     onerror="this.src='https://images.unsplash.com/photo-1523348837708-15d4a09cfac2?auto=format&fit=crop&q=80&w=800'">
            </div>
            <div class="flex justify-between items-start mb-4">
                <div>
                    <h4 class="font-bold text-xl text-primary">${alt.crop_name}</h4>
                    <span class="text-sm text-on-surface-variant">Recommended Alternative</span>
                </div>
                <span class="text-emerald-600 font-bold">+${alt.est_roi.toFixed(0)}% ROI</span>
            </div>
            <p class="text-sm text-on-surface-variant mb-6">${alt.recommendation_text || 'Resilient secondary option for your soil rotation cycle.'}</p>
            <button class="text-primary font-bold text-sm flex items-center gap-2 group-hover:translate-x-1 transition-transform">
                View Forecast
            </button>
        `;
        container.appendChild(card);
    });
}

document.addEventListener('DOMContentLoaded', renderResults);
