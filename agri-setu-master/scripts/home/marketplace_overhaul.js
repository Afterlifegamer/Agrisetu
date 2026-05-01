/**
 * marketplace_overhaul.js
 * Logic for the 'Premium Harvest' marketplace.
 * Dynamically renders crops from the original catalog.
 */

import { cart, addtoCart } from '../../data/cart.js';
import { products } from '../../data/products.js';
import { setupSearch } from './search_common.js';

function renderMarketplace(filterQuery = '') {
    const grid = document.getElementById('products-grid');
    if (!grid) return;

    grid.innerHTML = '';

    const filteredProducts = products.filter(p => 
        p.name.toLowerCase().includes(filterQuery.toLowerCase()) ||
        p.category.toLowerCase().includes(filterQuery.toLowerCase()) ||
        p.keywords.some(k => k.toLowerCase().includes(filterQuery.toLowerCase()))
    );

    if (filteredProducts.length === 0) {
        grid.innerHTML = `<div class="col-span-full py-20 text-center">
            <span class="material-symbols-outlined text-stone-300 text-6xl mb-4">search_off</span>
            <p class="text-stone-500 font-medium">No results found for "${filterQuery}"</p>
        </div>`;
        return;
    }

    filteredProducts.forEach(product => {
        const article = document.createElement('article');
        article.className = 'group bg-surface-container-lowest rounded-xl overflow-hidden flex flex-col premium-shadow transition-transform hover:-translate-y-2 duration-300';
        
        // Premium-style card HTML
        article.innerHTML = `
            <div class="relative h-80 overflow-hidden">
                <img alt="${product.name}" class="w-full h-full object-cover transition-transform duration-700 group-hover:scale-110" 
                     src="${product.image}"
                     onerror="this.src='https://images.unsplash.com/photo-1523348837708-15d4a09cfac2?auto=format&fit=crop&q=80&w=800'">
                <div class="absolute top-4 right-4 bg-white/90 backdrop-blur-md px-3 py-1 rounded-full text-[10px] font-black text-tertiary uppercase tracking-widest">
                    ${product.category.split('&')[0].trim()}
                </div>
            </div>
            <div class="p-8 flex flex-col flex-grow">
                <div class="flex justify-between items-start mb-4">
                    <div>
                        <h3 class="text-2xl font-headline font-bold text-primary mb-1">${product.name}</h3>
                        <p class="text-xs font-medium text-stone-500">Premium Grade • Kerala Harvest</p>
                    </div>
                    <div class="text-right">
                        <span class="text-xl font-headline font-black text-tertiary">Rs. ${product.price}</span>
                        <span class="block text-[10px] text-stone-400 uppercase font-bold">per kg</span>
                    </div>
                </div>
                <p class="text-sm text-on-surface-variant leading-relaxed mb-8 flex-grow">
                    High-quality ${product.name.toLowerCase()} sourced directly from sustainable farms. 
                    ${product.keywords.slice(0, 2).join(', ')} certified.
                </p>
                <div class="flex items-center justify-between mt-auto gap-2">
                    <div class="flex items-center gap-2 bg-surface-container-highest rounded-md px-2 py-1 border border-outline-variant/20">
                        <span class="text-[10px] font-bold text-stone-500 uppercase">Qty (kg)</span>
                        <input type="number" min="10" step="1" value="10" class="w-14 h-6 text-xs font-bold border-none bg-transparent focus:ring-0 text-center js-quantity-selector" data-id="${product.id}">
                    </div>
                    <button class="bg-primary text-white p-3 rounded-md hover:bg-primary-container transition-all flex items-center gap-2 group-hover:px-6 duration-300 js-add-to-cart" data-id="${product.id}">
                        <span class="text-xs font-bold hidden group-hover:block transition-all whitespace-nowrap">Add to Cart</span>
                        <span class="material-symbols-outlined text-sm">shopping_cart</span>
                    </button>
                </div>
            </div>
        `;
        grid.appendChild(article);
    });

    attachCartListeners();
}

function attachCartListeners() {
    document.querySelectorAll('.js-add-to-cart').forEach(button => {
        button.addEventListener('click', (e) => {
            e.preventDefault();
            const productId = button.getAttribute('data-id');
            const qtyInput = document.querySelector(`.js-quantity-selector[data-id="${productId}"]`);
            const quantity = qtyInput && !isNaN(parseInt(qtyInput.value)) ? parseInt(qtyInput.value) : 10;
            addtoCart(productId, quantity);
            updateCartUI();
        });
    });
}

function updateCartUI() {
    let cartTotal = 0;
    cart.forEach(item => { cartTotal += item.quantity; });
    const cartQty = document.querySelector('.js-cart-quantity');
    if (cartQty) cartQty.textContent = cartTotal;
}

document.addEventListener('DOMContentLoaded', () => {
    updateCartUI();
    
    // Check for search query in URL
    const urlParams = new URLSearchParams(window.location.search);
    const initialSearch = urlParams.get('search') || '';
    
    renderMarketplace(initialSearch);

    // Setup search listener
    setupSearch('input[placeholder*="Search"]', (query) => {
        renderMarketplace(query);
    });
});
