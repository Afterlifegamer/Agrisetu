import { cart, addtoCart } from '../data/cart.js';
import { products } from '../data/products.js';
import { formatCurrency } from './utils/money.js';

function renderProducts(productsToRender) {
  let productsHTML = '';

  if (productsToRender.length === 0) {
    productsHTML = `<p class="col-span-full text-center text-text-secondary-light dark:text-text-secondary-dark py-8 px-4 w-full">No products found matching your criteria.</p>`;
  } else {
    productsToRender.forEach((product) => {
      productsHTML += `<div class="flex flex-col gap-3 rounded-xl bg-surface-light dark:bg-surface-dark overflow-hidden border border-border-light dark:border-border-dark cursor-pointer transition-shadow hover:shadow-lg">
  <div class="w-full aspect-square">
    <img src="${product.image}"  class="w-full h-full object-cover" loading="lazy" />
  </div>
  <div class="p-3 pt-0">
  <p class="text-base font-semibold leading-normal text-text-light dark:text-text-dark">${product.name}</p>
  <p class="text-base font-bold leading-normal text-primary mt-1">₹${formatCurrency(product.price)}/kg</p>
  <p class="text-xs font-normal leading-normal text-muted-light dark:text-muted-dark">Min. 10kg</p>
  </div>
  <button class="flex h-10 min-w-[84px] cursor-pointer items-center justify-center overflow-hidden rounded-lg bg-primary px-4 text-sm font-bold leading-normal text-white js-add-to-cart" data-product-id="${product.id}">
  <span class="truncate">add to cart</span>
  </button>
  </div>`;
    });
  }

  document.querySelector('.js-products-grid').innerHTML = productsHTML;

  document.querySelectorAll('.js-add-to-cart').forEach((button) => {
    button.addEventListener('click', () => {
      const productId = button.dataset.productId;
      addtoCart(productId);
      updateCartQuantity();
    });
  });
}

function updateCartQuantity() {
  let cartTotal = 0;

  cart.forEach((cartItem) => {
    cartTotal += cartItem.quantity;
  });

  const cartQuantityElement = document.querySelector('.js-cart-quantity');
  if (cartQuantityElement) {
    cartQuantityElement.innerHTML = cartTotal;
  }
}

function applyFilters() {
  const searchQuery = document.querySelector('.js-search-bar')?.value.toLowerCase() || '';
  const categoryFilter = document.querySelector('[data-filter-type="category"]')?.dataset.selectedValue || '';
  const priceFilter = document.querySelector('[data-filter-type="price"]')?.dataset.selectedValue || '';
  const sortFilter = document.querySelector('[data-filter-type="sort"]')?.dataset.selectedValue || 'popular';

  let filteredProducts = products.filter(product => {
    // Check Search
    let matchesSearch = true;
    if (searchQuery) {
      matchesSearch = product.name.toLowerCase().includes(searchQuery) ||
        (product.keywords && product.keywords.some(k => k.toLowerCase().includes(searchQuery)));
    }

    // Check Category
    let matchesCategory = categoryFilter === '' || product.category === categoryFilter;

    // Check Price
    let matchesPrice = true;
    if (priceFilter === 'under5000') {
      matchesPrice = product.price < 5000;
    } else if (priceFilter === '5000to10000') {
      matchesPrice = product.price >= 5000 && product.price <= 10000;
    } else if (priceFilter === 'above10000') {
      matchesPrice = product.price > 10000;
    }

    return matchesSearch && matchesCategory && matchesPrice;
  });

  // Check Sort
  if (sortFilter === 'priceLowHigh') {
    filteredProducts.sort((a, b) => a.price - b.price);
  } else if (sortFilter === 'priceHighLow') {
    filteredProducts.sort((a, b) => b.price - a.price);
  } else if (sortFilter === 'nameAZ') {
    filteredProducts.sort((a, b) => a.name.localeCompare(b.name));
  }
  // If popular, keep default order (which is how they are listed in products.js)

  renderProducts(filteredProducts);
}

// Attach event listeners to search bar
document.querySelector('.js-search-bar')?.addEventListener('input', applyFilters);

// Dropdown Logic
document.querySelectorAll('.js-dropdown').forEach(dropdown => {
  const button = dropdown.querySelector('.js-dropdown-button');
  const menu = dropdown.querySelector('.js-dropdown-menu');
  const icon = dropdown.querySelector('.js-dropdown-icon');
  const items = dropdown.querySelectorAll('.js-dropdown-item');
  const label = dropdown.querySelector('.js-dropdown-label');

  // Toggle menu
  button.addEventListener('click', (e) => {
    e.stopPropagation();
    // Close other dropdowns
    document.querySelectorAll('.js-dropdown-menu').forEach(m => {
      if (m !== menu) m.classList.add('hidden');
    });
    document.querySelectorAll('.js-dropdown-icon').forEach(i => {
      if (i !== icon) i.classList.remove('rotate-180');
    });

    menu.classList.toggle('hidden');
    icon.classList.toggle('rotate-180');
  });

  // Handle item click
  items.forEach(item => {
    item.addEventListener('click', () => {
      const value = item.dataset.value;
      const labelText = item.dataset.label;

      // Update dropdown state
      dropdown.dataset.selectedValue = value;
      label.textContent = labelText;

      // Close menu
      menu.classList.add('hidden');
      icon.classList.remove('rotate-180');

      // Trigger filter update
      applyFilters();
    });
  });
});

// Close dropdowns when clicking outside
document.addEventListener('click', (e) => {
  if (!e.target.closest('.js-dropdown')) {
    document.querySelectorAll('.js-dropdown-menu').forEach(menu => menu.classList.add('hidden'));
    document.querySelectorAll('.js-dropdown-icon').forEach(icon => icon.classList.remove('rotate-180'));
  }
});

updateCartQuantity();
renderProducts(products); // Initial render with all products



