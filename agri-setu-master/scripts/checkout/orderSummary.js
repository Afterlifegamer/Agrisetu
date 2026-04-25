import { cart, removeFromCart, addtoCart, removeOneItem } from '../../data/cart.js';
import { products } from '../../data/products.js';
import { formatCurrency } from '../utils/money.js';
import { renderPaymentSummary } from './paymentSummary.js';


export function renderOrderSummary() {
  let cartSummaryHTML = '';

  cart.forEach((cartItem) => {
    const productId = cartItem.productId;

    let matchingProduct;

    products.forEach((product) => {
      if (product.id === productId) {
        matchingProduct = product;
      }
    });

    if (!matchingProduct) {
      return; // Skip this cart item if no matching product found
    }

    cartSummaryHTML += `
                <div
                  class="p-6 flex flex-col md:flex-row gap-6 items-start md:items-center js-cart-item-container-${matchingProduct.id} "
                >
                  <img
                    alt="Organic Tomatoes"
                    class="h-24 w-24 rounded-xl object-cover border border-slate-100"
                    src="${matchingProduct.image}"
                  />
                  <div class="flex-1 min-w-0">
                    <h4
                      class="font-bold text-lg text-slate-900 dark:text-white"
                    >
                      ${matchingProduct.name}
                    </h4>
                    <p
                      class="text-xs font-medium text-slate-500 uppercase tracking-wide mt-1"
                    >
                      Bulk Packaging: 50kg Bag
                    </p>
                    <div class="mt-2 text-primary font-bold flex items-center gap-4">
                      <span>₹${formatCurrency(matchingProduct.price)}<span class="text-xs text-slate-400 font-normal">/ kg</span></span>
                      
                      <div class="flex items-center gap-2">
                        <label class="text-sm text-slate-500">Qty:</label>
                        <div class="flex items-center gap-1 rounded-lg border border-slate-200 dark:border-slate-700 p-1">
                          <input
                            class="w-16 border-none bg-transparent text-center font-bold text-sm focus:ring-0 js-quantity-input"
                            type="number"
                            min="1"
                            data-product-id="${matchingProduct.id}"
                            value="${cartItem.quantity}"
                          />
                          <span class="px-2 text-xs font-bold text-slate-400">kg</span>
                        </div>
                      </div>
                    </div>
                  </div>
                  <div class="flex items-center gap-4">
                    <div class="text-right min-w-[120px]">
                      <p class="text-[10px] text-slate-400 uppercase font-bold">
                        Subtotal
                      </p>
                      <p
                        class="text-lg font-bold text-slate-900 dark:text-white"
                      >
                        ₹${formatCurrency(matchingProduct.price * cartItem.quantity)}
                      </p>
                    </div>
                    <button
                      class="p-2 text-slate-300 hover:text-red-500 transition-colors"
                    >
                      <span class="material-symbols-outlined js-delete-button" data-product-id = ${matchingProduct.id}  >delete</span>
                    </button>
                  </div>
                </div>`;
  });

  document.querySelector('.js-order-summary').innerHTML = cartSummaryHTML;


  document.querySelectorAll('.js-delete-button')
    .forEach((button) => {
      button.addEventListener('click', () => {
        const productId = button.dataset.productId;
        removeFromCart(productId);

        const container = document.querySelector(`.js-cart-item-container-${productId}`);
        if (container) container.remove();

        // re-calculate totals after deleting an item
        renderPaymentSummary();

      });
    });

  document.querySelectorAll('.js-quantity-input')
    .forEach((input) => {
      input.addEventListener('change', (e) => {
        const productId = input.dataset.productId;
        const newQuantity = Number(e.target.value);

        if (newQuantity > 0) {
          // Update cart item quantity directly
          let matchingItem;
          cart.forEach((cartItem) => {
            if (cartItem.productId === productId) {
              matchingItem = cartItem;
            }
          });

          if (matchingItem) {
            matchingItem.quantity = newQuantity;
            // Ideally we'd call saveToStorage() here if exported from cart.js
            localStorage.setItem('cart', JSON.stringify(cart));

            // Re-render
            renderOrderSummary();
            renderPaymentSummary();
          }
        } else {
          // Reset to 1 if user tries to enter 0 or negative
          e.target.value = 1;
        }
      });
    });

};
renderOrderSummary();
// also render the payment summary whenever the order summary is generated or updated
renderPaymentSummary();




