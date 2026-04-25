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
                    <div class="mt-2 text-primary font-bold">
                      ₹${formatCurrency(matchingProduct.price)}
                      <span class="text-xs text-slate-400 font-normal"
                        >/ kg</span
                      >
                    </div>
                  </div>
                  <div class="flex items-center gap-4">
                    <div
                      class="flex items-center gap-1 rounded-lg border border-slate-200 dark:border-slate-700 p-1"
                    >
                      <button
                        class="flex h-8 w-8 items-center justify-center rounded-md bg-slate-100 hover:bg-slate-200 text-slate-600 transition-colors js-remove-button"data-product-id = ${matchingProduct.id}
                      >
                        <span class="material-symbols-outlined text-sm"
                          >remove</span
                        >
                      </button>
                      <input
                        class="w-12 border-none bg-transparent text-center font-bold text-sm focus:ring-0"
                        type="number"
                        value="${cartItem.quantity}"
                      />
                      <button
                        class="flex h-8 w-8 items-center justify-center rounded-md bg-primary/10 hover:bg-primary/20 text-primary transition-colors js-add-button" data-product-id = ${matchingProduct.id}
                      >
                        <span class="material-symbols-outlined text-sm"
                          >add</span
                        >
                      </button>
                      <span class="px-2 text-xs font-bold text-slate-400"
                        >kg</span
                      >
                    </div>
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
        container.remove();

        // re-calculate totals after deleting an item
        renderPaymentSummary();

      });
    });

  document.querySelectorAll('.js-remove-button')
    .forEach((button) => {
      button.addEventListener('click', () => {
        const productId = button.dataset.productId;
        removeOneItem(productId);


        renderOrderSummary();
        // re-calculate totals after deleting an item
        renderPaymentSummary();

      });
    });


  document.querySelectorAll('.js-add-button')
    .forEach((button) => {
      button.addEventListener('click', () => {
        const productId = button.dataset.productId;
        addtoCart(productId);

        renderOrderSummary();
        // re-calculate totals after deleting an item
        renderPaymentSummary();

      });
    });

};
renderOrderSummary();
// also render the payment summary whenever the order summary is generated or updated
renderPaymentSummary();




