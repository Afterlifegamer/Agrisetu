import {cart} from "../../data/cart.js"; 
import {getProduct} from "../../data/products.js";
import {formatCurrency} from "../utils/money.js"

export function renderPaymentSummary()
{
    let productPriceCents = 0;
    let deliveryPrice = 0;
    let productCount = 0;
   

     cart.forEach((cartItem) =>
    {
        // increment by the quantity of this cart item, not just the number of entries
        productCount += cartItem.quantity;
        const product = getProduct(cartItem.productId);
        // products.json stores price in rupees; convert to cents for calculations
        productPriceCents += (product.price ) * cartItem.quantity;
    });

    // delivery fee ₹120 if more than 5 items (by quantity), else ₹0
    if (productCount > 5) {
        deliveryPrice = 120 * 100; // cents
    } else {
        deliveryPrice = 0;
    }
    
    const totalBeforeTaxCents = productPriceCents + deliveryPrice;
    const taxCents = totalBeforeTaxCents * 0.18; // Assuming 18% tax
    const grandTotal = totalBeforeTaxCents + taxCents;

    const paymentSummaryHTML =`<div class="sticky top-28 space-y-6">
              <div
                class="bg-white dark:bg-slate-900 rounded-2xl border border-slate-200 dark:border-slate-800 p-6 shadow-lg"
              >
                <h3
                  class="text-lg font-bold text-slate-900 dark:text-white mb-6 border-b border-slate-100 dark:border-slate-800 pb-4"
                >
                  Order Summary
                </h3>
                <div class="space-y-4">
                  <div class="flex justify-between text-slate-500">
                    <span>Items Subtotal</span>
                    <span class="font-bold text-slate-900 dark:text-white"
                      >₹${formatCurrency(productPriceCents)}</span
                    >
                  </div>
                  <div class="flex justify-between text-slate-500">
                    <span>GST (18%)</span>
                    <span class="font-bold text-slate-900 dark:text-white"
                      >₹${formatCurrency(taxCents)}</span
                    >
                  </div>
                  <div class="flex justify-between text-slate-500">
                    <span class="flex items-center gap-1"
                      >Delivery Fees
                      <span class="material-symbols-outlined text-xs"
                        >info</span
                      ></span
                    >
                    <span class="font-bold text-slate-900 dark:text-white"
                      >₹${formatCurrency(deliveryPrice)}</span
                    >
                  </div>
                  <div class="flex justify-between text-slate-500">
                    <span>Platform Fee</span>
                    <span class="font-bold text-primary">FREE</span>
                  </div>
                  <div
                    class="pt-4 mt-2 border-t border-slate-100 dark:border-slate-800"
                  >
                    <div class="flex justify-between items-end">
                      <div>
                        <p
                          class="text-[10px] font-bold text-slate-400 uppercase tracking-widest"
                        >
                          Grand Total
                        </p>
                        <p
                          class="text-3xl font-black text-slate-900 dark:text-white"
                        >
                          ₹${formatCurrency(grandTotal)}
                        </p>
                      </div>
                      <div class="text-right">
                        <div
                          class="flex items-center justify-end gap-1 text-[10px] font-bold text-slate-400 uppercase"
                        >
                          <span
                            class="material-symbols-outlined text-xs text-primary"
                            >verified</span
                          >
                          Secure
                        </div>
                        <p class="text-[9px] text-slate-400">Inc. all taxes</p>
                      </div>
                    </div>
                  </div>
                  <button
                    class="w-full mt-6 bg-primary text-white py-4 rounded-xl font-black text-lg shadow-lg shadow-primary/30 hover:brightness-110 active:scale-[0.98] transition-all flex items-center justify-center gap-2"
                  >
                    Place Order &amp; Pay
                    <span class="material-symbols-outlined">arrow_forward</span>
                  </button>
                </div>
              </div>
              <div class="flex flex-col items-center gap-2 text-slate-400">
                <div class="flex items-center gap-4 opacity-50">
                  <img
                    alt="PayPal"
                    class="h-6 grayscale"
                    src="https://upload.wikimedia.org/wikipedia/commons/b/b5/PayPal.svg"
                  />
                
                  <img
                    alt="Mastercard"
                    class="h-6 grayscale"
                    src="https://upload.wikimedia.org/wikipedia/commons/2/2a/Mastercard-logo.svg"
                  />
                </div>
                <p class="text-[10px] font-medium tracking-tight">
                  Trusted by 50,000+ Agri-businesses across India
                </p>
              </div>
            </div>
      `;
    document.querySelector('.js-payment-summary').innerHTML = paymentSummaryHTML;
};