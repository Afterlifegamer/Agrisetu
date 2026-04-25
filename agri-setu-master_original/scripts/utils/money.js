export function formatCurrency(priceCents)
{
  return (Math.round(priceCents/100*100)/100).toFixed(2);
}
