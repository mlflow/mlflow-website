type SiteAction = {
  id: string;
  name: string;
  value?: number;
  currency?: string;
};

function createActionId(id: string): string {
  return `${id}_${Date.now()}_${Math.random().toString(36).slice(2, 9)}`;
}

function emitAction({
  id,
  name,
  value = 1,
  currency = "USD",
}: SiteAction): void {
  if (typeof window === "undefined") {
    return;
  }

  const payload = {
    transaction_id: createActionId(id),
    value,
    currency,
    items: [
      {
        item_id: id,
        item_name: name,
        quantity: 1,
      },
    ],
  };

  window.dataLayer = window.dataLayer || [];
  window.dataLayer.push({ ecommerce: null });
  window.dataLayer.push({
    event: "purchase",
    ecommerce: payload,
  });

  if (typeof window.gtag === "function") {
    window.gtag("event", "purchase", payload);
  }
}

export function reportTryDemo(): void {
  emitAction({
    id: "try_demo",
    name: "Try Demo",
  });
}
