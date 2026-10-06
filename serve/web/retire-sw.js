// The old llama UI cached its index. Retire its worker without deleting chat databases or navigating clients.
self.addEventListener("install", event => event.waitUntil(self.skipWaiting()));
self.addEventListener("activate", event => event.waitUntil((async () => {
  await self.clients.claim();
  await self.registration.unregister();
})()));
