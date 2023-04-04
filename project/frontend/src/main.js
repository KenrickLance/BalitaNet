import { createApp } from 'vue'
import App from './App.vue'
import router from './router'
import store from './store'

import * as Sentry from "@sentry/vue";
import { BrowserTracing } from "@sentry/tracing";

import 'bootstrap'
import 'bootstrap/dist/css/bootstrap.min.css'

const app = createApp(App)

Sentry.init({
    app,
    dsn: "https://961508e4ee5d4f3b9afa0e3257f44b31@o1128174.ingest.sentry.io/4504486691274752",
    integrations: [
      new BrowserTracing({
        routingInstrumentation: Sentry.vueRouterInstrumentation(router),
        tracePropagationTargets: ["localhost", "my-site-url.com", /^\//],
      }),
    ],
    // Set tracesSampleRate to 1.0 to capture 100%
    // of transactions for performance monitoring.
    // We recommend adjusting this value in production
    tracesSampleRate: 1.0,
  });

app.use(store).use(router).mount('#app')
