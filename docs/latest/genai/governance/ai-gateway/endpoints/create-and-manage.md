# Create and Manage Endpoints

Endpoints define how requests are routed to AI models. Each endpoint can use a single model or leverage advanced routing features like traffic splitting and fallbacks.

## Accessing Endpoints[​](#accessing-endpoints "Direct link to Accessing Endpoints")

Navigate to the AI Gateway section at `http://localhost:5000/#/gateway`. The Endpoints tab shows all your configured endpoints.

![Gateway Overview](/mlflow-website/docs/latest/assets/images/gateway-overview-2bbf3cf41c12806962300767a53094c0.png)

## Creating an Endpoint[​](#creating-an-endpoint "Direct link to Creating an Endpoint")

### Basic Setup[​](#basic-setup "Direct link to Basic Setup")

1. Click **Create Endpoint**

2. Enter a unique endpoint name (e.g., `my-chat-endpoint`)
   <!-- -->
   * This name becomes part of your API path: `/gateway/my-chat-endpoint/...`

3. Select your provider from 100+ supported options

   <!-- -->

   * Common providers (OpenAI, Anthropic, Google Gemini) appear first
   * Click "View all providers" for the full LiteLLM catalog

4. Choose your model

   <!-- -->

   * The selector displays capability badges (Tools, Reasoning, Caching)
   * Context window size and token costs are shown
   * Use the search function for quick filtering

5. Configure API key:

   <!-- -->

   * **Create new API key**: Configure credentials inline (convenient for first-time setup)
   * **Use existing API key**: Select from previously created keys (recommended for consistency)

6. Review your configuration in the summary panel

7. Click **Create Endpoint**

![Create Endpoint](/mlflow-website/docs/latest/assets/images/create-endpoint-4b5244302954cd8e676b9f929168a048.png)

### Advanced Routing Options[​](#advanced-routing-options "Direct link to Advanced Routing Options")

For endpoints that need traffic splitting or fallbacks, see [Traffic Routing & Fallbacks](/mlflow-website/docs/latest/genai/governance/ai-gateway/traffic-routing-fallbacks.md).

## Managing Existing Endpoints[​](#managing-existing-endpoints "Direct link to Managing Existing Endpoints")

### Viewing Endpoint Details[​](#viewing-endpoint-details "Direct link to Viewing Endpoint Details")

Click on any endpoint name to view its configuration:

* **Provider and model**: The currently configured model
* **API key**: Which credentials are being used
* **Traffic split**: Percentage distribution across models (if configured)
* **Fallbacks**: Ordered list of fallback models (if configured)

### Editing Endpoints[​](#editing-endpoints "Direct link to Editing Endpoints")

To modify an endpoint:

1. Click on the endpoint name to open details

2. Update the configuration as needed:

   <!-- -->

   * Change the model
   * Switch API keys
   * Add or modify traffic splitting
   * Configure fallbacks

3. Changes take effect immediately with zero downtime

### Deleting Endpoints[​](#deleting-endpoints "Direct link to Deleting Endpoints")

1. Locate the endpoint in the list
2. Click the delete action
3. Confirm deletion

warning

Deleting an endpoint immediately removes it from service. Any applications using that endpoint will receive errors.

## Zero-Downtime Updates[​](#zero-downtime-updates "Direct link to Zero-Downtime Updates")

The AI Gateway supports dynamic configuration updates. You can:

* Add new endpoints without restarting the server
* Modify existing endpoint configurations
* Change API keys and credentials
* Adjust traffic splitting percentages
* Reorder fallback chains

All changes take effect immediately without disrupting running applications or requiring server restarts.
