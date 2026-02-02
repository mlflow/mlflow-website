# Create and Manage API Keys

API keys serve as reusable credentials that can be shared across multiple endpoints. When you have several endpoints using the same provider, this approach simplifies both initial setup and ongoing credential management.

## Accessing API Keys[​](#accessing-api-keys "Direct link to Accessing API Keys")

Navigate to the AI Gateway section at `http://localhost:5000/#/gateway` and click on the **API Keys** tab.

![API Keys Page](/mlflow-website/docs/latest/assets/images/api-keys-page-120794235927669caacd55978043270f.png)

## Creating an API Key[​](#creating-an-api-key "Direct link to Creating an API Key")

1. Click **Create API Key**

2. Enter a unique name for the key (e.g., `my-openai-key`)

3. Select your provider from the dropdown (OpenAI, Anthropic, Google Gemini, etc.)

4. Choose the authentication method if multiple options are available
   <!-- -->
   * For example, OpenAI supports both standard API key authentication and Azure-specific authentication

5. Enter your credentials (displayed as masked inputs for security)

6. Fill in any provider-specific configuration fields:

   <!-- -->

   * **Azure**: Endpoint URL
   * **GCP**: Project ID

7. Click **Create**

![Create API Key](/mlflow-website/docs/latest/assets/images/create-api-key-680a70020d74cde71f658c2bf2962d9a.png)

## Working with Existing Keys[​](#working-with-existing-keys "Direct link to Working with Existing Keys")

The API Keys page displays all your configured credentials along with important metadata:

* **Endpoints using this key**: See which endpoints depend on each key
* **Last updated**: When the credentials were last modified
* **Created date**: When the key was originally created

The credential values remain masked for security.

### Editing Keys[​](#editing-keys "Direct link to Editing Keys")

To update credentials for a provider:

1. Locate the key in the API Keys list
2. Click the **Edit** button
3. Update the credential value
4. Click **Save**

All endpoints using this key will automatically use the new credentials without requiring any configuration changes.

### Deleting Keys[​](#deleting-keys "Direct link to Deleting Keys")

When deleting a key:

1. The system warns you if any endpoints currently depend on it
2. Review the warning to prevent accidental disruptions
3. Confirm deletion only after ensuring no active endpoints need the key

tip

Creating reusable API keys simplifies credential rotation. When you need to update a credential, edit it once rather than updating every endpoint individually.

## Best Practices[​](#best-practices "Direct link to Best Practices")

1. **Use descriptive names**: Name keys by provider and purpose (e.g., `openai-production`, `anthropic-dev`)
2. **Separate development and production**: Use different keys for different environments
3. **Minimize key sharing**: Create separate keys when different teams or applications need isolated access
4. **Regular rotation**: Periodically rotate credentials for security (see [Encryption & Rotation](/mlflow-website/docs/latest/genai/governance/ai-gateway/api-keys/key-rotation.md))
