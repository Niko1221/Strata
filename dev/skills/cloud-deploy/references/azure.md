# Azure

- Container: **Azure Container Apps**; App Service for a classic web app.
- Static site: Azure Static Web Apps.
- Function: Azure Functions.

## Checklist (container on Container Apps)

1. `az login` and `az group create --name <rg> --location <REGION>`
2. `az containerapp up --name <app> --resource-group <rg> --source . --ingress external --target-port <PORT>`
3. Secrets: `az containerapp secret set` and reference them, never plain environment values.
4. Check: open the FQDN it prints. Undo: `az group delete --name <rg>` (removes everything in the group).
