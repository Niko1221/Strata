# Google Cloud

- Container: **Cloud Run** (scales to zero, pay per request).
- Static site: Firebase Hosting or a Cloud Storage bucket behind a load balancer.
- Function: Cloud Run functions.

## Checklist (container on Cloud Run)

1. `gcloud config set project <PROJECT_ID>`
2. `gcloud run deploy <app> --source . --region <REGION> --allow-unauthenticated` (builds with Cloud Build)
3. Secrets: `gcloud secrets create <name>` and `--set-secrets` on deploy, never plain environment values.
4. Check: open the URL it prints. Undo: `gcloud run services delete <app> --region <REGION>`.
