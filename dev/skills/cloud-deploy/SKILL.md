---
name: cloud-deploy
description: Plans and reviews deploying a web app or container to a cloud - AWS, Google Cloud or Azure - with the right service, the commands and a checklist. Use it for questions about hosting, deploying, publishing a service, Docker in the cloud, or choosing between AWS, GCP and Azure.
metadata:
  variants: [aws, gcp, azure]
  author: strata-dev example
---

# Cloud deploy

This skill holds one reference file per cloud. Read **only** the one that fits; the others would only fill the context.

## Choose the variant

| The person says or uses | Read |
| --- | --- |
| AWS, Amazon, ECS, Lambda, S3, `aws` CLI | `references/aws.md` |
| Google Cloud, GCP, Cloud Run, `gcloud` | `references/gcp.md` |
| Azure, Microsoft, App Service, `az` CLI | `references/azure.md` |
| no cloud named | ask which one; if they want advice, compare the three in two sentences each, then read the chosen file |

## Workflow

1. Find out what is deployed (static site, container, function) and where it should run (region, budget).
2. Read the variant file and follow its checklist.
3. Give the commands in order, each with one sentence of what it does. Never print a secret; use placeholders like `<ACCOUNT_ID>`.
4. End with how to check that it works and how to undo it.
