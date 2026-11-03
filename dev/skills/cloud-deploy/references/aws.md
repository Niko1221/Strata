# AWS

- Container: **App Runner** for one simple service; **ECS on Fargate** when it needs a VPC, several services or a load balancer.
- Static site: S3 bucket + CloudFront.
- Function: Lambda (+ API Gateway or a function URL).

## Checklist (container on App Runner)

1. `aws ecr create-repository --repository-name <app>`
2. `aws ecr get-login-password | docker login --username AWS --password-stdin <ACCOUNT_ID>.dkr.ecr.<REGION>.amazonaws.com`
3. `docker build -t <app> .`, then tag it as `<ACCOUNT_ID>.dkr.ecr.<REGION>.amazonaws.com/<app>:latest` and push it.
4. Create the App Runner service from the ECR image (console or `aws apprunner create-service`), port as in the Dockerfile.
5. Check: open the service URL. Undo: `aws apprunner delete-service`, then delete the ECR repository.
