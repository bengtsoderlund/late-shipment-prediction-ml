\# GitHubActionsDeployRole



\*\*Purpose\*\*

\- Role assumed by GitHub Actions (via OIDC) to build/push an image to ECR and deploy a new ECS task definition revision + service update.



\*\*Trust policy (OIDC)\*\*

\- Identity provider: `arn:aws:iam::845465762767:oidc-provider/token.actions.githubusercontent.com`

\- Trust JSON stored at: `infra/iam/trust/github\_actions\_oidc.json`

&nbsp; - Binds repo/branch via `Condition` claims (sub/aud).



\*\*Attached policies\*\*

\- `GitHubActionsDeployPolicy`

&nbsp; - ECR push: upload layers + `PutImage` to `arn:aws:ecr:eu-north-1:845465762767:repository/late-shipment-api`

&nbsp; - ECS deploy: `ecs:RegisterTaskDefinition`, `ecs:DescribeTaskDefinition`, `ecs:UpdateService`, `ecs:DescribeServices`

&nbsp; - `iam:PassRole` for:

&nbsp;   - `arn:aws:iam::845465762767:role/ECSLateShipmentTaskRole`

&nbsp;   - `arn:aws:iam::845465762767:role/ECSTaskExecutionRole`

&nbsp;   - with `iam:PassedToService = ecs-tasks.amazonaws.com`



\*\*Who assumes this\*\*

\- GitHub Actions workflow via `aws-actions/configure-aws-credentials@v4` (OIDC).



\*\*References\*\*

\- Policy JSON: `infra/iam/policies/GitHubActionsDeployPolicy.json`

\- Used by CI in `.github/workflows/deploy.yml`

