\# ECSTaskExecutionRole



\*\*Purpose\*\*

\- Execution role used by ECS \*\*agent\*\* to pull private images from ECR and write logs to CloudWatch.



\*\*Attached policies\*\*

\- AWS managed: `AmazonECSTaskExecutionRolePolicy`

&nbsp; - Grants: `ecr:GetAuthorizationToken`, `ecr:BatchGetImage`, `ecr:GetDownloadUrlForLayer`,

&nbsp;   `logs:CreateLogStream`, `logs:PutLogEvents`, etc.



\*\*Who assumes this\*\*

\- ECS tasks via `executionRoleArn` in the task definition.



\*\*References\*\*

\- Set in `infra/taskdef.template.json` → `"executionRoleArn": "arn:aws:iam::845465762767:role/ECSTaskExecutionRole"`

