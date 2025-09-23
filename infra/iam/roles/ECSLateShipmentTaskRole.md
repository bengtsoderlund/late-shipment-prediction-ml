\# ECSLateShipmentTaskRole



\*\*Purpose\*\*

\- \*\*Application task role\*\* used by the container to access runtime resources (read-only S3 artifacts).



\*\*Attached policies\*\*

\- Inline/attached: `LateShipmentArtifactsReadOnly`

&nbsp; - Grants:

&nbsp;   - `s3:ListBucket` on `arn:aws:s3:::late-shipments-artifacts-bengt`

&nbsp;   - `s3:GetObject` on `arn:aws:s3:::late-shipments-artifacts-bengt/\*`

&nbsp; - Scope: least-privilege read of model/preprocessing artifacts.



\*\*Who assumes this\*\*

\- ECS tasks via `taskRoleArn` in the task definition.



\*\*References\*\*

\- Set in `infra/taskdef.template.json` → `"taskRoleArn": "arn:aws:iam::845465762767:role/ECSLateShipmentTaskRole"`

\- Policy JSON stored at `infra/iam/policies/LateShipmentArtifactsReadOnly.json`

