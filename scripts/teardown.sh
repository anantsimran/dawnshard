#!/usr/bin/env zsh
# Delete every cloud resource the compute-backend ticket creates. Safe to rerun:
# each step ignores "not found". Names come from the environment so nothing is
# hard-coded:
#   GKE_CLUSTER, GKE_ZONE, GCP_PROJECT   the GKE cluster
#   KIND_CLUSTER                          the local kind cluster (default dawnshard)
#   S3_BUCKET                             data and checkpoint bucket to empty
#   SAGEMAKER_JOB_PREFIX                  stop training jobs with this name prefix
set -u

if [[ -n "${GKE_CLUSTER:-}" ]]; then
  gcloud container clusters delete "$GKE_CLUSTER" --zone "$GKE_ZONE" \
    --project "$GCP_PROJECT" --quiet || true
fi

kind delete cluster --name "${KIND_CLUSTER:-dawnshard}" 2>/dev/null || true

if [[ -n "${SAGEMAKER_JOB_PREFIX:-}" ]]; then
  aws sagemaker list-training-jobs --status-equals InProgress \
    --name-contains "$SAGEMAKER_JOB_PREFIX" --query 'TrainingJobSummaries[].TrainingJobName' \
    --output text | tr '\t' '\n' | xargs -r -n1 aws sagemaker stop-training-job --training-job-name
fi

if [[ -n "${S3_BUCKET:-}" ]]; then
  aws s3 rm "s3://$S3_BUCKET" --recursive || true
fi

echo "teardown done; check the billing console for anything left running"
