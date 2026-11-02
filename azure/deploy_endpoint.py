"""
Automated Azure ML Managed Online Endpoint Deployment Script
Uses Azure AI ML SDK v2 (azure-ai-ml, azure-identity)

Prerequisites:
  pip install azure-ai-ml azure-identity

Usage:
  python azure/deploy_endpoint.py \
    --subscription-id "<YOUR_AZURE_SUBSCRIPTION_ID>" \
    --resource-group "<YOUR_RESOURCE_GROUP_NAME>" \
    --workspace-name "<YOUR_AZURE_ML_WORKSPACE_NAME>"
"""

import argparse
import os
import sys
from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential
from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    Model,
    Environment,
    CodeConfiguration
)
from azure.ai.ml.constants import AssetTypes


def deploy(subscription_id, resource_group, workspace_name, endpoint_name=None):
    print("=" * 70)
    print("🚀 Azure ML Model Deployment Engine: Melbourne Housing Valuation & XAI")
    print("=" * 70)

    # 1. Authenticate with Azure
    print("\n[Step 1/5] Authenticating with Azure...")
    try:
        credential = DefaultAzureCredential()
        ml_client = MLClient(credential, subscription_id, resource_group, workspace_name)
        # Test connection
        ml_client.workspaces.get(workspace_name)
        print(f"✓ Connected to Azure ML Workspace: {workspace_name}")
    except Exception as e:
        print(f"Interactive browser login fallback due to: {e}")
        credential = InteractiveBrowserCredential()
        ml_client = MLClient(credential, subscription_id, resource_group, workspace_name)
        ml_client.workspaces.get(workspace_name)
        print(f"✓ Connected to Azure ML Workspace: {workspace_name}")

    if not endpoint_name:
        import uuid
        endpoint_name = f"melbourne-prop-val-{str(uuid.uuid4())[:6]}"

    print(f"\nTarget Endpoint Name: {endpoint_name}")

    # 2. Register Model Asset
    print("\n[Step 2/5] Registering Model Asset in Azure ML Registry...")
    model_path = "artifacts/models"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model directory not found at {model_path}")

    model_asset = Model(
        name="melbourne-housing-ebm-bundle",
        path=model_path,
        type=AssetTypes.CUSTOM_MODEL,
        description="Explainable Boosting Machine (EBM) & Extra Trees for Melbourne Eastern Suburbs Housing Valuation"
    )
    registered_model = ml_client.models.create_or_update(model_asset)
    print(f"✓ Registered Model '{registered_model.name}' Version: {registered_model.version}")

    # 3. Create Managed Online Endpoint
    print("\n[Step 3/5] Creating Managed Online Endpoint...")
    endpoint = ManagedOnlineEndpoint(
        name=endpoint_name,
        description="Real-time REST API for Melbourne Housing Valuation & XAI attributions",
        auth_mode="key"
    )
    endpoint_poller = ml_client.online_endpoints.begin_create_or_update(endpoint)
    created_endpoint = endpoint_poller.result()
    print(f"✓ Endpoint created: {created_endpoint.name}")
    print(f"  Scoring URI: {created_endpoint.scoring_uri}")

    # 4. Deploy Model with Code & Environment
    print("\n[Step 4/5] Creating Blue Deployment (Instance type: Standard_DS2_v2 or Standard_B2s)...")
    env = Environment(
        name="melbourne-val-env",
        conda_file="azure/conda.yaml",
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest"
    )

    deployment = ManagedOnlineDeployment(
        name="blue",
        endpoint_name=endpoint_name,
        model=registered_model,
        environment=env,
        code_configuration=CodeConfiguration(
            code=".",
            scoring_script="azure/score.py"
        ),
        instance_type="Standard_DS2_v2",  # Supported in Azure for Students quota
        instance_count=1
    )

    dep_poller = ml_client.online_deployments.begin_create_or_update(deployment)
    dep_poller.result()
    print("✓ Deployment completed successfully!")

    # Set 100% traffic to blue deployment
    endpoint.traffic = {"blue": 100}
    ml_client.online_endpoints.begin_create_or_update(endpoint).result()
    print("✓ Traffic allocated: 100% to 'blue' deployment.")

    # 5. Test Live Endpoint with Sample Payload
    print("\n[Step 5/5] Invoking live endpoint with sample Burwood East property...")
    sample_json = "azure/sample_request.json"
    if os.path.exists(sample_json):
        response = ml_client.online_endpoints.invoke(
            endpoint_name=endpoint_name,
            request_file=sample_json
        )
        print("\n=== Live Scoring Response from Azure ML Endpoint ===")
        print(response)
        print("====================================================")

    # Fetch Endpoint Keys for Examiner / Web app consumption
    keys = ml_client.online_endpoints.get_keys(name=endpoint_name)
    print("\n" + "=" * 70)
    print("🎉 DEPLOYMENT SUMMARY FOR EXAMINER SUBMISSION:")
    print("=" * 70)
    print(f"• Azure ML Studio Workspace: https://ml.azure.com/workspaces/{workspace_name}?wsid=/subscriptions/{subscription_id}/resourcegroups/{resource_group}/workspaces/{workspace_name}")
    print(f"• Endpoint Scoring URI:      {created_endpoint.scoring_uri}")
    print(f"• Swagger URI:               {created_endpoint.swagger_uri}")
    print(f"• Primary Auth Key:          {keys.primary_key}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deploy Melbourne Housing Valuation Model to Azure ML")
    parser.add_argument("--subscription-id", required=True, help="Azure Subscription ID")
    parser.add_argument("--resource-group", required=True, help="Azure Resource Group Name")
    parser.add_argument("--workspace-name", required=True, help="Azure ML Workspace Name")
    parser.add_argument("--endpoint-name", required=False, default=None, help="Custom Endpoint Name")

    args = parser.parse_args()
    deploy(
        subscription_id=args.subscription_id,
        resource_group=args.resource_group,
        workspace_name=args.workspace_name,
        endpoint_name=args.endpoint_name
    )

