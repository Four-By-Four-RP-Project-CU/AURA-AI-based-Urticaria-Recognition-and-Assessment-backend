# Deploy AURA Backend To Azure

This guide uses `Azure Container Apps` because it is the best fit for this FastAPI backend with two large ML models, OCR, and PDF generation.

## 1. Prepare Hugging Face repos

Create two Hugging Face model repos and upload these files at the repo root.

### Prescription model repo

Files required:

- `config.json`
- `model.pt`
- `temperature.json`
- `ood_stats.json`
- `prototypes.json`

Source from local folder:

- `IT22577160/artifacts/`

### Risk model repo

Files required:

- `config.json`
- `model.pt`
- `preprocess.joblib`

Source from local folder:

- `IT22607232/artifacts/`

Use a read-only Hugging Face token in Azure if the repos are private.

## 2. Local smoke test with Docker

Build:

```powershell
docker build -t aura-backend .
```

Run:

```powershell
docker run --rm -p 8000:8000 `
  -e HF_TOKEN=hf_xxx `
  -e PRESCRIPTION_MODEL_REPO=your-username/aura-prescription-model `
  -e RISK_MODEL_REPO=your-username/aura-risk-model `
  -e MONGODB_URI="your-mongodb-uri" `
  aura-backend
```

Test:

```powershell
curl http://localhost:8000/health
curl http://localhost:8000/IT22577160/health
curl http://localhost:8000/IT22607232/health
```

## 3. Azure CLI setup

Install/update Azure CLI, then run:

```powershell
az login
az upgrade
az extension add --name containerapp --upgrade
az provider register --namespace Microsoft.App
az provider register --namespace Microsoft.OperationalInsights
```

## 4. Create Azure resources

Set your own names first. The registry name must be globally unique and lowercase.

```powershell
$RG="aura-rg"
$LOC="eastus"
$ACR="aurabackendregistry12345"
$ENV="aura-env"
$APP="aura-backend"
```

Create the resource group and registry:

```powershell
az group create --name $RG --location $LOC
az acr create --name $ACR --resource-group $RG --sku Basic --admin-enabled true
```

Build and push the image in Azure:

```powershell
az acr build --registry $ACR --image aura-backend:latest .
```

Create the Container Apps environment:

```powershell
az containerapp env create --name $ENV --resource-group $RG --location $LOC
```

## 5. Create the container app

Get the registry credentials:

```powershell
$ACR_SERVER = az acr show --name $ACR --resource-group $RG --query loginServer -o tsv
$ACR_USER = az acr credential show --name $ACR --query username -o tsv
$ACR_PASS = az acr credential show --name $ACR --query "passwords[0].value" -o tsv
```

Create the app. Start with `2 CPU / 4 GiB` because this backend loads both ML runtimes at startup.

```powershell
az containerapp create `
  --name $APP `
  --resource-group $RG `
  --environment $ENV `
  --image "$ACR_SERVER/aura-backend:latest" `
  --target-port 8000 `
  --ingress external `
  --registry-server $ACR_SERVER `
  --registry-username $ACR_USER `
  --registry-password $ACR_PASS `
  --cpu 2.0 `
  --memory 4.0Gi `
  --min-replicas 0 `
  --max-replicas 1
```

## 6. Add secrets and environment variables

Set secrets:

```powershell
az containerapp secret set `
  --name $APP `
  --resource-group $RG `
  --secrets `
  hf-token="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxx" `
  mongo-uri="your-mongodb-uri"
```

Apply app settings:

```powershell
az containerapp update `
  --name $APP `
  --resource-group $RG `
  --set-env-vars `
  HF_TOKEN=secretref:hf-token `
  PRESCRIPTION_MODEL_REPO=your-username/aura-prescription-model `
  RISK_MODEL_REPO=your-username/aura-risk-model `
  MONGODB_URI=secretref:mongo-uri `
  MONGODB_DB=aura `
  MONGODB_RISK_COLLECTION=risk_results `
  MONGODB_PRESCRIPTION_COLLECTION=prescription_results `
  MONGODB_RISK_BUCKET=risk_assets `
  MONGODB_PRESCRIPTION_BUCKET=prescription_assets
```

## 7. Check deployment

Get the URL:

```powershell
az containerapp show --name $APP --resource-group $RG --query properties.configuration.ingress.fqdn -o tsv
```

Then test:

- `https://<your-url>/health`
- `https://<your-url>/IT22577160/health`
- `https://<your-url>/IT22607232/health`

## 8. Update after code changes

Rebuild and roll out a new image:

```powershell
az acr build --registry $ACR --image aura-backend:latest .
az containerapp update --name $APP --resource-group $RG --image "$ACR_SERVER/aura-backend:latest"
```

## 9. Common failure points

- If startup fails, check that both Hugging Face repos contain the required files at the repo root.
- If OCR fails on Linux, confirm that `tesseract-ocr` installed successfully in the image.
- If the app restarts repeatedly, increase memory to `6 GiB`.
- If MongoDB is optional for your demo, leave `MONGODB_URI` unset and the API will still start without persistence.
- If you want to bundle local model files inside the image instead of downloading from Hugging Face, remove the model-file entries from `.dockerignore` before building.
