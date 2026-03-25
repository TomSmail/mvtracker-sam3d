# EgoExo4D Download Setup

## Current Status

The download job (ID 60575957) **failed** on Mar 17, 2026 at 14:08 due to AWS credential issues.

**Error:** `Could not connect to the endpoint URL: "https://ego4d-consortium-sharing.s3.amazonaws.com/..."`

## The Problem

Your current AWS credentials are configured for **AWS Bedrock** (your personal account). EgoExo4D requires **separate credentials** from the EgoExo4D consortium.

## Solution: Configure EgoExo4D Credentials

### Option 1: Separate AWS Profile (Recommended)

1. **Get EgoExo4D credentials:**
   - Sign license at https://ego4ddataset.com/egoexo-license/
   - Wait ~2 days for approval
   - You'll receive Access Key ID and Secret Access Key for EgoExo4D

2. **Configure a separate AWS profile:**
   ```bash
   aws configure --profile egoexo4d
   # Enter the EgoExo4D access key and secret
   # Region: us-east-1
   # Output format: json
   ```

3. **Use the profile when downloading:**
   ```bash
   export AWS_PROFILE=egoexo4d
   bash scripts/download_egoexo4d_social.sh
   ```

### Option 2: Temporary Credential Swap

1. Save your current Bedrock credentials:
   ```bash
   cp ~/.aws/credentials ~/.aws/credentials.bedrock.backup
   ```

2. Configure EgoExo4D credentials temporarily:
   ```bash
   aws configure
   # Enter EgoExo4D credentials
   ```

3. Run download:
   ```bash
   bash scripts/download_egoexo4d_social.sh
   ```

4. Restore Bedrock credentials:
   ```bash
   mv ~/.aws/credentials.bedrock.backup ~/.aws/credentials
   ```

## Social Interactions Dataset

Created a new script: **`scripts/download_egoexo4d_social.sh`**

This downloads **only social interaction scenarios** (6 takes, ~50GB):
- ✅ **Basketball** (3 takes) - team sport, multi-person
- ✅ **Music** (2 takes) - collaborative playing
- ✅ **Dance** (1 take) - partner/group dance

**Excluded scenarios:**
- ❌ Cooking (solo)
- ❌ Bike repair (solo)
- ❌ Rock climbing (solo)

## Running the Download

Once AWS credentials are configured for EgoExo4D:

```bash
# Using profile (recommended)
export AWS_PROFILE=egoexo4d
bash scripts/download_egoexo4d_social.sh

# OR submit as SLURM job
sbatch scripts/slurm/download_egoexo4d_social.sh
```

## Alternative: Use SAM-3D-Body's Pre-processed Data

If you only need the data for SAM-3D, consider downloading the pre-processed subset:

```bash
egoexo -o /cluster/scratch/tsmail/datasets/egoexo4d --parts sam_3d_body -y
```

This downloads only the undistorted images used by SAM-3D-Body (much smaller than full VRS files).
