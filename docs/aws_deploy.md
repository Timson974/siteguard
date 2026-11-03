# Optional: deploy SiteGuard to AWS (EC2 + S3)

The simplest honest cloud deployment. Cost: an EC2 `t3.small`/`t3.medium` costs a few pence per hour —
**stop or terminate it when you finish** and set a billing alarm (AWS Budgets) first.

## 1. Store the model in S3
```bash
aws s3 mb s3://siteguard-<your-name>
aws s3 cp models/best.pt s3://siteguard-<your-name>/models/best.pt
```

## 2. Launch EC2
- AMI: Ubuntu 24.04, type `t3.medium` (YOLOv8n runs fine on CPU), 20 GB disk.
- Security group: allow SSH (22) from **your IP only**; allow 8501 (dashboard) from your IP only.
- IAM role for the instance with read access to the bucket (`AmazonS3ReadOnlyAccess` or a bucket-specific policy).

## 3. Install Docker and run
```bash
sudo apt-get update && sudo apt-get install -y docker.io docker-compose-v2 awscli git
sudo usermod -aG docker ubuntu && newgrp docker
git clone https://github.com/Timson974/siteguard.git && cd siteguard
aws s3 cp s3://siteguard-<your-name>/models/best.pt models/best.pt
cp .env.example .env   # change POSTGRES_PASSWORD
docker compose up -d --build
```
Open `http://<EC2-public-IP>:8501`.

## 4. Clean up
`docker compose down`, then stop/terminate the instance and delete the bucket if no longer needed.

When this works, the CV line can honestly say: *"Deployed on AWS EC2 with model weights in S3."*
