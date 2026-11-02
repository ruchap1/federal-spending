# AWS deployment

The target deployment is Docker on Amazon EC2, with Amazon ECR used as the
container registry. The application will query public USAspending endpoints
and use a local ChromaDB index for narrative documents.

The repository now includes a `Dockerfile`. Build and run it locally first:

```bash
docker build -t federal-spending-rag .
docker run --rm -p 8501:8501 federal-spending-rag
```

For a small portfolio demo, deploy the image to Amazon ECR and run it on a
small Amazon EC2 instance. Open TCP port 8501 only from trusted IP addresses
in the security group. The app needs no AWS credentials because it reads
public USAspending.gov endpoints.

The image builds the processed chunks and Chroma index during `docker build`.
For production, move Chroma to durable storage or use a managed vector
database, pin model artifacts in the image, and put the app behind HTTPS.
Never commit AWS credentials or API secrets.
