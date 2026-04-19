name: Machine Learning CI/CD

on:
  push:
    branches:
      - main
  pull_request:
    branches:
      - main

permissions:
  contents: write

jobs:
  ml_workflow:
    runs-on: ubuntu-latest
    
    steps:
    - name: Checkout Code
      uses: actions/checkout@v4
      with:
        fetch-depth: 0
        lfs: true

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'

    - name: Install Dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r MLProject/requirements.txt

    # Basic (2 pts): Membuat worflow CI yang dapat membuat model machine learning ketika trigger terpantik.
    - name: Train Model (MLflow Run)
      run: |
        cd MLProject
        mlflow run . --env-manager=local --experiment-name CreditRisk_Basic

    # Skilled (3 pts): Menyimpan artefak ke repositori (menggunakan Git LFS)
    - name: Commit and Push Model with Git LFS
      if: github.event_name == 'push' && github.ref == 'refs/heads/main'
      run: |
        git config --global user.name "github-actions[bot]"
        git config --global user.email "github-actions[bot]@users.noreply.github.com"
        
        # Inisialisasi Git LFS dan tracking (jika belum ada di .gitattributes)
        git lfs install
        
        # Tambahkan file model yang dihasilkan
        git add .gitattributes
        git add MLProject/saved_model/
        
        # Hanya commit jika ada perubahan
        if ! git diff --cached --quiet; then
          git commit -m "chore: update model artifacts via Git LFS [skip ci]"
          # Push ke branch terpisah 'model-artifacts'
          git push origin HEAD:model-artifacts --force
        else
          echo "No changes in model artifacts."
        fi

    # Advanced Testing: Run unit tests for model serving
    - name: Run Model Serving Tests
      run: |
        cd MLProject
        python -m pytest test_model_serving.py -v

    # Advance (4 pts): Membuat Docker Image ke Docker Hub menggunakan custom Dockerfile.
    - name: Login to Docker Hub
      if: github.event_name == 'push' && github.ref == 'refs/heads/main'
      uses: docker/login-action@v3
      with:
        username: ${{ secrets.DOCKERHUB_USERNAME }}
        password: ${{ secrets.DOCKERHUB_TOKEN }}

    - name: Build and Push Docker Image
      if: github.event_name == 'push' && github.ref == 'refs/heads/main'
      run: |
        cd MLProject
        # Membangun image menggunakan custom Dockerfile
        docker build -t ${{ secrets.DOCKERHUB_USERNAME }}/workflow-ci-msml-raditfiransah:latest .
        
        # Push ke Docker Hub
        docker push ${{ secrets.DOCKERHUB_USERNAME }}/workflow-ci-msml-raditfiransah:latest

    # Validasi Docker Image
    - name: Validate Docker Image
      if: github.event_name == 'push' && github.ref == 'refs/heads/main'
      run: |
        # Pull image dari Docker Hub
        docker pull ${{ secrets.DOCKERHUB_USERNAME }}/workflow-ci-msml-raditfiransah:latest
        
        # Jalankan container dalam background
        docker run -d -p 5000:5000 \
          -e MODEL_PATH=/app/saved_model \
          --name mlflow-server \
          ${{ secrets.DOCKERHUB_USERNAME }}/workflow-ci-msml-raditfiransah:latest
        
        # Tunggu server siap
        sleep 15
        
        # Validasi health endpoint
        echo "Validasi health endpoint..."
        HEALTH_STATUS=$(curl -s -w "%{http_code}" -o /dev/null http://localhost:5000/health || echo "000")
        if [ "$HEALTH_STATUS" -eq 200 ]; then
          echo "✓ Health endpoint berhasil diakses"
        else
          echo "✗ Health endpoint gagal diakses dengan status: $HEALTH_STATUS"
          docker logs mlflow-server
          exit 1
        fi
        
        # Validasi metrics endpoint
        echo "Validasi metrics endpoint..."
        METRICS_STATUS=$(curl -s -w "%{http_code}" -o /dev/null http://localhost:5000/metrics || echo "000")
        if [ "$METRICS_STATUS" -eq 200 ]; then
          echo "✓ Metrics endpoint berhasil diakses"
          # Cek apakah ada metrics yang di-expose
          curl -s http://localhost:5000/metrics | head -20
        else
          echo "✗ Metrics endpoint gagal diakses dengan status: $METRICS_STATUS"
          docker logs mlflow-server
          exit 1
        fi
        
        # Validasi predict endpoint dengan data dummy
        echo "Validasi predict endpoint..."
        PREDICT_STATUS=$(curl -s -w "%{http_code}" -o /dev/null -X POST \
          -H "Content-Type: application/json" \
        -d '{"features": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}' \
        http://localhost:5000/predict || echo "000")
        if [ "$PREDICT_STATUS" -eq 200 ]; then
        echo "✓ Predict endpoint berhasil diakses"
        curl -s -X POST -H "Content-Type: application/json" \
          -d '{"features": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}' \
          http://localhost:5000/predict
        else
          echo "✗ Predict endpoint gagal diakses dengan status: $PREDICT_STATUS"
          docker logs mlflow-server
          exit 1
        fi
        
        # Cleanup
        docker stop mlflow-server
        docker rm mlflow-server
        echo "✓ Semua validasi berhasil!"
