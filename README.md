# 云计算技术课程设计

本仓库为课程设计提交代码仓库，包含任务书要求的 Dockerfile、Kubernetes YAML、核心 Python 代码、Spark 作业、监控配置、CI/CD 流水线和分布式 AI 训练附加题代码。

## 目录结构

```text
Cloud Computing/
├── Dockerfile.backend                 # 第一部分：Flask 后端镜像
├── Dockerfile.frontend                # 第一部分：Nginx 前端镜像
├── docker-compose.yml                 # 任务1：本地联调
├── requirements.txt                   # 后端 Python 依赖
├── backend/
│   └── app.py                         # Flask API，连接 Redis
├── frontend/
│   ├── nginx.conf                     # 前端 Nginx 配置
│   └── static/index.html              # 前端静态页面
├── k8s/
│   ├── 01-backend-config-secret.yaml  # ConfigMap + Secret
│   ├── 02-redis-deployment.yaml       # Redis Deployment
│   ├── 03-backend-deployment.yaml     # Backend Deployment
│   ├── 04-services.yaml               # Service / LoadBalancer
│   ├── 05-redis-pvc.yaml              # Redis PVC
│   ├── 06-frontend-nginx-configmap.yaml
│   ├── 07-frontend-deployment.yaml
│   └── 08-backend-hpa.yaml            # HPA 弹性伸缩
├── spark/
│   ├── Dockerfile.pyspark             # A0 PySpark 镜像
│   ├── Dockerfile.pyspark-douban      # A1-A3 豆瓣分析镜像
│   ├── spark-rbac.yaml
│   ├── sparkapplication-*.yaml        # SparkApplication 作业
│   └── jobs/                          # PySpark / Pandas 核心代码
├── monitoring/
│   ├── kube-prometheus-stack-values.yaml
│   └── grafana-course-dashboard.yaml  # 附加题1 监控 Dashboard
├── C1-ai/
│   ├── Dockerfile.mnist-ddp           # 附加题3 C-1 训练镜像
│   ├── mnist_ddp_train.py             # PyTorch DDP MNIST CNN
│   └── k8s/                           # 单机训练与 2 Pod DDP Job
└── .github/workflows/cce-cicd.yml     # 附加题2 CI/CD 流水线
```

## 镜像地址

```text
swr.cn-north-4.myhuaweicloud.com/swjtu2551/backend
swr.cn-north-4.myhuaweicloud.com/swjtu2551/frontend
swr.cn-north-4.myhuaweicloud.com/swjtu2551/pyspark
swr.cn-north-4.myhuaweicloud.com/swjtu2551/pyspark-douban
swr.cn-north-4.myhuaweicloud.com/swjtu2551/mnist-ddp:cpu
```
