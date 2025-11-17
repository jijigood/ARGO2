# 上传ARGO2项目到GitHub

## 当前状态
- 本地Git仓库大小: 6.0GB
- 包含数据库文件 (4.2GB)
- 已完成一次提交

## ⚠️ 问题
GitHub限制单个文件不能超过100MB,您的数据库文件会导致推送失败。

## 解决方案

### 选项1: 排除数据库文件推送 (推荐且简单)

数据库文件通常不需要版本控制,建议排除:

\`\`\`bash
# 1. 撤销当前提交(保留文件)
git reset --soft HEAD~1

# 2. 从暂存区移除数据库文件
git reset HEAD ARGO/Environments/chroma_store/

# 3. 重新提交
git commit -m "Initial commit with ARGO project (database excluded)"

# 4. 添加GitHub远程仓库
# 先在GitHub创建仓库: https://github.com/new
git remote add origin https://github.com/YOUR_USERNAME/ARGO2.git

# 5. 推送到GitHub
git push -u origin main
\`\`\`

### 选项2: 使用Git LFS管理大文件 (复杂但完整)

如果必须上传数据库:

\`\`\`bash
# 1. 安装Git LFS (需要root权限)
# Ubuntu/Debian:
sudo apt-get update
sudo apt-get install git-lfs

# 2. 初始化Git LFS
git lfs install

# 3. 撤销当前提交
git reset --soft HEAD~1

# 4. 从暂存区移除数据库
git reset HEAD ARGO/Environments/chroma_store/

# 5. 配置LFS跟踪大文件
git lfs track "*.sqlite3"
git lfs track "*.bin"
git lfs track "*.pickle"

# 6. 添加 .gitattributes
git add .gitattributes

# 7. 提交项目文件
git commit -m "Initial commit with ARGO project"

# 8. 添加并提交大文件
git add ARGO/Environments/chroma_store/
git commit -m "Add database files via LFS"

# 9. 添加远程仓库
git remote add origin https://github.com/YOUR_USERNAME/ARGO2.git

# 10. 推送到GitHub
git push -u origin main
\`\`\`

## 推荐做法

**建议使用选项1**,原因:
- 数据库文件可以在本地重新生成
- 避免仓库体积过大
- 推送速度快
- 克隆仓库更快

如有需要,可以提供单独的数据库下载链接。
