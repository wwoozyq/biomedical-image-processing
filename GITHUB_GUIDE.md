# GitHub上传指南

## 创建GitHub仓库

1. 登录 [GitHub](https://github.com)
2. 点击右上角 "+" → "New repository"
3. 填写仓库信息：
   - Repository name: `biomedical-image-processing`
   - Description: `生物医学图像处理课程项目`
   - 选择 Public 或 Private
4. 点击 "Create repository"

## 上传项目

在项目根目录执行：

```bash
# 初始化Git仓库
git init

# 添加所有文件
git add .

# 提交
git commit -m "Initial commit"

# 关联远程仓库（替换your-username）
git remote add origin https://github.com/wwoozyq/biomedical-image-processing.git

# 推送
git branch -M main
git push -u origin main
```

## 后续更新

```bash
git add .
git commit -m "更新说明"
git push
```
