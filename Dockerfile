# 基础镜像：选择包含 Miniconda 的镜像
FROM one.reimu.moe/continuumio/miniconda3:24.7.1-0

# 设置工作目录
WORKDIR /MalTutor

# 复制项目文件到容器中
COPY . /MalTutor

# 复制并解压 Conda 环境
COPY myenv_tf.tar.gz /opt/
RUN mkdir -p /opt/conda_envs/myenv_tf && tar -xzf /opt/myenv_tf.tar.gz -C /opt/conda_envs/myenv_tf

# 设置环境变量 PATH
ENV PATH /opt/conda_envs/myenv_tf/bin:$PATH

# 安装额外的 Python 依赖（如果有的话）
RUN pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/

# 使用已打包的环境，无需再次创建环境
# 确保环境激活并执行主脚本或启动命令
# CMD ["python", "main.py"]  # 将 "main.py" 替换为项目实际的入口脚本名称
