# 我的软件介绍

这是一款我设计的软件，本文将分为项目部署和功能介绍两大模块进行详细介绍。

## 目录

1. [项目部署](#项目部署)
2. [功能介绍](#功能介绍)
---

## 项目部署

项目部署包括软件安装、配置和启动等步骤。

### 软件安装

**在Ubuntu上安装**：

1. **安装Java 1.8**

   打开终端并运行以下命令来添加 PPA：
   ```bash
   sudo add-apt-repository ppa:openjdk-r/ppa
   sudo apt-get update
   ```
   接着，安装 Java 1.8：
   ```bash
   sudo apt-get install openjdk-8-jdk
   ```
   您可以使用下面的命令来验证您的安装：
   ```bash
   java -version
   ```

2. **安装 IntelliJ IDEA**
   
   你可以从官方网站下载 IntelliJ IDEA。打开终端并运行以下命令：
   ```bash
   sudo snap install intellij-idea-community --classic
   ```
   这将会安装 IntelliJ IDEA Community Edition。

3. **安装 Maven**
   
   使用下面的命令来安装 Maven：
   ```bash
   sudo apt-get install maven
   ```
   验证 Maven 安装：
   ```bash
   mvn -version
   ```

4. **安装 MySQL 8.0**
   
   执行以下命令添加 MySQL APT 存储库：
   ```bash
   wget https://dev.mysql.com/get/mysql-apt-config_0.8.15-1_all.deb
   sudo dpkg -i mysql-apt-config_0.8.15-1_all.deb
   ```
   当提示选择 MySQL 产品版本时，选择 Ok 并按 Enter 确认。然后更新 APT 存储库：
   ```bash
   sudo apt-get update
   ```
   最后，安装 MySQL：
   ```bash
   sudo apt-get install mysql-server
   ```
   在安装过程中，系统将提示你设置 MySQL 的 root 用户密码。

**在 Windows 上安装**：

1. **安装 Java 1.8**

   首先从 Oracle 官方网站下载 Java SE Development Kit 8。下载完成后，运行安装程序并按照提示进行安装。安装完成后，你需要在系统环境变量中配置 Java。

2. **安装 IntelliJ IDEA**

   从 IntelliJ IDEA 官方网站下载安装程序，并按照提示进行安装。

3. **安装 Maven**

   从 Maven 官方网站下载 zip 文件。解压缩文件到你想要的位置（例如，C:\Program Files\Apache\maven）。然后，需要在系统环境变量中配置 Maven。

4. **安装 MySQL 8.0**

   从 MySQL 官方网站下载 MySQL Installer。运行安装程序，并按照提示进行安装。在安装过程中，你将需要设置 root 用户的密码。

**在两个操作系统上进行配置**：

1. **配置 Maven 为阿里云镜像**

   打开 Maven 的 settings.xml 文件（通常位于Maven安装目录的conf文件夹下），在 `<mirrors></mirrors>` 标签内添加阿里云镜像：
   ```xml
   <mirror>
      <id>alimaven</id>
      <mirrorOf>central</mirrorOf>
      <name>aliyun maven</name>
      <url>http://maven.aliyun.com/nexus/content/groups/public/</url>
   </mirror>
   ```
   保存并关闭文件。

2. **MySQL 的相关配置**

   在 MySQL 中，你可以使用命令行或者 GUI 工具（如 MySQL Workbench）进行配置。以下是一些基础的命令：
   
   - 登录到 MySQL：
     ```bash
     mysql -u root -p
     ```
     你将需要输入之前设置的 root 密码。
   - 创建新的数据库：
     ```sql
     CREATE DATABASE mydatabase;
     ```
   - 创建新用户：
     ```sql
     CREATE USER 'myuser'@'localhost' IDENTIFIED BY 'mypassword';
     ```
   - 给用户授权：
     ```sql
     GRANT ALL PRIVILEGES ON mydatabase.* TO 'myuser'@'localhost';
     ```

3. **在 IntelliJ IDEA 上配置 Maven 和 Java 1.8**

   - **配置 Maven**：打开 IntelliJ IDEA，点击 `File -> Settings -> Build, Execution, Deployment -> Build Tools -> Maven`。在 `Maven home directory` 中输入你的 Maven 安装目录，然后在 `User settings file` 中输入你的 settings.xml 文件路径。

   - **配置 Java**：点击 `File -> Project Structure -> Project`，在 `Project SDK` 中选择你的 Java 安装目录。

完成上述所有步骤后，你的开发环境应该已经准备就绪。如果你遇到任何问题，都可以在网上查找相关教程，或者询问具有经验的开发者。



**在 Ubuntu 上安装 Anaconda：**

1. 从 Anaconda 的[官方网站](https://www.anaconda.com/products/distribution)下载对应版本的 Anaconda 安装包。

2. 打开终端，导航到下载的安装包所在的目录，并运行以下命令来安装 Anaconda：
    ```bash
    bash Anaconda3-2021.05-Linux-x86_64.sh
    ```
3. 安装过程中，需要接受许可协议并选择安装目录。安装完成后，关闭并重新打开终端窗口以使变动生效。

**在 Windows 上安装 Anaconda：**

1. 从 Anaconda 的[官方网站](https://www.anaconda.com/products/distribution)下载对应版本的 Anaconda 安装包。

2. 双击下载的 `.exe` 文件开始安装。

3. 在安装过程中，选择 "Just Me"，除非你想让所有用户都能使用 Anaconda。在 "Advanced Options" 页面，选中 "Add Anaconda to my PATH environment variable" 以便在命令行中使用 `conda` 命令。点击 "Install" 开始安装。

**在两个操作系统上安装 PyCharm：**

在 JetBrains 的[官方网站](https://www.jetbrains.com/pycharm/download/)下载并安装 PyCharm。按照提示完成安装过程。安装完成后，你可以通过搜索 "PyCharm" 来启动它。

**设置清华源：**

打开终端或命令行，执行以下命令来设置 pip 的镜像源为清华大学的源：

```bash
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

**安装 Python 包：**

在终端或命令行中，使用 `pip install` 命令来安装你需要的 Python 包。例如：

```bash
pip install flask requests mysql-connector-python pandas numpy tensorflow keras torch scikit-learn scipy
```

注意：一些包可能已经在 Anaconda 中预装好了，所以在安装时可能会显示 "Requirement already satisfied"。

**在 PyCharm 中配置 Python 环境：**

在 PyCharm 中，选择 `File -> Settings -> Project -> Project Interpreter`，点击右上角的齿轮图标，选择 "Add..."。在弹出的对话框中，选择你的 Anaconda 安装目录下的 `python.exe` 文件作为 Python 解释器。点击 "OK"，再次点击 "OK"，你的 Python 环境就配置完成了。

以上就是在 Ubuntu 和 Windows 上安装 Anaconda、PyCharm，以及配置 Python 环境的步骤。希望对你有所帮助。

### 配置

- 你想要的Markdown格式的文档应该如下：

  ## 在 IDEA 中打开项目

  1. 解压下载的项目压缩文件。
  2. 打开 JetBrains IDEA，选择 `File -> Open...`，导航到解压后的项目目录，点击 "OK"。IDEA 将会加载项目。

  ## 配置并构建项目

  1. 打开 IDEA 的内置终端（在底部工具栏找到 "Terminal" 选项卡），或者在项目目录中打开系统终端。
  2. 在终端中输入以下命令：
      ```bash
      mvn clean install
      ```
      这将会下载项目需要的依赖并构建项目。

  ## 配置项目

  打开 `application.yml`（或相应的配置文件），并输入以下配置：

  ```yaml
  spring:
    web:
      resources:
        static-locations: classpath:/static/
    datasource:
      url: jdbc:mysql://localhost:3306/artificial_train?useUnicode=true&characterEncoding=UTF-8&useSSL=false&serverTimezone=UTC&allowPublicKeyRetrieval=true # 这个URL你得看你具体的数据库的IP，也许要改
      username: root # 你可以修改这个数据库角色名称
      password: GJSE8YqDww # 你可以修改这个数据库密码
      driver-class-name: com.mysql.cj.jdbc.Driver
    servlet:
      multipart:
        max-file-size: 50MB
        max-request-size: 50MB
  
  mybatis:
    mapper-locations: classpath:mapper/*.xml
    type-aliases-package: com.jerry86189.artifitialmanagement.entity
  
  logging:
    level:
      mybatis: DEBUG
      org:
        springframework:
          web: DEBUG
  
  server:
    port: 8888 # 你可以修改这个端口号
  
  app:
    file:
      upload-dir: C:/my_upload_dir/ # 修改为你的上传目录，你得确保这个目录真实并且存在
  ```

  **注意：** 请根据你的操作系统和服务器环境修改 `upload-dir` 的值。

  ## 创建数据库和表

  在 MySQL 中，运行以下 SQL 语句来创建数据库和表：

  ```sql
  create schema artificial_train;
  
  create table artificial_train.user
  (
      user_id  bigint auto_increment
          primary key,
      username varchar(255)           not null,
      password varchar(255)           not null,
      role     enum ('ADMIN', 'NORM') not null,
      constraint username
          unique (username)
  )
      charset = utf8mb3;
  
  create table artificial_train.model
  (
      model_id              bigint auto_increment
          primary key,
      x_names_json          text null,
      y_names_json          text null,
      deep_cnn_hy_para_json text null,
      knn_hy_para_json      text null,
      nn_hy_para_json       text null
  )
      charset = utf8mb3;
  
  create table artificial_train.file_info
  (
      file_id          bigint auto_increment
          primary key,
      uploader_id      bigint       null,
      file_name        varchar(255) null,
      upload_timestamp timestamp    not null,
      file_size        bigint       not null,
      file_path        varchar(255) not null,
      constraint file_name
          unique (file_name),
      constraint fk_file_info_uploader_id
          foreign key (uploader_id) references user (user_id)
              on update cascade on delete set null
  )
      charset = utf8mb3;
  
  create table artificial_train.missing_msg
  (
      missing_id     bigint auto_increment
          primary key,
      columns_json   longtext                                           null,
      type           tinyint(1)                                         null,
      missing_method enum ('mean', 'median', 'mode', 'knn', 'rf', 'lr') null
  )
      charset = utf8mb3;
  
  create table artificial_train.outlier_msg
  (
      outlier_id     bigint auto_increment
          primary key,
      cols_json      longtext                                                       null,
      delete_method  enum ('mean_std', 'boxplot', 'cluster')                        null,
      threshold      int                                                            null,
      type           tinyint(1)                                                     null,
      replace_method enum ('mean', 'median', 'mode', 'constant', 'random', 'model') null,
      detect_method  varchar(20)                                                    null
  )
      charset = utf8mb3;
  
  create table artificial_train.operate_msg
  (
      operate_id bigint auto_increment
          primary key,
      file_id    bigint null,
      user_id    bigint null,
      missing_id bigint null,
      outlier_id bigint null,
      model_id   bigint null,
      accuracy   double null,
      pre        double null,
      recall     double null,
      f1         double null,
      constraint operate_msg_ibfk_1
          foreign key (file_id) references file_info (file_id)
              on update cascade on delete set null,
      constraint operate_msg_ibfk_2
          foreign key (user_id) references user (user_id)
              on update cascade on delete cascade,
      constraint operate_msg_ibfk_3
          foreign key (missing_id) references missing_msg (missing_id)
              on update cascade on delete set null,
      constraint operate_msg_ibfk_4
          foreign key (outlier_id) references outlier_msg (outlier_id)
              on update cascade on delete set null,
      constraint operate_msg_ibfk_5
          foreign key (model_id) references model (model_id)
              on update cascade on delete set null
  )
      charset = utf8mb3;
  
  create index file_id
      on artificial_train.operate_msg (file_id);
  
  create index missing_id
      on artificial_train.operate_msg (missing_id);
  
  create index model_id
      on artificial_train.operate_msg (model_id);
  
  create index outlier_id
      on artificial_train.operate_msg (outlier_id);
  
  create index user_id
      on artificial_train.operate_msg (user_id);
  ```

  完成以上步骤后，你就可以运行并使用你的 Spring Boot 项目了。如果在配置或运行过程中遇到任何问题，可以查阅相应的文档或者寻求帮助。

  ### 配置 Python 环境

  1. 首先解压你的 Python 项目的压缩文件。
  2. 打开 JetBrains PyCharm，选择 `File -> Open...`，导航到解压后的项目目录，点击 "OK"。PyCharm 将会加载项目。
  3. 在 PyCharm 中配置 Python 解释器。你可以通过选择 `File -> Settings -> Project: projectName -> Python Interpreter`。点击右上角的齿轮图标，然后选择 "Add..."，然后选择你刚刚通过 Anaconda 安装的 Python 环境。

  ### 配置数据库连接

  在 Python 代码中，你需要配置 MySQL 数据库连接。下面是一个配置的例子：

  ```python
  # 创建数据库连接
  config = {
      'user': 'root',# 你可以修改这个数据库角色名称
      'password': 'GJSE8YqDww',# 你可以修改这个数据库密码
      'host': 'localhost',# 这个你得看你具体的数据库的IP，也许要改
      'database': 'artificial_train',
      'raise_on_warnings': True
  }
  ```

  你需要根据你的 MySQL 数据库的实际情况来修改 `'user'`，`'password'` 和 `'host'` 字段的值。

  完成以上步骤后，你就可以运行并使用你的 Python 项目了。如果在配置或运行过程中遇到任何问题，可以查阅相应的文档或者寻求帮助。



[返回目录](#目录)

---

## 功能介绍

功能介绍部分将详细说明软件的各项功能及其用法。

### 功能1：数据读取模块

 - 用户从前端页面传递数据（CSV文件），并将数据传入后端。

### 功能2：数据预处理模块

 - 在数据预处理模块中，用户可以选择对**空值**进行删除或填补，当用户选择填补空值时，用户可以选择用均值，中位数，众数或自己指定的值来进行填补。同时，用户也可以选择用模型来填补数据，用户可以选择用Knn，随机森林或线性回归模型对空值进行填补。
 - 同样，用户也对**异常值**进行删除或填补，当我们先择删除异常值时，首先需要识别异常值，在这里我们提供三种方法来识别异常值分别是，基于均值和标准差，基于四分位数和基于KMeans聚类放法来识别异常值，之后删除；当我们选择删除异常值时，同样先识别异常值，这里给出四种方法来识别异常值，分别是基于z-score，基于四分位数，基于椭圆包络法和基于局部异常因子法，之后再填充异常值，用户可以选择用均值，中位数，众数，自己指定的值，随机填补或模型填补。

### 功能3：模型训练模块

- 在这里我们提供了三种模型来给用户选择，分别是 Knn, DeepCnn, LSTM的神经网络
- 用户可以自己选择模型的参数，等待模型创建完成后，模型会根据十折验证来评估模型，并返回平均召回率，平均准确率，平均精准率和平均F1-score给前端。

### 功能4：可视化模块
- 前端收到后端的四个评估分数，并用这四个分数来做出图形，例如条形图和扇形图等，将分数可视化。

[返回目录](#目录)

---、
