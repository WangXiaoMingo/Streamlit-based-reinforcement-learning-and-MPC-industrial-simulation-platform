# Streamlit-based-reinforcement-learning-and-MPC-industrial-simulation-platform
为利用强化学习进行控制算法设计，提供仿真平台

The industrial process simulation platform is a simulation verification platform that integrates model simulation, 
control algorithm research, online model update and data visualization. 
Here you can select different validation objects for the verification of control algorithms, 
supporting mainstream deep learning and machine learning libraries such as Tensorflow, pytorch, sklearn, etc. 
From here you can get the data, design the control algorithm, and upload the model. 

**👈 Select a example from the left sidebar** !

### What you can do ?

#### 1. Data Generation
- Select different models through the drop-down button, set the algorithm parameters, and click Run!

#### 2. Control algorithms research
- Select different algorithms through the drop-down button
- Click the Upload button to upload the designed model！

#### 3. Online Control algorithms research
- Select different algorithms through the drop-down button
- Click the Upload button to upload the designed model！

#### 4. Data visualization
- Select different data to display and analyze!

### How to use ?
- See the help documentation for detailed use!


## Begining，  环境配置

1. 创建环境： conda create -n tf2.6_torch1.10 python=3.6
   
3. 激活环境： conda activate tf2.6_torch1.10
   
5. 安装包：  已配置cuda11.6，python3.6最高支持torch=1.10.2，https://download.pytorch.org/whl/torch/
   
pip install opencv-python==4.3.0.38

pip install mpctools （mpctools仅支持python3.6，默认torch=1.10.2. opencv-python 出错时手动安装：pip install opencv-python==4.3.0.38）

pip install tqdm

pip install tensorflow-gpu==2.6.0  (python=3.6仅支持2.6.0版本, 可参考网站：https://tensorflow.google.cn/install/pip?hl=zh-cn)

pip install torch==1.10.2+cu113 torchvision==0.11.3+cu113 torchaudio==0.10.2 --extra-index-url https://download.pytorch.org/whl/cu113

pip install pyyaml

pip install streamlit


# 注上述的mpctools工具并非我们所需要的， 但安装mpctools可以自动安装常用的依赖包，方便。**一个缺点就是mpctools仅支持python3.6**。

# 因此需要按照上面的安装环境来。当然，**你也可以不安装mpctools，因此就不局限于上述环境，可以安装任意版本的python和tensorflow、pytorch等**

# 我们所需的mpctools见网址：https://bitbucket.org/rawlings-group/mpc-tools-casadi/src/master/

# 手动下载安装包后再安装

# **首先安装 casadi库，否则不成功**。网址：https://github.com/casadi/casadi/releases

# 可以使用

pip install casadi

# 然后解压下载的mpctools压缩包（我将其解压到：D:\Users\Administrator\anaconda3\envs\tf2.6_torch1.10\Lib\site-packages\mpc-tools-casadi，**可放于任意文件夹**），

# 进入到mpctoolssetup.py菜单，输入cmd，进入命令行，此时为：D:\Users\Administrator\anaconda3\envs\tf2.6_torch1.10\Lib\site-packages\mpc-tools-casadi>

# 下一步激活环境：conda activate tf2.6_torch1.10

# 即可看到在虚拟环境中：(tf2.6_torch1.10) D:\Users\Administrator\anaconda3\envs\tf2.6_torch1.10\Lib\site-packages\mpc-tools-casadi>

# 输入安装命令：

python mpctoolssetup.py install

# 完成安装，Writing D:\Users\Administrator\anaconda3\envs\tf2.6_torch1.10\Lib\site-packages\MPCTools-2.4.2-py3.6.egg-info

4.退出：conda deactivate
