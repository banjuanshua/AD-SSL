mkdir ~/.pip
touch ~/.pip/pip.conf
echo "[global]" >> ~/.pip/pip.conf
echo "index-url = https://pypi.tuna.tsinghua.edu.cn/simple" >> ~/.pip/pip.conf
echo "[install]" >> ~/.pip/pip.conf
echo "trusted-host=pypi.tuna.tsinghua.edu.cn" >> ~/.pip/pip.conf

pip3 install torch 
pip3 install tensorflow
pip3 install tensorboard
pip3 install tensorboardX
pip3 install opencv-python

