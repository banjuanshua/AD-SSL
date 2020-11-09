pwd=$1
echo "$pwd" | sudo -S apt install vim
echo "$pwd" | sudo -S apt updatae 
echo "$pwd" | sudo -S apt upgrade -y
echo "$pwd" | sudo -S apt install build-essential python-dev python-setuptools python-pip python-smbus build-essential libncursesw5-dev libgdbm-dev libc6-dev zlib1g-dev libsqlite3-dev tk-dev libssl-dev openssl libffi-dev -y
echo "$pwd" | sudo -S apt install libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev -y
echo "$pwd" | sudo -S apt install build-essential checkinstall -y
echo "$pwd" | sudo -S apt install libreadline-gplv2-dev libncursesw5-dev libssl-dev libsqlite3-dev tk-dev libgdbm-dev libc6-dev libbz2-dev -y



