`clang++ env.cpp -o env.out -std=c++17 -O0 -Wall -Wextra -fsanitize=address`
`python3 -m handyrl.envs.ahc008`


server
```bash
sudo apt update

# g++ とか入れる
sudo apt install zip unzip build-essential sysstat -y

# Rust の環境構築
# curl -s 進捗を表示しない -S エラーは表示する -L リダイレクトがあったらリダイレクト先に行く
# sh -s 標準入力でコマンドを指定
curl https://sh.rustup.rs -sSf | sh -s -- -y --default-toolchain stable
source .bashrc
# インストールできたか確認
cargo --version

# Python の環境構築 (Anaconda を使う、適宜最新版を確認 https://www.anaconda.com/products/individual)
# curl -O ファイルに保存
curl https://repo.anaconda.com/archive/Anaconda3-2021.11-Linux-x86_64.sh -sSLO
# Anaconda をインストールするスクリプトの -b は yes の入力を省略したりするオプション
sh Anaconda3-2021.11-Linux-x86_64.sh -b
rm Anaconda3-2021.11-Linux-x86_64.sh
anaconda3/bin/conda init
source .bashrc
# インストールできたか確認
conda --version

# nvidia-driver を入れる
curl https://raw.githubusercontent.com/GoogleCloudPlatform/compute-gpu-installation/main/linux/install_gpu_driver.py --output install_gpu_driver.py
sudo python3 install_gpu_driver.py

# PyTorch のインストール
conda install pytorch cudatoolkit=11.3 -c pytorch -y


git clone https://lgeu:$TOKEN@github.com/lgeu/ahc008.git
cd ahc008
wget https://img.atcoder.jp/ahc008/tools_v3.zip
unzip tools_v3.zip
cd tools
seq 0 999 > seeds.txt
cargo run --release --bin gen seeds.txt
cargo run --release --bin tester
cd -
g++ -std=c++17 -Wall -Wextra -Ofast env.cpp -o env.out
cd HandyRL/
sed -i -e 's/num_batchers:.*/num_batchers: 3/' config.yaml
sed -i -e 's/maximum_episodes:.*/maximum_episodes: 64/' config.yaml
sed -i -e 's/minimum_episodes:.*/minimum_episodes: 16/' config.yaml
sed -i -e 's/forward_steps:.*/forward_steps: 16/' config.yaml
sed -i -e 's/update_episodes:.*/update_episodes: 100/' config.yaml
sed -i -e 's/restart_epoch:.*/restart_epoch: 13/' config.yaml
cat config.yaml
python3 main.py --train-server
```

workder
```bash
sudo apt update

# g++ とか入れる
sudo apt install zip unzip build-essential sysstat -y

# Rust の環境構築
# curl -s 進捗を表示しない -S エラーは表示する -L リダイレクトがあったらリダイレクト先に行く
# sh -s 標準入力でコマンドを指定
curl https://sh.rustup.rs -sSf | sh -s -- -y --default-toolchain stable
source .bashrc
# インストールできたか確認
cargo --version

# Python の環境構築 (Anaconda を使う、適宜最新版を確認 https://www.anaconda.com/products/individual)
# curl -O ファイルに保存
curl https://repo.anaconda.com/archive/Anaconda3-2021.11-Linux-x86_64.sh -sSLO
# Anaconda をインストールするスクリプトの -b は yes の入力を省略したりするオプション
sh Anaconda3-2021.11-Linux-x86_64.sh -b
rm Anaconda3-2021.11-Linux-x86_64.sh
anaconda3/bin/conda init
source .bashrc
# インストールできたか確認
conda --version

# PyTorch のインストール
conda install pytorch -c pytorch -y


git clone https://lgeu:$TOKEN@github.com/lgeu/ahc008.git
cd ahc008
wget https://img.atcoder.jp/ahc008/tools_v3.zip
unzip tools_v3.zip
cd tools
seq 0 999 > seeds.txt
cargo run --release --bin gen seeds.txt
cargo run --release --bin tester
cd -
g++ -std=c++17 -Wall -Wextra -Ofast env.cpp -o env.out
cd HandyRL/
sed -i -e 's/num_parallel:.*/num_parallel: 30/' config.yaml
cat config.yaml
```

`python3 main.py --worker`
