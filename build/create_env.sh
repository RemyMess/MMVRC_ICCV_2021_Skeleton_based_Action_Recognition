echo "AUTHOR: 'Setting environment.'"
sudo apt-get update
sudo apt-get install python3-pip
sudo pip3 install virtualenv 

if ! [ -d ".env_mmvrc" ]; then
  virtualenv -p python3 .env_mmvrc
fi