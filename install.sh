#!/bin/bash

unameOut="$(uname -s)"
case "${unameOut}" in
    Linux*)     machine=Linux;;
    Darwin*)    machine=Mac;;
    *)          machine="UNKNOWN:${unameOut}"
esac
echo "Installing on ${machine}..."

if [ "$machine" = "Linux" ]
then
  ubuntuRelease=$(grep -oP 'VERSION_ID="\K[\d.]+' /etc/os-release)

  sudo apt-get install curl
  curl -fsSL https://raw.githubusercontent.com/dreal/dreal4/master/setup/ubuntu/"${ubuntuRelease}"/install.sh | sudo bash
elif [ "$machine" = "Mac" ]
then
  /usr/bin/curl -fsSL https://raw.githubusercontent.com/dreal/dreal4/master/setup/mac/install.sh | bash
  dreal
else
  echo "Installation on ${machine} not supported. Supported operating systems: macOS, Ubuntu 18.04, Ubuntu 20.04."
  exit
fi

pip install z3-solver
pip install pyparsing
pip install numpy