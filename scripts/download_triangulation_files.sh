echo -e "\e[91m Downloading DAVIS triangulation data\e[39m"
gdown https://drive.google.com/uc?id=1U07e9xtwYbBZPpJ2vfsLaXYMWATt4XyB -O - --quiet | tar xvf -


echo -e "\e[91m Downloading shutterstock triangulation data\e[39m"
gdown https://drive.google.com/uc?id=1om58tVKujaq1Jo_ShpKc4sWVAWBoKY6U -O - --quiet | tar xvf -