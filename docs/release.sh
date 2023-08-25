make clean
make html
sudo cp -r _build/html/* /var/www/simple_rl/
sudo systemctl restart nginx