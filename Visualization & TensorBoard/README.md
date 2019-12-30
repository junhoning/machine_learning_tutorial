# README

# 시작하기

---

위의 코드들은 기본적으로 google colaboratory로 작성 하였기 때문에 local에서보다는 google colab 사용을 권해드립니다. 

google colab으로 바로 실행하는 방법은 뒤의 [github.com/](http://github.com/) 까지 지우고 대신 '[https://colab.research.google.com/github/](https://colab.research.google.com/github/)"를 넣어줍니다. 

예로 들어 [https://github.com/junhoning/machine_learning_tutorial/blob/master/Visualization %26 TensorBoard/[TensorBoard] Grad-CAM.ipynb](https://github.com/junhoning/machine_learning_tutorial/blob/master/Visualization%20%26%20TensorBoard/%5BTensorBoard%5D%20Grad-CAM.ipynb) 는 

[https://colab.research.google.com/github/junhoning/machine_learning_tutorial/blob/master/Visualization %26 TensorBoard/[TensorBoard] Grad-CAM.ipynb](https://colab.research.google.com/github/junhoning/machine_learning_tutorial/blob/master/Visualization%20%26%20TensorBoard/%5BTensorBoard%5D%20Grad-CAM.ipynb)  이렇게 됩니다. 

    %%shell
    # pip install --upgrade tensorflow-gpu
    # pip install --upgrade grpcio
    
    rm -rf ./logs/ 
    
    file="./logs/"
    if [ -d "$file" ]
    then
    	echo "$file found."
    else
        export fileid=1Tu3AWHzXfT6PUSbNIGfvu86PqI8GEke3
        export filename=basic_log.zip
    
        wget --save-cookies cookies.txt 'https://docs.google.com/uc?export=download&id='$fileid -O- \
             | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1/p' > confirm.txt
    
        wget --load-cookies cookies.txt -O $filename \
             'https://docs.google.com/uc?export=download&id='$fileid'&confirm='$(<confirm.txt)
    
        rm -f confirm.txt cookies.txt
        
        unzip $filename -d ./logs | awk 'BEGIN {ORS=" "} {if(NR%10==0)print "."}'
    fi

위의 코드는 기본적인 library와 데이터들을 다운 받기 위함 입니다. 

tensorflow는 현재 기준 19.12.07 에는 tensorflow 1.15로 설치 되어있기 때문에 tensorflow 2.0 - gpu로 업그레이드 해줍니다. 

그리고 tensorboard가 바로 작동이 잘 되지 않기 때문에 grpcio를 설치 해줍니다. 

위의 두 library가 설치가 되었으면 kernel을 restart 하고서 다시 실행 해줍니다. 

아래의 shell 명령어들은 한번 다운 받았으면 더 이상은 다운 받을 필요가 없습니다. 

# Offline으로 log 보기

---

코드의 맨 아래를 보면 각 파일 별로 logs 라는 폴더 안에 log들이 저장이 되는데 

이를 다운 받아 원하는 위치에 압축을 푼 후에 그 위치에서 cmd를 열어 'tensorboard —logdir ./'를 입력해주시면 됩니다. 

Google Colab에서 저장한 파일을 다운 받는 방법은 좌측의 메뉴 중 file을 누르시면 local에서 다운 받으실 수 있습니다. 혹시 보이지 않는 경우 작은 > 모양으로 "Open the left pane"을 누르시면 됩니다. 

혹시나 이해가 되지 않으시는 부분이 있다면 blue_mind88@hotmail.com 연락 주시면 도와드리도록 하겠습니다. 

감사합니다.