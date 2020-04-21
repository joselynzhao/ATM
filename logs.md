# file_funtions
this file was created to record the function of python files
(文件功能描述和运行命令)
## atm.pyc
- 拷贝U_data
- 以one-shot数据作为标签计算的中心
- 不采用方差采样
- tagper加完所有的伪标签之后就不再训练
- 运行命令
    - python3.6 atm.py  --total_step 5 --train_tagper_step 3 --exp_order 0

## atm02.py
- 拷贝U_data
- 以one-shot数据作为标签计算的中心
- 不采用方差采样
- 全程训练tagper
- 运行命令
    - python3.6 atm02.py  --total_step 5 --train_tagper_step 3 --exp_order 1


## atm03.py
- 拷贝U_data
- 以one-shot和带标注数据的均值数据作为标签计算的中心
- 不采用方差采样
- 全程训练tagper
- 运行命令
    - python3.6 atm03.py  --total_step 5 --train_tagper_step 3 --exp_order 2
    - python3.6 atm03.py  --total_step 8 --train_tagper_step 4 --exp_order 3

## atm04.py
- 拷贝U_data
- 以one-shot和做为标签计算的中心
- 采用方差采样: tagper方差衰退到最后一步,reid通过stop_vari_step来控制停用
- 全程训练tagper
- 运行命令
    - python3.6 atm04.py  --total_step 5 --train_tagper_step 3 --percent_vari 0.8 --stop_vari_step 4 --exp_order 4
    
## atm05.py
- 拷贝U_data
- 以one-shot和带标注数据的一起来作为L_data来打标签
- 不采用方差采样
- 全程训练tagper
- 运行命令
    - python3.6 atm05.py  --total_step 5 --train_tagper_step 3 --exp_order 5





