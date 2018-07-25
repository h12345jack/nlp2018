### 安装环境

1. 解压 

   ```
   tar zxvf CRF++-0.58.tar.gz
   ```

2. 编译安装CRF++-0.58 

   ```
   cd CRF++-0.58
   ./configure
   make
   sudo make install
   ```

3. 安装python的CRF++包

   ```
   cd python
   python setup.py build 
   sudo python setup.py install 
   ```

4. import CRFPP测试是否安装成功

5. 配置ld.so.conf 

   ```
   若出现ImportError: libcrfpp.so.0: cannot open shared object file: No such file or directory 
   sudo vim /etc/ld.so.conf
   添加：
   include /usr/local/lib
   保存后加载一下
   sudo /sbin/ldconfig -v
   或者直接：ln -s /usr/local/lib/libcrfpp.so.0 /usr/lib/ 
   ```

### API使用

- model文件夹中存放着三个模型：pku.model，msr.model, weibo.model。分别是北大、微软研究院和微博的数据集训练的模型
- crf_api.py为crf的api程序
  - 类CRF：CRF分词器的类
  - 函数CRF.\_\_init__：初始化CRF类，输入参数为指定数据集训练的模型代号（pku, msr, weibo），根据模型代号，确定CRF模型和分词器。
  - 函数CRF.cut：使用tagger（crf分词器， tagger = CRFPP.Tagger("-m " + self.model)，self.model为模型文件路径）对中文字符串进行分词，输入参数为中文字符串，返回分词结果
- test.py为测试api程序

