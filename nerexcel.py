

#mecab 설치
!set -x \
&& pip install konlpy \
&& curl -s https://raw.githubusercontent.com/konlpy/konlpy/master/scripts/mecab.sh | bash -x

import konlpy
import pandas as pd
import openpyxl
mecab=konlpy.tag.Mecab()
from google.colab import files
file=files.upload()

import io
data = pd.read_csv(io.BytesIO(file['Kiosk_Intent_dataset.csv']),encoding='cp949')
data.head()

column=data[["Q"]]   #가져오는 엑셀파일 문장의 열이름으로 바꿔야함
global data2
data2=pd.DataFrame(columns=['Setence','Word','POS','Tag'])
data2['Word']=items
data2['POS']=pos

for i in range(10):   #문장갯수로 숫자 바꿔야함
   sentence=column.iat[i,0]
   list=mecab.pos(sentence)
   for (item,ipos) in list:
     data2=data2.append([{'Word':item,'POS':ipos}])
 
 data2 









