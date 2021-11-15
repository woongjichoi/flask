//실행환경:colab

from google.colab import drive
drive.mount('/content/drive')

%cd /content/drive/MyDrive/Colab Notebooks

!pip install import_ipynb

import import_ipynb

import lstmNBsfvoting as intent

%cd /content/drive/MyDrive/Colab Notebooks/

import LSTM_NER_finals as ner

%cd /content/drive/MyDrive/Colab Notebooks/model

import KioskChatbot_attention_im as kiosk

def entity_recognition(sentence):
  return ner.entity_predict(sentence)

#cart  -> (item,num)으로 저장되는 리스트 

#checklist -> cart에서 저장된 (item,num)에 대하여 pack,point,pay를 저장하는 리스트   
#checklist={('item','num'):([],[1]),'pack':[],'point':[],'pay':[]}


###cart에 (item,num)이 하나라도 있어야만 실행될수 있는 의도:4,5,6,7,9,10,11,12 로 만약 카트가 비었는데 호출되면 "먼저 상품을 선택해주세요" 답변 출력 및 entity 요청 
###input 의도가 3일떄만 cart에 item,num을 입력한다
###cart에 input 문장의 의도가 3일때만 (item,num)을 저장해준다 
###cart에 item만 먼저 입력되었을떈 num 입력을 요청한다
###중간의도 19,3 을 23으로 우선 설정

class scenario:
  def __init__(self, item:list, num):
    self.item = []
    self.num = []
    self.bio=[]   #ner tagging 결과 저장
    self.temp=[] #한번은 임시 저장
    self.alert_intent=[4,5,6,7,9,10,11,12]
    self.response=""
    self.pack=""
    self.ask=0 #주문 계속할지 묻는거
    self.flag=0  #주문하던 적이 있는지 체크
    print(f"{self.item} cart가 생성 되었습니다.")
    print(f"상품 : {self.item}, 수량 : {self.num}")
    
  #def check_pack(self,text):
    

  def check_entity(self,text):    #entity=[B-item,I-item,I-item ]이라 가정 
     self.bio=ner.entity_predict(text)
    
  def check_intent(self,text):
     self.intent=intent.predict(text)

  def answer(self,text):
     self.response=kiosk.predict(text)

 # def add_cart():
     
  def fill_cart(self,text):
    
       
     self.check_entity(text) #모든문장이 입력되면 모두 개체명 인식은 진행
     self.check_intent(text) #모든문장이 입력되면 모두 인텐트 파악은 진행
     self.temp=self.bio #임시저장
     self.answer(text)
     if self.intent==3:  #의도가 3일때만 카트에 담기
       if not self.item:
          self.flag=1 
          for w in self.bio:
           if w[1]=='B-item' or w[1]=='I-item':  #상품명 추가
              if w[0]=='아아':
                self.item.append('아이스 아메리카노')
              else:  
                self.item.append(w[0])
           elif w[1]=='B-num' or w[1]=='I-num': #숫자 추가 
              if w[0]=='한 잔' or w[0]=='한잔' or w[0]=='하나' or w[0]=='한':  #bio 태깅 오류로 정해놓음
                self.num.append(1)
              elif w[0]=='주세요' or w[0]=='주문할게요' or w[0]=='잔':   #bio 태깅 오류로 정해놓음
                continue
              else:
                self.num.append(w[0])  
          print(self.response)
          print(f"상품 : {self.item}, 수량 : {self.num}")

       else: #중간의도 중 상품 주문하는것이 3으로 인식된 경우로 self.item이 null값이 아님
          for a in self.bio:
            if a[0]=='한 잔' or a[0]=='한잔' or a[0]=='하나' or a[0]=='한':  #bio 태깅 오류로 정해놓음
                self.num.append(1)
            elif a[0]=='주세요' or a[0]=='주문할게요' or a[0]=='잔':   #bio 태깅 오류로 정해놓음
                continue
            else:
                self.num.append(a[0]) 
            #주문내역 보여주기     
            print(self.item[0],self.num[0],"주문하셨습니다.")
     elif self.intent in self.alert_intent:  
          if (not self.item) or (not self.num): #카트가 비어있으면 카트 추가 x
              print("먼저 상품 주문을 해주세요!")
          else:  #카트가 비어있지 않으면 답변 출력 가능
              print(self.response)
        #elif self.intent==8:
     elif self.intent==14 or self.intent==18:
           print(self.response)
        
           print(self.item,"을 주문하던 중이었습니다.")
           ans=input("주문을 계속 진행하시겠습니까?")
           if ans=='네':
             print(self.item[0],self.num[0],"드릴까요?")
             ans2=input()
             if ans2=='네':
               print(self.item[0],self.num[0],"주문하셨습니다.")
     #elif self.intent==23:  #앞 문장에서 self.item이 개체명인식만 되고 카트에 담기진 않은 상태,뒤이어서 앞서 물어보았던 상품에 대한 주문을 진행하는 상황
       
        else:
            print(self.response)
     self.bio=[] #bio 태깅은 다시 갱신되기 위하여 비움  
        
     
 def chat():
   cart=scenario("",0)
   while True:
     text=input('사용자: ')
     if text =='끝':
        break
     print('AI점원:')
     cart.fill_cart(text)
     #체크리스트가 모두 채워지면 새 시나리오로 바꾸기 
   
    
