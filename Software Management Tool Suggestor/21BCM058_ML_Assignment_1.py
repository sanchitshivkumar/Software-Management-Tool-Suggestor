from kivy.config import Config
Config.set('graphics','width','700')
Config.set('graphics','height','350')
Config.set('graphics', 'resizable', 0)
Config.set('graphics', 'borderless', 1)
from kivy.uix.floatlayout import FloatLayout
from kivy.core.window import Window
from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.uix.popup import Popup
from kivy.uix.image import Image
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder
from pandas import read_csv,DataFrame
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
def one(roll):
  print(roll)
  d=read_csv('Machine Learning.csv')
  d=DataFrame(d)
  x1=np.array(d['Roll Number'])
  y=np.array(d['Tool1'])
  l=[]
  for i in range(66):
    l.append(np.array(x1[i][5:8]))
  x=np.array(l).reshape(-1,1)
  x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=7)
  model=DecisionTreeClassifier()
  model.fit(x_train,y_train)
  kfold=ShuffleSplit(n_splits=10,test_size=0.2,random_state=7)
  scoring='accuracy'
  cv=cross_val_score(model,x_train,y_train,cv=kfold,scoring=scoring)
  #print(cv.mean()*100)
  tool=model.predict(np.array([roll]).reshape(-1,1))
  return tool[0].strip().lower()
def two(roll_tool1):
  print(roll_tool1)
  d=read_csv('Machine Learning.csv')
  d=DataFrame(d)
  x=d['Roll Number']
  x1=d['Tool1']
  y=d['Tool2']
  data=[]
  for i in range(66):
    l=(x[i],x1[i],y[i])
    data.append(l)
  X = [(roll_no, tool1) for roll_no, tool1, _ in data]
  y = [target for _, _, target in data]
  preprocessor = ColumnTransformer(
      transformers=[
          ('roll_no', OneHotEncoder(handle_unknown='ignore'), [0]),
          ('Tool1', OneHotEncoder(handle_unknown='ignore'), [1])
      ],
      remainder='passthrough'
  )
  pipeline = Pipeline(steps=[
      ('preprocessor', preprocessor),
      ('model', DecisionTreeClassifier())
  ])
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
  pipeline.fit(X_train, y_train)
  y_pred = pipeline.predict(X_test)
  kfold=ShuffleSplit(n_splits=10,test_size=0.2,random_state=7)
  scoring='accuracy'
  cv=cross_val_score(pipeline,X_train,y_train,cv=kfold,scoring=scoring)
  #print(cv.mean()*100)
  new_sample = roll_tool1
  predicted_target = pipeline.predict(new_sample)
  return (predicted_target[0])
def three(roll_tool1_tool2):
  print(roll_tool1_tool2)
  d=read_csv('Machine Learning.csv')
  d=DataFrame(d)
  x=d['Roll Number']
  x1=d['Tool1']
  x2=d['Tool2']
  y=d['Tool3']
  data=[]
  for i in range(66):
    l=(x[i],x1[i],x2[i],y[i])
    data.append(l)
  X = [(roll_no, tool1,tool2) for roll_no, tool1,tool2, _ in data]
  y = [target for _, _,_, target in data]
  preprocessor = ColumnTransformer(
      transformers=[
          ('roll_no', OneHotEncoder(handle_unknown='ignore'), [0]),
          ('Tool1', OneHotEncoder(handle_unknown='ignore'), [1]),
          ('Tool2', OneHotEncoder(handle_unknown='ignore'), [2])
      ],
      remainder='passthrough'
  )
  pipeline = Pipeline(steps=[
      ('preprocessor', preprocessor),
      ('model', DecisionTreeClassifier())
  ])
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
  pipeline.fit(X_train, y_train)
  y_pred = pipeline.predict(X_test)
  kfold=ShuffleSplit(n_splits=10,test_size=0.2,random_state=7)
  scoring='accuracy'
  cv=cross_val_score(pipeline,X_train,y_train,cv=kfold,scoring=scoring)
  #print(cv.mean()*100)
  new_sample = roll_tool1_tool2
  predicted_target = pipeline.predict(new_sample)
  return (predicted_target[0])
class MyButton(Button):
    pass
class MyL(FloatLayout):
    popup=Image(source='profile.jpg',size_hint=(0.4,0.9),pos=(25,20))
    label=Label(text='Roll No : ',pos=(10,100))
    label1=Label(text='',pos=(18,0))
    textfield=TextInput(size_hint=(0.37,0.1),pos=(330,210),multiline=False)
    button=Button(text='Confirm',size_hint=(0.1,0.1),pos=(335,30))
    button1=Button(text='Next',size_hint=(0.1,0.1),pos=(520,30))
    img=Image(source="Verified.png",size_hint=(0,0),pos=(350,70))
    tool1,tool2,tool3='','',''
    count,s1,s2,s3,s4=0,'','','',''
    def __init__(self,**kwargs):
        super(MyL,self).__init__(**kwargs)
        Window.bind(on_request_close=self.exit_app)
        self.add_widget(MyL.popup)
        self.add_widget(MyL.label)
        self.add_widget(MyL.textfield)
        self.add_widget(MyL.button)
        self.add_widget(MyL.button1)
        self.add_widget(MyL.img)
        self.add_widget(MyL.label1)
        MyL.button.bind(on_press=self.confirm)
        MyL.button1.bind(on_press=self.next)
    def next(self,instance):
        if (MyL.count==0):
            pass
        elif(MyL.count==1):
            MyL.label.text='Tool 1'
            MyL.label.pos=(1,100)
            MyL.textfield.text=''
            MyL.img.size_hint=(0,0)
            MyL.label1.text=' Suggested : \n '+MyL.tool1
        elif(MyL.count==2):
            MyL.label.text='Tool 2'
            MyL.label.pos=(1,100)
            MyL.textfield.text=''
            MyL.img.size_hint=(0,0)
            MyL.label1.text=' Suggested : \n '+MyL.tool2
        elif(MyL.count==3):
            MyL.label.text='Tool 3'
            MyL.label.pos=(1,100)
            MyL.textfield.text=''
            MyL.img.size_hint=(0,0)
            MyL.label1.text=' Suggested : \n '+MyL.tool3
    def confirm(self,instance):
        MyL.img.source="Verified.png"
        if MyL.count==0:
            MyL.s1=MyL.textfield.text
            if(len(MyL.s1)==8):
                MyL.count=1
                MyL.tool1=(one(int(MyL.s1[len(MyL.s1)-2:])))
                MyL.img.size_hint=(0.3,0.3)
            else:
                pass
        elif MyL.count==1:
            MyL.label1.text=''
            s=MyL.textfield.text
            MyL.s2=s
            print(MyL.tool1,MyL.s2)
            if(MyL.s2!=MyL.tool1):
              MyL.img.source="sad.png"
            if(s!=''):
                MyL.count=2
                MyL.img.size_hint=(0.3,0.3)
                MyL.tool2=two([(MyL.s1,s)])
                MyL.popup.source=s.lower()+".png"
        elif MyL.count==2:
            MyL.label1.text=''
            s=MyL.textfield.text
            if(s!=MyL.tool2):
              MyL.img.source="sad.png"
            if(s!=''):
                MyL.count=3
                MyL.img.size_hint=(0.3,0.3)
                MyL.tool3=three([(MyL.s1,MyL.s2,s)])
                MyL.popup.source=s.lower()+".png"
        elif MyL.count==3:
            MyL.label1.text=''
            s=MyL.textfield.text
            if(s!=MyL.tool3):
              MyL.img.source="sad.png"
            if(s!=''):
                MyL.count=4
                MyL.img.size_hint=(0.3,0.3)
                MyL.s4=s
                MyL.tool3=(MyL.s4)
                MyL.popup.source=s.lower()+".png"
    def exit_app(self,instance):
        App.get_running_app().stop()
        Window.close()
class My(App):
    def build(self):
        return MyL()
if __name__=='__main__':
    My().run()
