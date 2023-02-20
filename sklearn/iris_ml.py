import pandas as pd

'사이킷런 버전 확인'
import sklearn
print(sklearn.__version__)

'붓꽃 예측을 위한 사이킷런 필요 모듈 로딩'
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

'붓꽃 데이터 세트 로드'
iris = load_iris()
iris_data = iris.data # .data = feature data numpy
iris_label = iris.target #.target = label data numpy
print("붗꽃 데이터 세트의 키들: ", iris.keys())

print('iris target값: ', iris_label)
print('iris target명: ', iris.target_names)

'데이터 프레임 변환'
iris_df = pd.DataFrame(data=iris_data, columns=iris.feature_names)
iris_df['label'] = iris.target
print(iris_df.head(3))

'학습 데이터와 테스트 데이터 분리'
X_train, X_test, y_train, y_test = train_test_split(iris_data, iris_label,
                                                    test_size=0.2, random_state=11)

'학습 수행'
dt_ctf = DecisionTreeClassifier(random_state=11) # DTC 객체 생성
dt_ctf.fit(X_train,y_train) # 학습 진행

'테스트 데이터 세트로 예측 수행'
pred = dt_ctf.predict(X_test)

print(pred)

'예측 정확도 평가'
print('에측 정확도: {0:.4f}'.format(accuracy_score(y_test,pred)))

'''
* 키는 보통 data, target, target_names, feature_name,DESCR로 구성
data는 피처의 데이터 세트를 가리킴
target은 분퓨 시 레이블 값, 회귀 일때는 숫자 결과값 데이터 세트
target_name은 개별 레이블의 이름을 나타냄
feature_names는 피처의 이름을 나타냄
DESCR은 데이터 세트에 대한 설명과 각 피지의 설명을 나타냄

print('\n feature_names 의 type:',type(iris.feature_names))
print(' feature_names 의 shape:',len(iris.feature_names))
print(iris.feature_names)
print('\n target_names 의 type:',type(iris.target_names))
print(' feature_names 의 shape:',len(iris.target_names))
print(iris.target_names)
print('\n data 의 type:',type(iris.data))
print(' data 의 shape:',iris.data.shape)
print(iris['data'])
print('\n target 의 type:',type(iris.target))
print(' target 의 shape:',iris.target.shape)
print(iris.target)
'''